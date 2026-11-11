"""Training times for the paper's timing table (CARDS_30000), from the runs of run_timing.sh.

Quantity, as for the other rows of the table (table_rows_from_logs.py): time of the
hyperparameter search of one model on one resampled training set divided by its number of grid points, i.e.
the mean time per configuration (fit + validation prediction). Here grid_time_s / grid_size of the extended
metrics; mean and sample SD over the 10 CARDS_30000 split seeds. The time of one fit of the
validation-selected configuration (fit_time_s) is reported alongside.

"Best tree model": (a) per seed and resampler, the tree model (DTR RFR ET HGB, 24 variables) with the best
validation F1, ties in that order; (b) RFR alone, the model most often selected over all resamplers (benchmark).

Checks: every timing run reproduces the selection, validation F1, test F1 and test predictions of the same
model, resampler and seed in the baselines (trees, DNN) or the benchmark (KNN SGD LM P LSVM); NB is not in either (they use NBT).

Writes results/timing/{runs,summary,best_tree,checks}.csv and table.md.

    python analysis/timing/timing.py [--logs DIR] [--pilot DIR] [--single DIR] [--out DIR]
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "timing"
_spec = importlib.util.spec_from_file_location("runs", ROOT / "result_analysis" / "runs.py")
runs_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runs_mod)

RESAMPLERS = ["None", "SMOTE", "kSMOTE", "CC"]
TREES = ["DTR", "RFR", "ET", "HGB"]
ROWS = ["KNN", "SGD", "NB", "LM", "P", "LSVM", "DNN"]
# The submitted table (s, mean ± SD over 100 timing runs; QRC features, DNN on the 12 features)
PUBLISHED = {
    "KNN": [(0.12, 0.01), (0.23, 0.01), (0.24, 0.11), (0.10, 0.01)],
    "SGD": [(2.86, 0.34), (3.43, 0.40), (2.82, 0.31), (1.22, 0.15)],
    "NB": [(0.11, 0.01), (0.23, 0.01), (0.20, 0.01), (0.10, 0.01)],
    "LM": [(0.14, 0.01), (0.31, 0.02), (0.30, 0.10), (0.08, 0.01)],
    "P": [(0.18, 0.02), (0.28, 0.02), (0.27, 0.01), (0.09, 0.01)],
    "DNN": [(26.52, 0.26), (43.32, 2.77), (42.97, 1.77), (13.62, 0.62)],
    # submitted SVM row: SVC (libsvm, linear kernel), 8 grid points divided by 4; LinearSVC replaces it
    "LSVM": [(1515.93, 397.65), (4359.86, 1076.96), (3976.44, 1077.47), (150.71, 83.96)],
}
# Divisor the submitted script used where it differs from the grid size (SGD has 19 grid points)
SUBMITTED_DIVISOR = {"SGD": 10}


def load(log_dir):
    r = runs_mod.load_runs(str(log_dir))
    r = r[r.status == "completed"].rename(columns={"resampling": "resampler", "F1-score": "f1"})
    r["input"] = r.run.str.split("_").str[1]
    for c in ["grid_time_s", "grid_size", "fit_time_s", "val_f1", "f1", "n_train"]:
        r[c] = pd.to_numeric(r[c])
    r["per_config_s"] = r.grid_time_s / r.grid_size
    return r[["run", "input", "model", "seed", "resampler", "grid_size", "grid_time_s", "per_config_s",
              "fit_time_s", "n_train", "val_f1", "f1", "best_params", "git_commit", "git_dirty"]]


def summarise(r, by):
    g = r.groupby(by)
    s = g.agg(n=("seed", "size"), per_config_mean=("per_config_s", "mean"), per_config_sd=("per_config_s", "std"),
              fit_selected_mean=("fit_time_s", "mean"), fit_selected_sd=("fit_time_s", "std"),
              grid_mean=("grid_time_s", "mean"), grid_size=("grid_size", "first"), n_train=("n_train", "mean"))
    return s.reset_index()


def best_tree(r):
    t = r[r.model.isin(TREES)].copy()
    t["_o"] = t.model.map(TREES.index)
    t = t.sort_values(["seed", "resampler", "_o"])
    return t.loc[t.groupby(["seed", "resampler"]).val_f1.idxmax()].drop(columns="_o")


def checks(r, log_dir):
    """Same selection and test predictions as the baseline / benchmark runs of the same model, resampler and seed."""
    rows = []
    for _, x in r.iterrows():
        if x.model in TREES:
            ref = ROOT / "results/baselines/logs" / f"c30000_fc24_trees_{x.seed}"
        elif x.model == "DNN":
            ref = ROOT / "results/baselines/logs" / f"c30000_fc12_rest_{x.seed}"
        elif x.model in ("KNN", "SGD", "LM", "P", "LSVM"):
            ref = ROOT / "results/benchmark/logs" / f"c30000_dethw_{x.model}_{x.seed}"
        else:
            continue
        old = pd.DataFrame(runs_mod.parse_log(f"{ref}.log")).rename(columns={"resampling": "resampler", "F1-score": "f1"})
        old = old[(old.model == x.model) & (old.resampler == x.resampler)].iloc[0]
        p_new = runs_mod.load_predictions(Path(log_dir) / f"{x.run}.predictions.csv.gz")
        p_old = runs_mod.load_predictions(f"{ref}.predictions.csv.gz")
        k = ["resampler", "model", "ID"]
        p = p_new.merge(p_old, on=k, suffixes=("", "_old"))
        p = p[(p.model == x.model) & (p.resampler == x.resampler)]
        rows.append(dict(run=x.run, model=x.model, seed=x.seed, resampler=x.resampler, reference=ref.name,
                         same_params=runs_mod.best_params(x.best_params) == runs_mod.best_params(old.best_params),
                         same_val_f1=x.val_f1 == old.val_f1, same_f1=x.f1 == old.f1,
                         n_test=len(p), same_predictions=bool((p.y_pred == p.y_pred_old).all()) and len(p) > 0,
                         max_score_diff=float((p.score - p.score_old).abs().max()) if len(p) else np.nan))
    return pd.DataFrame(rows)


def cells(d, model, col, scale=1.0):
    """One table row: mean ± SD of `col` per resampler ("—" where the run is missing)."""
    out = []
    for x in RESAMPLERS:
        g = d[(d.resampler == x) & ((d.model == model) if model else True)]
        out.append(f"{g[col + '_mean'].iloc[0] * scale:.2f} ± {g[col + '_sd'].iloc[0] * scale:.2f}" if len(g) else "—")
    return " | ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(OUT / "logs"))
    ap.add_argument("--pilot", default=str(OUT / "pilot_t27"), help="pilot logs, same settings as the main runs (seed 8)")
    ap.add_argument("--single", default=str(OUT / "pilot"), help="pilot logs on one core per process (seed 8)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = load(a.logs)
    r.to_csv(out / "runs.csv", index=False)
    s = summarise(r, ["model", "resampler"])
    s.to_csv(out / "summary.csv", index=False)
    b = best_tree(r)
    b.to_csv(out / "best_tree.csv", index=False)
    bs = summarise(b, ["resampler"])
    c = checks(r, a.logs)
    c.to_csv(out / "checks.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"runs: {len(r)} ({r.run.nunique()} logs), seeds {sorted(r.seed.unique())}, commits {sorted(r.git_commit.unique())}, "
          f"dirty {r.git_dirty.any()}")
    print("\nPer model and resampler (s): per-configuration time and one fit of the selected configuration")
    print(s.round(3).to_string(index=False))
    print("\nBest tree model per seed and resampler (validation-selected among DTR RFR ET HGB):")
    print(pd.crosstab(b.model, b.resampler).reindex(columns=RESAMPLERS).to_string())
    print(bs.round(3).to_string(index=False))
    ok = ["same_params", "same_val_f1", "same_f1", "same_predictions"]
    print(f"\nChecks against baseline/benchmark: {len(c)} runs; " + ", ".join(f"{k} {c[k].all()}" for k in ok)
          + f", max |score diff| {c.max_score_diff.max():.2e}")
    if not c[ok].all().all():
        print(c[~c[ok].all(axis=1)].to_string(index=False))

    head = ["| Classifier | " + " | ".join(RESAMPLERS) + " |", "|---" * 5 + "|"]
    lines = ["Time per hyperparameter configuration (s), mean ± SD over seeds:", ""] + head
    for m in [x for x in ROWS if x in set(s.model)]:
        lines.append(f"| {m} (published{', SVC' if m == 'LSVM' else ''}) | " + " | ".join(f"{u:.2f} ± {v:.2f}" for u, v in PUBLISHED[m]) + " |")
        lines.append(f"| {m} (this machine) | {cells(s, m, 'per_config')} |")
        if m in SUBMITTED_DIVISOR:
            f = s[s.model == m].grid_size.iloc[0] / SUBMITTED_DIVISOR[m]
            lines.append(f"| {m} (this machine, grid time ÷ {SUBMITTED_DIVISOR[m]} as submitted) | {cells(s, m, 'per_config', f)} |")
    for m in [x for x in TREES if x in set(s.model)]:
        lines.append(f"| {m}, 24 variables | {cells(s, m, 'per_config')} |")
    lines.append(f"| Best tree model (validation-selected per seed) | {cells(bs, None, 'per_config')} |")
    lines += ["", "One fit of the validation-selected configuration (s), mean ± SD over seeds:", ""] + head
    for m in [x for x in ROWS + TREES if x in set(s.model)]:
        lines.append(f"| {m} | {cells(s, m, 'fit_selected')} |")
    lines.append(f"| Best tree model (validation-selected per seed) | {cells(bs, None, 'fit_selected')} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    for label, d in [("repeat of seed 8 under the same conditions / main run", a.pilot),
                     ("seed 8 on one core (10 single-core processes) / main run", a.single)]:
        if Path(d).exists() and any(Path(d).glob("*.log")):
            p = load(d)
            j = p.merge(r, on=["model", "seed", "resampler"], suffixes=("_other", ""))
            j["ratio"] = j.per_config_s_other / j.per_config_s
            print(f"\nPer-configuration time, {label}:")
            print(j.groupby("model").ratio.agg(["min", "median", "max"]).round(2).to_string())

if __name__ == "__main__":
    main()
