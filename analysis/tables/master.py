"""Master tables: every classification run in one table.

Sources (no new runs): the runs.csv of the benchmark (which includes the baselines' FC / DNN / tree logs), the ablation (its own logs only;
the rungs it shares with the benchmark are the benchmark's runs) and the subsets (the independent subsamples; its single-seed regression
re-runs on the QPU subsets are left out), plus the shot-count logs, parsed here. Each run, i.e. (log, model,
resampler), appears once, under the study that wrote the log. Every row carries explicit descriptors
(dataset, subset, input, encoding, backend, shots, readouts, observables, number of features) and the
whole 'Extended metrics' line of its log: the metrics, the confusion matrix, the selected
hyperparameters, the grid size and the timings.

Summaries:
  summary_configurations.csv  per (study, dataset, pipeline, shots, model, resampler): mean, SD and
                              t-based 95 % CI half-width over seeds of every metric; mean confusion matrix;
                              the selected hyperparameters (most frequent value, its count, and the full
                              distribution as JSON); seeds without a result (validation F1 = 0 at every
                              grid point, so the driver logs no test metrics).
  selected_per_seed.csv       per (study, dataset, pipeline, shots, seed): the (model, resampler) with the
                              best validation F1 (the benchmark's select(): ties to the first in resampler order, then
                              model order).
  summary_selected.csv        their mean, SD and CI, the mean confusion matrix and the selection counts.
  resampling.csv              class counts before and after each resampler, from the run manifests.
  regression.csv              checks against every study's own CSVs (see check_*()).

    python analysis/tables/master.py
"""
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "results"
OUT = REV / "tables"
_spec = importlib.util.spec_from_file_location("benchmark_analysis", ROOT / "analysis" / "benchmark" / "analysis.py")
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)

PAPER, TREES, RESAMPLERS = bm.PAPER, bm.TREES, bm.RESAMPLERS
MODEL_ORDER = PAPER + ["DNN"] + TREES
METRICS = ["precision", "recall", "f1", "balanced_acc", "roc_auc", "pr_auc"]
EXTENDED = ["tn", "fp", "fn", "tp", "n_train", "n_test", "grid_size", "fit_data", "score_source",
            "fit_time_s", "grid_time_s", "predict_time_s", "n_params"]
DATASETS = {"30000": ("CARDS_30000", 29601, "full dataset", ""),
            "2500": ("CARDS_2500", 2571, "QPU subset", 7),
            "1000": ("CARDS_1000", 1000, "QPU subset (of CARDS_2500)", 7)}
SHOTS_NAME = re.compile(r"^(detuning|position)_(aquila|emu|fc|model|sv)(?:_s(\d+))?_(\d+)$")
SUBSAMPLE = re.compile(r"^n(2571|1000)s(\d\d)$")
SUBSAMPLE_FIRST_STATE = {"2571": 101, "1000": 201}             # results/subsets/subsets_manifest.json

# --- pipeline descriptors ------------------------------------------------------------------------
ENCODING = {"det": "detuning", "pos": "position"}
ATOMS = {"det": 12, "pos": 13}
BACKEND = {"hw": "statevector emulation", "shot": "shot emulation", "aquila": "Aquila", "sv": "ideal continuous evolution"}
READOUT_T = ["0.5", "1.0", "1.5", "2.0", "2.5"]
CLASSICAL = {
    # id: (family, input, n_features)
    "FC": ("paper classifiers", "12 aggregated features", 12),
    "FC24": ("paper classifiers", "24 preprocessed variables", 24),
    "DNN": ("DNN", "12 aggregated features", 12),
    "DNN24": ("DNN", "24 preprocessed variables", 24),
    "trees12": ("tree models", "12 aggregated features", 12),
    "trees24": ("tree models", "24 preprocessed variables", 24),
    "pca": ("feature map", "11 PCs of the 24 variables", 11),
    "poly2i": ("feature map", "12 features + pairwise products", 78),
    "poly2": ("feature map", "degree-2 polynomial of the 12 features", 90),
    "poly3": ("feature map", "degree-3 polynomial of the 12 features", 454),
    "rff1": ("feature map", "random Fourier features", 390),
    "rff05": ("feature map", "random Fourier features, bandwidth x 0.5", 390),
    "rff2": ("feature map", "random Fourier features, bandwidth x 2", 390),
    "tanh": ("feature map", "random tanh features", 390),
    "CRC": ("CRC", "mean-field reservoir (CRC)", 390),
}
QRC_ID = re.compile(r"^(det|pos)(hw|shot|aquila|sv)(?::([tz]\d+))?$")


def describe(pipeline, shots=None):
    """Explicit descriptors of a pipeline id (benchmark / ablation naming)."""
    if pipeline in CLASSICAL:
        fam, inp, n = CLASSICAL[pipeline]
        ro = "0.5-2.5 us (5)" if pipeline == "CRC" else ""
        return dict(family=fam, input=inp, encoding="", backend="", shots="", readouts=ro,
                    observables="<Z>, <ZZ>" if pipeline == "CRC" else "", n_features=n)
    m = QRC_ID.match(pipeline)
    if not m:
        raise ValueError(f"unknown pipeline {pipeline}")
    enc, be, sub = m.groups()
    sub = sub or "t12345"
    n = ATOMS[enc]
    block = n + n * (n - 1) // 2
    idx = [int(c) - 1 for c in sub[1:]]
    times = [READOUT_T[i] for i in idx]
    ro = f"{times[0]} us" if len(times) == 1 else f"{times[0]}-{times[-1]} us ({len(times)})"
    if be in ("shot", "aquila") and shots is None:
        shots = 240
    return dict(family="QRC", input=f"QRC {ENCODING[enc]}", encoding=ENCODING[enc], backend=BACKEND[be],
                shots=shots if be in ("shot", "aquila") else "", readouts=ro,
                observables="<Z>" if sub[0] == "z" else "<Z>, <ZZ>",
                n_features=len(idx) * (n if sub[0] == "z" else block))


# --- collection ----------------------------------------------------------------------------------
def parse_log(log):
    out = {}
    for line in log.read_text().splitlines():
        if "Extended metrics: " in line:
            d = json.loads(line.split("Extended metrics: ", 1)[1])
            out[(d["model"], d["resampler"])] = d
    return out


def study_of(log):
    return (log if log.is_absolute() else ROOT / log).relative_to(REV).parts[0]


def from_runs_csv(path, keep):
    r = pd.read_csv(path, keep_default_na=False, dtype={"dataset": str}, float_precision="round_trip")
    r = r[keep(r)].copy()
    r["log"] = r.predictions.str.replace(".predictions.csv.gz", ".log", regex=False)
    return r[["dataset", "pipeline", "seed", "model", "resampler", "log"] + METRICS + ["val_f1", "best_params"]]


def collect():
    bench_runs = from_runs_csv(REV / "benchmark" / "runs.csv", lambda r: r.pipeline != "")
    abl_runs = from_runs_csv(REV / "ablation" / "runs.csv",
                        lambda r: r.predictions.str.contains("results/ablation/") & ~r.pipeline.str.endswith(":prefix"))
    sub_runs = from_runs_csv(REV / "subsets" / "runs.csv", lambda r: r.dataset.str.match(SUBSAMPLE.pattern))
    listed = pd.concat([bench_runs, abl_runs, sub_runs], ignore_index=True)
    listed["shots"] = np.nan
    rows = []
    for log in sorted((REV / "shots" / "logs").glob("*.log")):
        m = SHOTS_NAME.match(log.stem)
        if not m:
            continue
        enc, src, shots, seed = m.groups()
        e = enc[:3]
        pipe = {"fc": "FC", "model": f"{e}hw", "sv": f"{e}sv", "emu": f"{e}shot", "aquila": f"{e}aquila"}[src]
        for (model, res), d in parse_log(log).items():
            rows.append(dict(dataset="2500" if enc == "detuning" else "1000", pipeline=pipe, seed=int(seed),
                             model=model, resampler=res, log=str(log), shots=int(shots) if shots else None,
                             **{k: d[k] for k in METRICS + ["val_f1"]}, best_params=json.dumps(d["best_params"])))
    runs = pd.concat([listed, pd.DataFrame(rows)], ignore_index=True)

    # enrich from the logs; the listed metrics must be the logged ones
    ext, mismatch = [], 0
    cache = {}
    for r in runs.itertuples():
        if r.log not in cache:
            cache[r.log] = parse_log(Path(r.log))
        d = cache[r.log][(r.model, r.resampler)]
        mismatch += sum(not np.isclose(getattr(r, k), d[k], rtol=0, atol=0, equal_nan=True)
                        for k in METRICS if d[k] is not None)
        ext.append({k: d.get(k) for k in EXTENDED})
    runs = pd.concat([runs, pd.DataFrame(ext)], axis=1)
    runs["study"] = [study_of(Path(p)) for p in runs.log]
    dup = runs.duplicated(["log", "model", "resampler"]).sum()

    # descriptors
    ds = []
    for r in runs.itertuples():
        if r.dataset in DATASETS:
            name, n, subset, state = DATASETS[r.dataset]
        else:
            size, k = SUBSAMPLE.match(r.dataset).groups()
            name, n = f"S{size}-{k}", int(size)
            subset, state = "independent stratified subsample", SUBSAMPLE_FIRST_STATE[size] + int(k) - 1
        shots = None if pd.isna(r.shots) else int(r.shots)
        ds.append(dict(dataset_name=name, n_samples=n, subset=subset, subset_random_state=state,
                       **describe(r.pipeline, shots)))
    runs = pd.concat([runs.drop(columns="shots"), pd.DataFrame(ds)], axis=1)     # descriptor: 240 for benchmark/ablation
    runs["log"] = [str(Path(p).relative_to(ROOT)) if Path(p).is_absolute() else p for p in runs.log]
    runs["_order"] = runs.resampler.map(RESAMPLERS.index) * 100 + runs.model.map(MODEL_ORDER.index)
    return runs, dict(metric_mismatches=mismatch, duplicate_runs=int(dup), n_logs=len(cache))


# --- summaries -----------------------------------------------------------------------------------
KEY = ["study", "dataset", "pipeline", "shots"]
DESCR = ["dataset_name", "n_samples", "subset", "subset_random_state", "family", "input", "encoding", "backend",
         "readouts", "observables", "n_features"]


def stats(g, row):
    n = len(g)
    for m in METRICS + ["val_f1"]:
        v = g[m].astype(float)
        row |= {f"{m}_mean": v.mean(), f"{m}_sd": v.std(),
                f"{m}_ci95": student_t.ppf(0.975, n - 1) * v.std() / np.sqrt(n) if n > 1 else np.nan}
    for c in ["tn", "fp", "fn", "tp"]:
        row[f"{c}_mean"] = g[c].astype(float).mean()
    row["n_test_mean"] = g.n_test.astype(float).mean()
    return row


def hyper(best_params):
    c = Counter(best_params)
    mode, k = c.most_common(1)[0]
    return dict(params_mode=mode, params_mode_count=k, params_distribution=json.dumps(dict(c.most_common())))


def summarise_configurations(runs):
    all_seeds = runs.groupby(KEY, dropna=False).seed.agg(lambda s: sorted(set(s)))
    rows = []
    for key, g in runs.groupby(KEY + ["model", "resampler"], sort=False, dropna=False):
        expected = all_seeds[key[:len(KEY)]]
        row = dict(zip(KEY + ["model", "resampler"], key), **g.iloc[0][DESCR].to_dict(), n_seeds=len(g),
                   missing_seeds=" ".join(str(s) for s in expected if s not in set(g.seed)))
        row = stats(g, row) | hyper(g.best_params)
        row |= dict(grid_size=g.grid_size.iloc[0], fit_data=g.fit_data.iloc[0], score_source=g.score_source.iloc[0],
                    fit_time_s_median=g.fit_time_s.median(), grid_time_s_median=g.grid_time_s.median(),
                    n_params_mean=pd.to_numeric(g.n_params, errors="coerce").mean())
        rows.append(row)
    # a (model, resampler) of a pipeline without a result in any seed is listed with n_seeds = 0
    combos = runs.groupby(KEY, dropna=False).apply(
        lambda g: {(m, r) for m in set(g.model) for r in set(g.resampler)}, include_groups=False)
    have = {tuple(r[k] for k in KEY + ["model", "resampler"]) for r in rows}
    for key, cs in combos.items():
        for m, r in sorted(cs):
            if key + (m, r) not in have:
                first = next(x for x in rows if tuple(x[k] for k in KEY) == key)
                rows.append(dict(zip(KEY + ["model", "resampler"], key + (m, r)), **{d: first[d] for d in DESCR},
                                 n_seeds=0, missing_seeds=" ".join(map(str, all_seeds[key]))))
    return pd.DataFrame(rows)


def select(runs):
    r = runs.sort_values(KEY + ["seed", "_order"], na_position="first")
    return r.loc[r.groupby(KEY + ["seed"], sort=False, dropna=False).val_f1.idxmax()].reset_index(drop=True)


def summarise_selected(sel):
    rows = []
    for key, g in sel.groupby(KEY, sort=False, dropna=False):
        row = dict(zip(KEY, key), **g.iloc[0][DESCR].to_dict(), n_seeds=len(g))
        row = stats(g, row)
        cfg = Counter(f"{m}/{r}" for m, r in zip(g.model, g.resampler))
        row["selected"] = json.dumps(dict(cfg.most_common()))
        row["selected_models"] = json.dumps(dict(Counter(g.model).most_common()))
        rows.append(row)
    return pd.DataFrame(rows)


def resampling_table():
    """Class counts (majority, minority) of the training set before and after each resampler, from the
    manifests of the FC runs (the same splits as every pipeline of a dataset)."""
    rows = []
    for log in sorted((REV / "baselines" / "logs").glob("*_fc12_*.manifest.json")):
        m = json.loads(log.read_text())
        ds = re.match(r"^c(\d+)_", log.name).group(1)
        for res, d in m["resamplers"].items():
            rows.append(dict(dataset=ds, seed=m["args"]["seed"], resampler=res, n0_before=d["n_before"][0],
                             n1_before=d["n_before"][1], n0_after=d["n_after"][0], n1_after=d["n_after"][1],
                             params=json.dumps({k: v for k, v in d.items()
                                                if k not in ("time_s", "n_before", "n_after", "random_state")}),
                             random_state=d["random_state"]))
    # CARDS_30000 has two FC logs per seed (trees / the rest): same split, so the same counts
    df = pd.DataFrame(rows).drop_duplicates()
    if df.duplicated(["dataset", "seed", "resampler"]).any():
        raise SystemExit("resampling: runs with the same split report different class counts")
    return df.reset_index(drop=True)


# --- regression against the studies' own outputs ------------------------------------------------
def _cmp(name, mine, theirs, on, cols):
    j = mine.merge(theirs, on=on, how="outer", suffixes=("", "_ref"), indicator=True)
    both = j[j._merge == "both"]
    same = np.ones(len(both), bool)
    for c in cols:
        a, b = both[c], both[f"{c}_ref"]
        same &= (a == b).to_numpy() if a.dtype == object else np.isclose(a.astype(float), b.astype(float),
                                                                          rtol=0, atol=1e-12, equal_nan=True)
    return dict(check=name, rows_mine=int((j._merge != "right_only").sum()), rows_ref=int((j._merge != "left_only").sum()),
                matched=len(both), identical=int(same.sum()))


def regression(runs, sel, cfg):
    out = []
    ref = pd.read_csv(REV / "benchmark" / "selected_per_seed.csv", keep_default_na=False, dtype={"dataset": str})
    mine = sel[sel.study.isin(["baselines", "benchmark"])]
    out.append(_cmp("benchmark selected_per_seed", mine, ref, ["dataset", "pipeline", "seed"], ["model", "resampler", "f1", "pr_auc"]))
    ref = pd.read_csv(REV / "benchmark" / "summary_model_resampler.csv", keep_default_na=False, dtype={"dataset": str})
    mine = cfg[cfg.study.isin(["baselines", "benchmark"])]
    out.append(_cmp("benchmark summary_model_resampler", mine, ref, ["dataset", "pipeline", "model", "resampler"],
                    ["f1_mean", "f1_sd", "pr_auc_mean", "roc_auc_mean", "precision_mean", "recall_mean"]))
    ref = pd.read_csv(REV / "ablation" / "selected_per_seed.csv", keep_default_na=False, dtype={"dataset": str})
    ref = ref[~ref.pipeline.str.endswith(":prefix")].copy()
    ref["pipeline"] = ref.pipeline.str.replace(":t12345", "", regex=False)
    # The ablation selects among the paper's six classifiers only; the baselines' FC24 / FC logs also hold the DNN and trees,
    # which are separate pipelines here, so FC / FC24 / CRC / full QRC vectors compare directly with the benchmark's
    abl_sel = sel[sel.study.isin(["baselines", "benchmark", "ablation"]) & sel.pipeline.isin(set(ref.pipeline))]
    out.append(_cmp("ablation selected_per_seed (no prefix rows)", abl_sel, ref, ["dataset", "pipeline", "seed"],
                    ["model", "resampler", "f1", "pr_auc"]))
    ref = pd.read_csv(REV / "subsets" / "selected_per_seed.csv", keep_default_na=False, dtype={"dataset": str})
    ref = ref[ref.dataset.str.match(SUBSAMPLE.pattern)]
    out.append(_cmp("subsets selected_per_seed (subsamples)", sel[sel.study == "subsets"], ref, ["dataset", "pipeline", "seed"],
                    ["model", "resampler", "f1", "pr_auc"]))
    ref = pd.read_csv(REV / "shots" / "f1_selected.csv", keep_default_na=False)
    mine = sel[sel.study == "shots"].copy()
    src = {"hw": "model", "sv": "sv", "shot": "emu", "aquila": "aquila"}
    mine["source"] = ["fc" if p == "FC" else src[QRC_ID.match(p).group(2)] for p in mine.pipeline]
    mine["encoding"] = np.where(mine.dataset == "2500", "detuning", "position")
    for d in (mine, ref):           # the shot analysis writes shots = 0 for the fc / model / sv rows, here they are blank
        d["shots_n"] = pd.to_numeric(d.shots, errors="coerce").fillna(0).astype(int)
    out.append(_cmp("shots f1_selected", mine.drop(columns="shots"), ref.drop(columns="shots"),
                    ["encoding", "source", "shots_n", "seed"], ["model", "resampler", "f1"]))
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs, info = collect()
    print("collection:", info)
    cols = (["study", "dataset", "pipeline", "shots"] + DESCR + ["seed", "model", "resampler", "val_f1"] + METRICS
            + ["tn", "fp", "fn", "tp", "n_train", "n_test", "best_params", "grid_size", "fit_data", "score_source",
               "fit_time_s", "grid_time_s", "predict_time_s", "n_params", "log"])
    runs[cols].to_csv(OUT / "runs_master.csv.gz", index=False)
    cfg = summarise_configurations(runs)
    cfg.to_csv(OUT / "summary_configurations.csv", index=False)
    sel = select(runs)
    sel[cols].to_csv(OUT / "selected_per_seed.csv", index=False)
    ssum = summarise_selected(sel)
    ssum.to_csv(OUT / "summary_selected.csv", index=False)
    resampling_table().to_csv(OUT / "resampling.csv", index=False)
    reg = regression(runs, sel, cfg)
    reg = pd.concat([pd.DataFrame([dict(check="listed metrics == logged metrics", rows_mine=len(runs),
                                        rows_ref=len(runs), matched=len(runs),
                                        identical=len(runs) if info["metric_mismatches"] == 0 else -1),
                                   dict(check="duplicate (log, model, resampler)", rows_mine=info["duplicate_runs"],
                                        rows_ref=0, matched=0, identical=0)]), reg], ignore_index=True)
    reg.to_csv(OUT / "regression.csv", index=False)
    print(f"{len(runs)} runs from {info['n_logs']} logs; {len(cfg)} configurations; {len(ssum)} selected pipelines")
    print(runs.groupby("study").size().to_string())
    print("\nregression\n", reg.to_string())
    short = cfg[cfg.n_seeds < cfg.groupby(KEY, dropna=False).n_seeds.transform("max")]
    print(f"\nconfigurations with fewer seeds than their pipeline: {len(short)}")
    print(short.groupby(["study", "model"]).size().to_string())


if __name__ == "__main__":
    main()
