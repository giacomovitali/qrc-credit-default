"""Benchmark analysis: every pipeline over 10 seeds (CARDS_2500/1000: 1-10; CARDS_30000: the pre-registered
split seeds, preprocessing/split_seeds.py), pre-registered paired tests, test-set bootstrap, and the
regenerated Figs 6, 9 and 12.

Pipelines. From run_classification.sh: CRC, QRC-hw (noiseless hardware-protocol emulation),
QRC-shot (240 shots) and QRC-Aquila, detuning and position. From the baselines (same splits and seeds): FC (the
paper's six classifiers on the 12 features), DNN (12 features) and the best strong baseline (tree models
on the 24 variables); FC-24, DNN-24 and trees-12 are kept as descriptive rows.

Selection is on validation only: per (dataset, pipeline, seed) the (model, resampler) with the best
validation F1 among the pipeline's models (ties: resampler order, then model order = log order).

Tests (fixed before any benchmark result): paired Wilcoxon signed-rank over seeds on the selected test F1,
two-sided, Holm-corrected within each dataset (p_holm); p_holm_all corrects over all tests at once.
Bootstrap: each dataset's first seed (1; CARDS_30000: 8), the selected configuration of each pipeline, 1,000 test-set resamples (percentile
95 % CI), plus paired resamples of the tested differences.

Writes results/benchmark/{runs,selected_per_seed,summary_selected,summary_model_resampler,tests,
descriptive,bootstrap_first_seed,bootstrap_diff_first_seed}.csv and figures/. Read the CSVs with keep_default_na=False.

    python analysis/benchmark/analysis.py
"""
import os
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "preprocessing"))
from split_seeds import SEEDS, check_seeds  # noqa: E402  per-dataset split seeds

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "benchmark"
BASELINE_LOGS = ROOT / "results" / "baselines" / "logs"
FIG = OUT / "figures"
BENCHMARK_NAME = re.compile(r"^c(30000|2500|1000)_(crc|dethw|detshot|detaquila|poshw|posshot|posaquila)(?:_[A-Z]+)?_(\d+)$")
BASELINE_NAME = re.compile(r"^c(30000|2500|1000)_(fc12|fc24)_(trees|rest|all)_(\d+)$")
PAPER = ["KNN", "SGD", "NBT", "LM", "P", "LSVM"]
TREES = ["DTR", "RFR", "ET", "HGB"]
RESAMPLERS = ["None", "SMOTE", "kSMOTE", "CC"]
# pipeline -> (label, colour); bar series in fixed categorical order (dataviz reference palette)
PIPELINES = {
    "FC": ("FC", "#2a78d6"),
    "CRC": ("CRC", "#eb6834"),
    "dethw": ("QRC detuning, statevector", "#1baf7a"),
    "detshot": ("QRC detuning, 240 shots", "#eda100"),
    "detaquila": ("QRC detuning, Aquila", "#e87ba4"),
    "poshw": ("QRC position, statevector", "#008300"),
    "posshot": ("QRC position, 240 shots", "#6250d6"),
    "posaquila": ("QRC position, Aquila", "#e34948"),
    "DNN": ("DNN, 12 features", "#0b0b0b"),
    "trees24": ("Best tree model, 24 variables", "#52514e"),
    "FC24": ("Paper classifiers, 24 variables", None),
    "DNN24": ("DNN, 24 variables", None),
    "trees12": ("Best tree model, 12 features", None),
}
DATASETS = [("30000", "CARDS_30000"), ("2500", "CARDS_2500"), ("1000", "CARDS_1000")]
# pre-registered comparisons (A - B on test F1), per dataset
TESTS = {
    "30000": [("dethw", b) for b in ["FC", "CRC", "trees24", "DNN"]],
    "2500": [("dethw", b) for b in ["FC", "CRC", "trees24", "DNN"]]
            + [("detaquila", "dethw"), ("detaquila", "detshot")],
    "1000": [(f"{e}hw", b) for e in ["det", "pos"] for b in ["FC", "CRC", "trees24", "DNN"]]
            + [(f"{e}aquila", f"{e}{b}") for e in ["det", "pos"] for b in ["hw", "shot"]]
            + [("detaquila", "posaquila")],
}
METRICS = ["f1", "precision", "recall", "roc_auc", "pr_auc", "balanced_acc"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 4,
    "pdf.fonttype": 42,
})
N_BOOT, BOOT_SEED = 1000, 2026        # bootstrap run: the dataset's first seed, SEEDS[ds][0]


def read_log(log, **kw):
    rows = []
    for line in log.read_text().splitlines():
        if "Extended metrics: " in line:
            d = json.loads(line.split("Extended metrics: ", 1)[1])
            rows.append(dict(**kw, model=d["model"], resampler=d["resampler"], val_f1=d["val_f1"],
                             **{m: d[m] for m in METRICS}, n_test=d["n_test"], n_params=d["n_params"],
                             best_params=json.dumps(d["best_params"]),
                             predictions=os.path.relpath(log.with_suffix(".predictions.csv.gz"), ROOT)))
    return rows


def load_runs():
    rows = []
    for log in sorted((OUT / "logs").glob("*.log")):
        m = BENCHMARK_NAME.match(log.stem)
        if m and log.with_suffix(".predictions.csv.gz").exists():
            ds, pipe, seed = m.groups()
            rows += read_log(log, dataset=ds, pipeline="CRC" if pipe == "crc" else pipe, seed=int(seed))
    for log in sorted(BASELINE_LOGS.glob("*.log")):
        m = BASELINE_NAME.match(log.stem)
        if m and log.with_suffix(".predictions.csv.gz").exists():
            ds, inp, _, seed = m.groups()
            for r in read_log(log, dataset=ds, seed=int(seed)):
                k = "12" if inp == "fc12" else "24"
                r["pipeline"] = ({"12": "FC", "24": "FC24"}[k] if r["model"] in PAPER else
                                 {"12": "DNN", "24": "DNN24"}[k] if r["model"] == "DNN" else f"trees{k}")
                rows.append(r)
    df = pd.DataFrame(rows)
    df["_order"] = df.resampler.map(RESAMPLERS.index) * 100 + df.model.map((PAPER + ["DNN"] + TREES).index)
    return df


def select(runs):
    """Per (dataset, pipeline, seed), the (model, resampler) with the best validation F1."""
    r = runs.sort_values(["dataset", "pipeline", "seed", "_order"])
    return r.loc[r.groupby(["dataset", "pipeline", "seed"], sort=False).val_f1.idxmax()].reset_index(drop=True)


def summarise(df, by):
    rows = []
    for key, g in df.groupby(by):
        n = len(g)
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)), n=n)
        for m in METRICS:
            mean, sd = g[m].mean(), g[m].std()
            row |= {f"{m}_mean": mean, f"{m}_sd": sd,
                    f"{m}_ci95": student_t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan}
        row["val_f1_mean"] = g.val_f1.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def tests(sel):
    rows = []
    for ds, pairs in TESTS.items():
        s = sel[sel.dataset == ds]
        for a, b in pairs:
            x = s[s.pipeline == a].set_index("seed")
            y = s[s.pipeline == b].set_index("seed")
            seeds = x.index.intersection(y.index)
            if len(seeds) < 2:
                continue
            row = dict(dataset=ds, a=a, b=b, comparison=f"{PIPELINES[a][0]} - {PIPELINES[b][0]}", n=len(seeds))
            for m in ["f1", "pr_auc", "roc_auc"]:
                d = x.loc[seeds, m] - y.loc[seeds, m]
                row |= {f"{m}_a": x.loc[seeds, m].mean(), f"{m}_b": y.loc[seeds, m].mean(),
                        f"{m}_diff": d.mean(), f"{m}_diff_sd": d.std(),
                        f"{m}_diff_ci95": student_t.ppf(0.975, len(d) - 1) * d.std() / np.sqrt(len(d))}
                if m == "f1":
                    row |= {"f1_a_wins": int((d > 0).sum()),
                            "p": wilcoxon(x.loc[seeds, m], y.loc[seeds, m]).pvalue if (d != 0).any() else 1.0}
                else:
                    row[f"{m}_p"] = wilcoxon(x.loc[seeds, m], y.loc[seeds, m]).pvalue if (d != 0).any() else 1.0
            rows.append(row)
    t = pd.DataFrame(rows)
    t["p_holm"] = t.groupby("dataset").p.transform(lambda p: pd.Series(holm(p), index=p.index))
    t["p_holm_all"] = holm(t.p)
    return t


def descriptive(sel, runs):
    """Checks asked for by the manuscript plan, outside the pre-registered family (uncorrected):
    shot noise (shot - hw), detuning - position in emulation and on Aquila (selected, and averaged over
    all 24 configurations as in the earlier letter draft), and whether the Aquila gap to the emulation is
    smaller on CARDS_2500 than on CARDS_1000 (Fig 12 claim; the splits are not nested, so Mann-Whitney)."""
    from scipy.stats import mannwhitneyu
    rows = []

    def paired_row(ds, a, b, x, y, what):
        d = (x - y.reindex(x.index)).dropna()
        rows.append(dict(dataset=ds, comparison=f"{PIPELINES[a][0]} - {PIPELINES[b][0]}", selection=what, n=len(d),
                         diff=d.mean(), diff_sd=d.std(), a_wins=int((d > 0).sum()),
                         p=wilcoxon(d).pvalue if (d != 0).any() else 1.0, test="Wilcoxon"))

    allcfg = runs.groupby(["dataset", "pipeline", "seed"]).f1.mean()
    pairs = [(ds, f"{e}shot", f"{e}hw") for ds, e in [("30000", "det"), ("2500", "det"), ("1000", "det"), ("1000", "pos")]]
    pairs += [("1000", f"det{b}", f"pos{b}") for b in ["hw", "shot", "aquila"]]
    for ds, a, b in pairs:
        s = sel[sel.dataset == ds].set_index(["pipeline", "seed"]).f1
        if a in s.index.get_level_values(0) and b in s.index.get_level_values(0):
            paired_row(ds, a, b, s[a], s[b], "selected on validation")
            paired_row(ds, a, b, allcfg[ds, a], allcfg[ds, b], "mean over the 24 configurations")
    s = sel.set_index(["dataset", "pipeline", "seed"]).f1
    for ref in ["hw", "shot"]:
        try:
            g25, g10 = s["2500", "detaquila"] - s["2500", f"det{ref}"], s["1000", "detaquila"] - s["1000", f"det{ref}"]
        except KeyError:
            continue
        rows.append(dict(dataset="2500 vs 1000", comparison=f"detuning gap Aquila - {ref}: CARDS_2500 minus CARDS_1000",
                         selection="selected on validation", n=len(g25), diff=g25.mean() - g10.mean(),
                         diff_sd=np.nan, a_wins=np.nan, p=mannwhitneyu(g25, g10).pvalue, test="Mann-Whitney"))
    # the paper's NB + CC paragraph: FC with NB and CC against the DNN with CC and the selected DNN
    cfg = runs.set_index(["dataset", "pipeline", "model", "resampler", "seed"]).f1
    for ds, _ in DATASETS:
        nb = cfg[ds, "FC", "NBT", "CC"]
        for ref, what in [(cfg[ds, "DNN", "DNN", "CC"], "DNN with CC"), (s[ds, "DNN"], "DNN selected on validation")]:
            d = nb - ref.reindex(nb.index)
            rows.append(dict(dataset=ds, comparison=f"FC NB with CC - {what}", selection="fixed configuration",
                             n=len(d), diff=d.mean(), diff_sd=d.std(), a_wins=int((d > 0).sum()),
                             p=wilcoxon(d).pvalue, test="Wilcoxon"))
    return pd.DataFrame(rows)


def test_predictions(row):
    p = pd.read_csv(row.predictions, keep_default_na=False, na_values={"score": ["", "nan", "NaN"]})
    p = p[(p.model == row.model) & (p.resampler == row.resampler)].sort_values("ID")
    if len(p) != row.n_test:
        raise ValueError(f"{row.predictions}: {len(p)} predictions for {row.model}/{row.resampler}, expected {row.n_test}")
    return p.set_index("ID")


def boot_metrics(y, pred, score, idx):
    out = {"f1": [], "pr_auc": [], "roc_auc": []}
    for i in idx:
        yy = y[i]
        out["f1"].append(f1_score(yy, pred[i]))
        out["pr_auc"].append(average_precision_score(yy, score[i]))
        out["roc_auc"].append(roc_auc_score(yy, score[i]))
    return {k: np.array(v) for k, v in out.items()}


def bootstrap(sel):
    """First-seed run: percentile CIs per pipeline and for each tested difference (paired: same resamples)."""
    rows, diff_rows = [], []
    for ds, _ in DATASETS:
        s = sel[(sel.dataset == ds) & (sel.seed == SEEDS[ds][0])].set_index("pipeline")
        preds = {p: test_predictions(r) for p, r in s.iterrows()}
        ids = next(iter(preds.values())).index
        for p, q in preds.items():
            if not q.index.equals(ids):
                raise ValueError(f"{ds}: pipeline {p} has a different test set")
        y = next(iter(preds.values())).y_true.to_numpy()
        idx = np.random.default_rng(BOOT_SEED).integers(0, len(y), size=(N_BOOT, len(y)))
        boot = {}
        for p, q in preds.items():
            boot[p] = boot_metrics(y, q.y_pred.to_numpy(), q.score.to_numpy(), idx)
            row = dict(dataset=ds, seed=SEEDS[ds][0], pipeline=p, model=s.loc[p, "model"], resampler=s.loc[p, "resampler"], n_test=len(y))
            for m, v in boot[p].items():
                row |= {m: s.loc[p, m], f"{m}_lo": np.percentile(v, 2.5), f"{m}_hi": np.percentile(v, 97.5)}
            rows.append(row)
        for a, b in TESTS[ds]:
            if a in boot and b in boot:
                row = dict(dataset=ds, seed=SEEDS[ds][0], a=a, b=b, comparison=f"{PIPELINES[a][0]} - {PIPELINES[b][0]}")
                for m in ["f1", "pr_auc"]:
                    d = boot[a][m] - boot[b][m]
                    row |= {f"{m}_diff": s.loc[a, m] - s.loc[b, m], f"{m}_lo": np.percentile(d, 2.5),
                            f"{m}_hi": np.percentile(d, 97.5), f"{m}_p_le0": (d <= 0).mean()}
                diff_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(diff_rows)


def figure_benchmark(per_cfg, sel_sum, ds, title, pipes):
    """Benchmark figure (Figs 6, 8, 9): one panel per resampler, the six classifiers on x, one bar per
    pipeline (mean ± SD over seeds); DNN mean ± SD as a band; the best tree model (validation-selected,
    24 variables) as a dotted line. No test-set maximum line (that was selection on test)."""
    pipes = [p for p in pipes if ((per_cfg.dataset == ds) & (per_cfg.pipeline == p)).any()]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.7), sharey=True)
    w = 0.84 / len(pipes)
    x = np.arange(len(PAPER))
    for ax, res in zip(axes, RESAMPLERS):
        d = per_cfg[(per_cfg.dataset == ds) & (per_cfg.resampler == res)]
        for k, p in enumerate(pipes):
            a = d[d.pipeline == p].set_index("model").reindex(PAPER)
            ax.bar(x - 0.42 + w * (k + 0.5), a.f1_mean, w * 0.9, yerr=a.f1_sd, color=PIPELINES[p][1],
                   error_kw=dict(elinewidth=0.6, ecolor=INK2, capsize=0), label=PIPELINES[p][0])
        dnn = d[d.pipeline == "DNN"]
        if len(dnn):
            m, sd = dnn.f1_mean.iloc[0], dnn.f1_sd.iloc[0]
            ax.axhspan(m - sd, m + sd, color=INK, alpha=0.10, lw=0)
            ax.axhline(m, color=INK, ls="--", lw=1, label="DNN (mean ± SD)")
        tr = sel_sum[(sel_sum.dataset == ds) & (sel_sum.pipeline == "trees24")]
        if len(tr):
            ax.axhline(tr.f1_mean.iloc[0], color=INK2, ls=":", lw=1.2, label=PIPELINES["trees24"][0] + ", selected")
        ax.set_xticks(x, ["NB" if m == "NBT" else m for m in PAPER], fontsize=7)
        ax.set_title(f"Resampling = {res}", fontsize=8, color=INK)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 0.7)
    axes[0].set_ylabel("test F1 (mean ± SD, 10 seeds)")
    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=8.5, color=INK, y=0.99)
    fig.legend(h, l, loc="lower center", ncol=min(4, len(l)), fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07 * ((len(l) + 3) // 4), 1, 0.97))
    stem = f"benchmark_{title}"
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=200, bbox_inches="tight"), plt.close(fig)


def figure_selected(sel):
    """Headline: validation-selected test F1 per pipeline and dataset, every seed plus mean ± SD."""
    order = ["trees24", "DNN", "FC", "CRC", "dethw", "detshot", "detaquila", "poshw", "posshot", "posaquila"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, (ds, title) in zip(axes, DATASETS):
        s = sel[sel.dataset == ds]
        for yi, p in zip(y, order):
            v = s[s.pipeline == p].f1
            if not len(v):
                continue
            c = PIPELINES[p][1]
            ax.plot(v, np.full(len(v), yi) + np.random.default_rng(0).uniform(-0.18, 0.18, len(v)), "o", ms=2.5,
                    color=c, alpha=0.35, mec="none")
            ax.errorbar(v.mean(), yi, xerr=v.std(), fmt="o", color=c, ms=5, elinewidth=1.2, capsize=0,
                        mec="#fcfcfb", mew=1)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_xlabel("test F1, selected on validation")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [PIPELINES[p][0] for p in order])
    fig.tight_layout()
    fig.savefig(FIG / "benchmark_selected.pdf", bbox_inches="tight")
    fig.savefig(FIG / "benchmark_selected.png", dpi=200, bbox_inches="tight"), plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    check_seeds(runs, ("dataset", "pipeline"), "benchmark ")
    runs.drop(columns="_order").to_csv(OUT / "runs.csv", index=False)
    print("seeds per (dataset, pipeline):\n", runs.groupby(["dataset", "pipeline"]).seed.nunique().unstack(0).to_string())
    sel = select(runs)
    sel.drop(columns="_order").to_csv(OUT / "selected_per_seed.csv", index=False)
    sel_sum = summarise(sel, ["dataset", "pipeline"])
    sel_sum.to_csv(OUT / "summary_selected.csv", index=False)
    per_cfg = summarise(runs, ["dataset", "pipeline", "model", "resampler"])
    per_cfg.to_csv(OUT / "summary_model_resampler.csv", index=False)
    t = tests(sel)
    t.to_csv(OUT / "tests.csv", index=False)
    desc = descriptive(sel, runs)
    desc.to_csv(OUT / "descriptive.csv", index=False)
    b, bd = bootstrap(sel)
    b.to_csv(OUT / "bootstrap_first_seed.csv", index=False)
    bd.to_csv(OUT / "bootstrap_diff_first_seed.csv", index=False)

    figure_benchmark(per_cfg, sel_sum, "30000", "CARDS_30000", ["FC", "CRC", "dethw", "detshot"])
    figure_benchmark(per_cfg, sel_sum, "2500", "CARDS_2500", ["FC", "CRC", "dethw", "detshot", "detaquila"])
    figure_benchmark(per_cfg, sel_sum, "1000", "CARDS_1000",
                     ["FC", "CRC", "dethw", "detshot", "detaquila", "poshw", "posshot", "posaquila"])
    figure_selected(sel)

    pd.set_option("display.width", 220)
    cols = ["f1_mean", "f1_sd", "f1_ci95", "pr_auc_mean", "pr_auc_sd", "roc_auc_mean"]
    for ds, title in DATASETS:
        print(f"\n{title} — selected on validation\n", sel_sum[sel_sum.dataset == ds].set_index("pipeline")[cols].round(3).to_string())
        print(" selected models:", sel[sel.dataset == ds].groupby("pipeline").model.agg(lambda s: s.value_counts().to_dict()).to_dict())
    print("\npre-registered tests (F1)\n", t[["dataset", "comparison", "f1_a", "f1_b", "f1_diff", "f1_diff_ci95", "f1_a_wins",
                                                "p", "p_holm", "p_holm_all", "pr_auc_diff", "pr_auc_p"]].round(4).to_string())
    print("\ndescriptive (uncorrected)\n", desc.round(4).to_string())
    nb = per_cfg[per_cfg.model == "NBT"].pivot(index=["dataset", "pipeline"], columns="resampler", values="f1_mean")
    print("\nNB (train only), test F1 per resampler\n", nb[RESAMPLERS].round(3).to_string())
    print("\nbootstrap, first seed\n", b.round(3).to_string())
    print("\nbootstrap differences, first seed\n", bd.round(3).to_string())


if __name__ == "__main__":
    main()
