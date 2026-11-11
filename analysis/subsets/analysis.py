"""Robustness of the CARDS_2500 / CARDS_1000 results to the choice of subset.

Subsamples (make_subsets.py): 10 independent stratified subsamples of the cleaned 29,601 at each size,
2,571 and 1,000, each run with the benchmark protocol (run_classification.sh): seeds 1-10, four resamplers,
validation selection with the benchmark's select(). Pipelines: FC, DNN (12 features), trees24 (best tree model,
24 variables), CRC, QRC-hw detuning and (size 1,000) QRC-hw position, i.e. every benchmark pipeline that does not
need hardware. The actual subsets come from benchmark/baseline (the same runs as in the benchmark's tables).

Fixed before the sweep's results (pre-registered):
1. Placement (descriptive). Per size and pipeline, the subsample's score is its mean selected test F1 (and
   PR-AUC) over the 10 seeds. Report the 10 scores (mean, SD, range) and where the actual subset's score
   falls: the fraction of subsamples below it, its rank among the 11, and z = (actual - mean) / SD.
2. Variance decomposition (descriptive). One-way ANOVA over subsamples (10 x 10 seeds): the F-test, and
   the between-subsample SD component sqrt(max(0, (MS_between - MS_within) / 10)) against the seed SD
   sqrt(MS_within).
3. Robustness of the benchmark's comparisons. For QRC-hw detuning - {FC, CRC, trees24, DNN} at both sizes, and QRC-hw
   position - {FC, CRC, trees24, DNN} at size 1,000: the per-subsample paired difference (mean over seeds);
   across subsamples its mean, SD, the number of positive subsamples, and a Wilcoxon signed-rank test over
   the 10 subsample differences (the subsample is the unit), Holm-corrected within each size (4 and 8
   tests). Per subsample also the benchmark's test over seeds (paired Wilcoxon), counted as significant at 0.05 in each
   direction. The actual subset's difference (benchmark) is placed among the subsample differences as in (1).
   Descriptive: QRC-hw detuning - position at size 1,000.
Regression: the actual subsets written through make_subsets.py (tags cards2500 / cards1000, seed 1) must
reproduce the baseline/benchmark rows of seed 1.

Writes results/subsets/{runs,selected_per_seed,subsample_scores,placement,variance,comparisons,
comparisons_per_subsample,regression}.csv and figures/. Read the CSVs with keep_default_na=False.

    python analysis/subsets/analysis.py
"""
import importlib.util
import re
from pathlib import Path

import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
# the benchmark's analysis (select, holm, log reader, palette; importing it also sets the shared figure style)
_spec = importlib.util.spec_from_file_location("benchmark_analysis", ROOT / "analysis" / "benchmark" / "analysis.py")
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)

plt = bm.plt
OUT = ROOT / "results" / "subsets"
FIG = OUT / "figures"
NAME = re.compile(r"^(n2571s\d\d|n1000s\d\d|cards2500|cards1000)_(fc12|fc24|crc|dethw|poshw)_(\d+)$")
SIZES = [("2500", 2571, "CARDS_2500 size (2,571)"), ("1000", 1000, "CARDS_1000 size (1,000)")]
PIPES = {"2500": ["FC", "DNN", "trees24", "CRC", "dethw"], "1000": ["FC", "DNN", "trees24", "CRC", "dethw", "poshw"]}
BASELINES = ["FC", "CRC", "trees24", "DNN"]
COMPARISONS = {"2500": [("dethw", b) for b in BASELINES],
               "1000": [(q, b) for q in ["dethw", "poshw"] for b in BASELINES]}
DESCRIPTIVE = {"1000": [("dethw", "poshw")]}
METRICS = ["f1", "pr_auc"]


def pipeline_of(inp, model):
    if inp == "fc12":
        return "DNN" if model == "DNN" else "FC"
    return {"fc24": "trees24", "crc": "CRC"}.get(inp, inp)


def load_runs():
    rows = []
    for log in sorted((OUT / "logs").glob("*.log")):
        m = NAME.match(log.stem)
        if m and log.with_suffix(".predictions.csv.gz").exists():
            tag, inp, seed = m.groups()
            for r in bm.read_log(log, dataset=tag, seed=int(seed)):
                r["pipeline"] = pipeline_of(inp, r["model"])
                rows.append(r)
    df = pd.DataFrame(rows)
    df["_order"] = df.resampler.map(bm.RESAMPLERS.index) * 100 + df.model.map((bm.PAPER + ["DNN"] + bm.TREES).index)
    return df


def regression(runs, ref):
    """Subsample code path on the actual subsets vs the baseline/benchmark rows, seed 1, every (pipeline, model, resampler)."""
    rows = []
    for tag, ds in [("cards2500", "2500"), ("cards1000", "1000")]:
        a = runs[(runs.dataset == tag) & (runs.seed == 1)].set_index(["pipeline", "model", "resampler"])
        b = ref[(ref.dataset == ds) & (ref.seed == 1)].set_index(["pipeline", "model", "resampler"])
        for key in a.index.intersection(b.index):
            d = {m: abs(a.loc[key, m] - b.loc[key, m]) for m in ["f1", "val_f1", "roc_auc", "pr_auc"]}
            rows.append(dict(dataset=ds, pipeline=key[0], model=key[1], resampler=key[2],
                             **{f"absdiff_{m}": v for m, v in d.items()},
                             identical=all(v == 0 for v in d.values()), within_1e12=all(v < 1e-12 for v in d.values())))
    return pd.DataFrame(rows)


def place(value, ref):
    ref = np.asarray(ref, float)
    return dict(frac_below=(ref < value).mean(), rank_of_11=int((ref < value).sum()) + 1,
                z=(value - ref.mean()) / ref.std(ddof=1))


def anova(g):
    """One-way ANOVA of the selected F1 over subsamples (rows: subsample, seed)."""
    groups = [v.to_numpy() for _, v in g.groupby("dataset").f1]
    k, n = len(groups), len(groups[0])
    grand = np.concatenate(groups).mean()
    msb = n * sum((x.mean() - grand) ** 2 for x in groups) / (k - 1)
    msw = sum(((x - x.mean()) ** 2).sum() for x in groups) / (k * (n - 1))
    return dict(k=k, n_per=n, F=msb / msw, p=f_dist.sf(msb / msw, k - 1, k * (n - 1)),
                sd_between=np.sqrt(max(0.0, (msb - msw) / n)), sd_seed=np.sqrt(msw))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    runs.drop(columns="_order").to_csv(OUT / "runs.csv", index=False)
    ref_runs = bm.load_runs()
    ref_runs = ref_runs[ref_runs.dataset.isin(["2500", "1000"])]
    reg = regression(runs, ref_runs)
    reg.to_csv(OUT / "regression.csv", index=False)

    sub = runs[runs.dataset.str.match(r"n\d+s\d\d")]
    sel = bm.select(sub)
    sel.drop(columns="_order").to_csv(OUT / "selected_per_seed.csv", index=False)
    sel["size"] = sel.dataset.str.slice(1, 5).map({"2571": "2500", "1000": "1000"})
    ref = bm.select(ref_runs)
    scores = sel.groupby(["size", "dataset", "pipeline"])[METRICS].mean().reset_index()
    scores.to_csv(OUT / "subsample_scores.csv", index=False)
    print("seeds per (subsample, pipeline):", sel.groupby(["dataset", "pipeline"]).seed.nunique().value_counts().to_dict())

    placement, variance = [], []
    for size, _, _ in SIZES:
        for p in PIPES[size]:
            sc = scores[(scores["size"] == size) & (scores.pipeline == p)]
            actual = ref[(ref.dataset == size) & (ref.pipeline == p)]
            if not len(sc):
                continue
            row = dict(size=size, pipeline=p, n_subsamples=len(sc))
            for m in METRICS:
                row |= {f"{m}_sub_mean": sc[m].mean(), f"{m}_sub_sd": sc[m].std(), f"{m}_sub_min": sc[m].min(),
                        f"{m}_sub_max": sc[m].max(), f"{m}_actual": actual[m].mean()}
                row |= {f"{m}_actual_{k}": v for k, v in place(actual[m].mean(), sc[m]).items()}
            placement.append(row)
            variance.append(dict(size=size, pipeline=p, **anova(sel[(sel["size"] == size) & (sel.pipeline == p)]),
                                 sd_seed_actual=actual.f1.std()))
    placement, variance = pd.DataFrame(placement), pd.DataFrame(variance)
    placement.to_csv(OUT / "placement.csv", index=False)
    variance.to_csv(OUT / "variance.csv", index=False)

    comp, per_sub = [], []
    for size, _, _ in SIZES:
        for kind, pairs in [("pre-registered", COMPARISONS[size]), ("descriptive", DESCRIPTIVE.get(size, []))]:
            for a, b in pairs:
                s = sel[sel["size"] == size].set_index(["dataset", "seed", "pipeline"]).f1.unstack("pipeline")
                if a not in s or b not in s:
                    continue
                d = (s[a] - s[b]).dropna()
                diffs = []
                for tag, dd in d.groupby(level="dataset"):
                    p = wilcoxon(dd).pvalue if (dd != 0).any() else 1.0
                    per_sub.append(dict(size=size, a=a, b=b, subsample=tag, n_seeds=len(dd), f1_diff=dd.mean(),
                                        a_wins_seeds=int((dd > 0).sum()), p_seeds=p))
                    diffs.append(dd.mean())
                diffs = np.array(diffs)
                r = ref[ref.dataset == size].set_index(["seed", "pipeline"]).f1.unstack("pipeline")
                actual = (r[a] - r[b]).mean()
                ps = [x for x in per_sub if x["size"] == size and x["a"] == a and x["b"] == b]
                comp.append(dict(size=size, kind=kind, a=a, b=b,
                                 comparison=f"{bm.PIPELINES[a][0]} - {bm.PIPELINES[b][0]}",
                                 n_subsamples=len(diffs), f1_diff_mean=diffs.mean(), f1_diff_sd=diffs.std(ddof=1),
                                 f1_diff_min=diffs.min(), f1_diff_max=diffs.max(), n_positive=int((diffs > 0).sum()),
                                 p=wilcoxon(diffs).pvalue,
                                 n_sig_pos_seeds=sum(x["p_seeds"] < 0.05 and x["f1_diff"] > 0 for x in ps),
                                 n_sig_neg_seeds=sum(x["p_seeds"] < 0.05 and x["f1_diff"] < 0 for x in ps),
                                 actual_diff=actual, **{f"actual_{k}": v for k, v in place(actual, diffs).items()}))
    comp, per_sub = pd.DataFrame(comp), pd.DataFrame(per_sub)
    pre = comp.kind == "pre-registered"
    comp.loc[pre, "p_holm"] = comp[pre].groupby("size").p.transform(lambda p: pd.Series(bm.holm(p), index=p.index))
    comp.to_csv(OUT / "comparisons.csv", index=False)
    per_sub.to_csv(OUT / "comparisons_per_subsample.csv", index=False)

    figure_scores(scores, ref)
    figure_differences(per_sub, ref)

    pd.set_option("display.width", 220)
    print(f"\nregression (subsample path vs baseline/benchmark, seed 1): {int(reg.identical.sum())}/{len(reg)} rows identical, "
          f"{int(reg.within_1e12.sum())} within 1e-12; max |dF1| {reg.absdiff_f1.max():.3g}")
    if (~reg.identical).any():
        print(reg[~reg.identical].round(6).to_string())
    print("\nplacement of the actual subsets (F1 and PR-AUC; subsample = mean over 10 seeds)\n",
          placement[["size", "pipeline", "f1_sub_mean", "f1_sub_sd", "f1_sub_min", "f1_sub_max", "f1_actual",
                     "f1_actual_frac_below", "f1_actual_rank_of_11", "f1_actual_z", "pr_auc_sub_mean", "pr_auc_actual",
                     "pr_auc_actual_frac_below"]].round(3).to_string())
    print("\nvariance over subsamples vs seeds\n", variance.round(4).to_string())
    print("\ncomparisons across subsamples\n",
          comp[["size", "kind", "comparison", "f1_diff_mean", "f1_diff_sd", "f1_diff_min", "f1_diff_max", "n_positive",
                "p", "p_holm", "n_sig_pos_seeds", "n_sig_neg_seeds", "actual_diff", "actual_frac_below"]].round(4).to_string())
    print("\nselected models (subsamples):", sel.groupby(["size", "pipeline"]).model.agg(lambda s: s.value_counts().to_dict()).to_dict())


def figure_scores(scores, ref):
    """Per size: each subsample's mean selected test F1 (dots), their mean ± SD, and the actual subset (benchmark)."""
    order = ["trees24", "DNN", "FC", "CRC", "dethw", "poshw"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, (size, _, title) in zip(axes, SIZES):
        for yi, p in zip(y, order):
            v = scores[(scores["size"] == size) & (scores.pipeline == p)].f1
            if not len(v):
                continue
            c = bm.PIPELINES[p][1]
            ax.plot(v, np.full(len(v), yi) + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)), "o", ms=3,
                    color=c, alpha=0.45, mec="none", label="independent subsample (10 seeds)" if yi == y[0] else None)
            ax.errorbar(v.mean(), yi, xerr=v.std(), fmt="o", color=c, ms=5, elinewidth=1.2, capsize=0,
                        mec="#fcfcfb", mew=1, label="mean ± SD over subsamples" if yi == y[0] else None)
            a = ref[(ref.dataset == size) & (ref.pipeline == p)].f1.mean()
            ax.plot(a, yi + 0.3, "v", ms=6, color=bm.INK, mec="#fcfcfb", mew=0.8,
                    label="QPU subset" if yi == y[0] else None)
        ax.set_title(title, fontsize=8.5, color=bm.INK)
        ax.set_xlabel("test F1 (validation-selected, mean over 10 seeds)", fontsize=7.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [bm.PIPELINES[p][0].replace("QRC ", "QRC-").replace(", statevector", " (statevector)") for p in order])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "subsets_f1.pdf", bbox_inches="tight")
    fig.savefig(FIG / "subsets_f1.png", dpi=200, bbox_inches="tight"), plt.close(fig)


def figure_differences(per_sub, ref):
    """Per size: QRC-hw minus each baseline, one dot per subsample, and the actual subset's difference."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
    for ax, (size, _, title) in zip(axes, SIZES):
        pairs = COMPARISONS[size] + DESCRIPTIVE.get(size, [])
        y = np.arange(len(pairs))[::-1]
        r = ref[ref.dataset == size].set_index(["seed", "pipeline"]).f1.unstack("pipeline")
        for yi, (a, b) in zip(y, pairs):
            d = per_sub[(per_sub["size"] == size) & (per_sub.a == a) & (per_sub.b == b)].f1_diff
            c = bm.PIPELINES[a][1]
            first = yi == y[0] and size == "2500"
            ax.plot(d, np.full(len(d), yi) + np.random.default_rng(0).uniform(-0.15, 0.15, len(d)), "o", ms=3,
                    color=c, alpha=0.45, mec="none", label="independent subsample (10 seeds)" if first else None)
            ax.errorbar(d.mean(), yi, xerr=d.std(), fmt="o", color=c, ms=5, elinewidth=1.2, capsize=0,
                        mec="#fcfcfb", mew=1, label="mean ± SD over subsamples" if first else None)
            ax.plot((r[a] - r[b]).mean(), yi + 0.3, "v", ms=6, color=bm.INK, mec="#fcfcfb", mew=0.8,
                    label="QPU subset" if first else None)
        ax.axvline(0, color=bm.INK2, lw=0.8)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.025))
        short = {"dethw": "QRC-det", "poshw": "QRC-pos", "trees24": "trees-24"}
        ax.set_yticks(y, [f"{short.get(a, a)} − {short.get(b, b)}" for a, b in pairs], fontsize=7)
        ax.set_title(title, fontsize=8.5, color=bm.INK)
        ax.set_xlabel("difference in test F1 (mean over 10 seeds)", fontsize=7.5)
        ax.grid(axis="y", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "subsets_differences.pdf", bbox_inches="tight")
    fig.savefig(FIG / "subsets_differences.png", dpi=200, bbox_inches="tight"), plt.close(fig)


if __name__ == "__main__":
    main()
