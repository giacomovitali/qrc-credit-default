"""Ablation analysis: the ablation ladder of the reservoir's contribution.

Rungs, all read out by the paper's six classifiers x four resamplers with the benchmark protocol (10 seeds, the same
splits; CARDS_30000 on the pre-registered split seeds, with pca / rff / tanh fitted on the
training rows), each selected on validation per (dataset, rung, seed) with the benchmark's select():
  FC24     the 24 variables (baselines)                      pca      the 11 whitened PCs of the 24 variables
  FC       the 12 aggregated features (baselines)            poly2i   12 + 66 products (one readout's structure)
  poly2    full degree 2 (90)                         poly3    full degree 3 (454)
  rff1     390 random Fourier features (cos)          tanh     390 random tanh units
  rff05 / rff2  bandwidth x 0.5 / x 2 (CARDS_2500 and CARDS_1000; descriptive)
  CRC      the mean-field reservoir (benchmark)
  <enc><backend>:<subset>  the quantum reservoir (enc = det / pos; backend = hw noiseless, shot 240, aquila)
           restricted to readouts t1..t5 (0.5 ... 2.5 µs), cumulative t12, t123, t1234, all (t12345, = benchmark),
           <Z> only (z12345) or <Z> at 2.5 µs (z5)
  <enc><backend>:prefix  the cumulative prefix (t1 ... t12345) chosen on validation together with the model
           and resampler (ties: the longer prefix)

Tests (fixed in this file before the sweep's results were read): two-sided paired Wilcoxon signed-rank over
the seeds on the selected test F1, Holm-corrected within each dataset. For every noiseless reservoir H
(detuning on each dataset, position on CARDS_1000):
  H:t12345 - poly2i   does the reservoir beat the quadratic control of the same observable structure?
  H:t12345 - rff1     ... a random nonlinear feature map of the same dimension?
  H:t5 - poly2i       one readout (78) against the quadratic control (78)
  H:t12345 - H:z12345 do the correlators <Z_i Z_j> add to the magnetisations?
and for every noisy reservoir N (240 shots; Aquila):
  N:prefix - N:t12345 does dropping the late readouts help?
Everything else (every rung against QRC-hw and FC, PR-AUC) is descriptive and uncorrected.

Writes results/ablation/{runs,selected_per_seed,summary_selected,tests,descriptive,timesteps}.csv,
ablation_table.md and figures/. Read the CSVs with keep_default_na=False.

    python analysis/ablation/analysis.py
"""
import importlib.util
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("benchmark_analysis", ROOT / "analysis" / "benchmark" / "analysis.py")
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)

OUT = ROOT / "results" / "ablation"
FIG = OUT / "figures"
BENCHMARK_LOGS = ROOT / "results" / "benchmark" / "logs"
ABLATION_NAME = re.compile(r"^c(30000|2500|1000)_([a-z0-9]+)(?:-([tz]\d+))?(?:_[A-Z]+)?_(\d+)$")
DATASETS = bm.DATASETS
PREFIXES = ["t1", "t12", "t123", "t1234", "t12345"]
SINGLES = ["t1", "t2", "t3", "t4", "t5"]
RESERVOIRS = {"30000": ["dethw", "detshot"], "2500": ["dethw", "detshot", "detaquila"],
              "1000": ["dethw", "detshot", "detaquila", "poshw", "posshot", "posaquila"]}
BACKEND = {"hw": "statevector", "shot": "240 shots", "aquila": "Aquila"}
ENCODING = {"det": "detuning", "pos": "position"}
# ladder order (table rows and figure) and labels
CLASSICAL = [("FC24", "24 variables"), ("pca", "11 PCs (24 variables)"), ("FC", "12 features (FC)"),
             ("poly2i", "12 + pairwise products (78)"), ("poly2", "degree 2 (90)"), ("poly3", "degree 3 (454)"),
             ("rff1", "random Fourier features (390)"), ("rff05", "random Fourier, bandwidth x 0.5"),
             ("rff2", "random Fourier, bandwidth x 2"), ("tanh", "random tanh features (390)"), ("CRC", "CRC (390)")]
SUBSET_LABEL = {"t12345": "all 5 readouts", "t1": "0.5 µs only", "t2": "1.0 µs only", "t3": "1.5 µs only",
                "t4": "2.0 µs only", "t5": "2.5 µs only", "t12": "0.5-1.0 µs", "t123": "0.5-1.5 µs",
                "t1234": "0.5-2.0 µs", "z12345": "⟨Z⟩ only, 5 readouts", "z5": "⟨Z⟩ only, 2.5 µs",
                "prefix": "readout prefix chosen on validation"}
COLOURS = {"hw": "#1baf7a", "shot": "#eda100", "aquila": "#e87ba4"}      # = the benchmark's detuning series
MARKERS = {"hw": "o", "shot": "s", "aquila": "^"}
plt.rcParams.update(bm.plt.rcParams)


def label(p):
    if ":" in p:
        base, sub = p.split(":")
        return f"QRC {ENCODING[base[:3]]}, {BACKEND[base[3:]]}: {SUBSET_LABEL[sub]}"
    return dict(CLASSICAL)[p]


def tests_spec():
    out = {}
    for ds, res in RESERVOIRS.items():
        t = []
        for h in [r for r in res if r.endswith("hw")]:
            t += [(f"{h}:t12345", "poly2i"), (f"{h}:t12345", "rff1"), (f"{h}:t5", "poly2i"),
                  (f"{h}:t12345", f"{h}:z12345")]
        t += [(f"{n}:prefix", f"{n}:t12345") for n in res if not n.endswith("hw")]
        out[ds] = t
    return out


TESTS = tests_spec()


def load_runs():
    rows = []
    for log in sorted((OUT / "logs").glob("*.log")):
        m = ABLATION_NAME.match(log.stem)
        if m and log.with_suffix(".predictions.csv.gz").exists():
            ds, base, sub, seed = m.groups()
            rows += bm.read_log(log, dataset=ds, pipeline=f"{base}:{sub}" if sub else base, seed=int(seed))
    bm_runs = bm.load_runs()                      # benchmark reservoirs (full vectors) and the baselines' FC / FC24
    bm_runs = bm_runs[bm_runs.pipeline.isin(["FC", "FC24", "CRC"] + sum(RESERVOIRS.values(), []))].copy()
    full = ~bm_runs.pipeline.isin(["FC", "FC24", "CRC"])
    bm_runs.loc[full, "pipeline"] = bm_runs.loc[full, "pipeline"] + ":t12345"
    df = pd.concat([pd.DataFrame(rows), bm_runs.drop(columns="_order")], ignore_index=True)
    df = df[df.model.isin(bm.PAPER)]
    df["_order"] = df.resampler.map(bm.RESAMPLERS.index) * 100 + df.model.map(bm.PAPER.index)
    # the prefix pipelines: the cumulative readout prefix is one more hyperparameter chosen on validation
    pre = df[df.pipeline.str.contains(":") & df.pipeline.str.split(":").str[1].isin(PREFIXES)].copy()
    k = pre.pipeline.str.split(":").str[1].map(PREFIXES.index)
    pre["_order"] += (len(PREFIXES) - 1 - k) * 1000                  # ties go to the longer prefix
    pre["prefix"] = pre.pipeline.str.split(":").str[1]
    pre["pipeline"] = pre.pipeline.str.split(":").str[0] + ":prefix"
    complete = pre.groupby(["dataset", "pipeline", "seed"]).prefix.transform("nunique") == len(PREFIXES)
    pre = pre[complete]
    return pd.concat([df, pre], ignore_index=True)


def paired(sel, ds, a, b, metric="f1"):
    s = sel[sel.dataset == ds]
    x, y = s[s.pipeline == a].set_index("seed")[metric], s[s.pipeline == b].set_index("seed")[metric]
    seeds = x.index.intersection(y.index)
    d = x[seeds] - y[seeds]
    p = wilcoxon(x[seeds], y[seeds]).pvalue if len(d) > 1 and (d != 0).any() else 1.0
    ci = student_t.ppf(0.975, len(d) - 1) * d.std() / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return dict(n=len(d), a_mean=x[seeds].mean(), b_mean=y[seeds].mean(), diff=d.mean(), diff_ci95=ci,
                a_wins=int((d > 0).sum()), p=p)


def tests(sel):
    rows = []
    for ds, pairs in TESTS.items():
        for a, b in pairs:
            r = paired(sel, ds, a, b)
            if r["n"] < 2:
                continue
            pr = paired(sel, ds, a, b, "pr_auc")
            rows.append(dict(dataset=ds, a=a, b=b, comparison=f"{label(a)} - {label(b)}", **r,
                             pr_auc_diff=pr["diff"], pr_auc_p=pr["p"]))
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["p_holm"] = t.groupby("dataset").p.transform(lambda p: pd.Series(bm.holm(p), index=p.index))
    return t


def descriptive(sel):
    """Every rung against the noiseless reservoir of the dataset and against FC (uncorrected)."""
    rows = []
    for ds, _ in DATASETS:
        pipes = sel[sel.dataset == ds].pipeline.unique()
        for ref in [f"{h}:t12345" for h in RESERVOIRS[ds] if h.endswith("hw")] + ["FC"]:
            for p in pipes:
                if p == ref:
                    continue
                for m in ["f1", "pr_auc"]:
                    r = paired(sel, ds, p, ref, m)
                    rows.append(dict(dataset=ds, a=p, b=ref, metric=m, comparison=f"{label(p)} - {label(ref)}", **r))
    return pd.DataFrame(rows)


def ablation_table(sel_sum):
    """Markdown: one row per rung, F1 and PR-AUC mean ± SD per dataset."""
    order = [p for p, _ in CLASSICAL]
    for enc in ["det", "pos"]:
        for be in ["hw", "shot", "aquila"]:
            order += [f"{enc}{be}:{s}" for s in ["t12345"] + SINGLES + ["t12", "t123", "t1234", "z12345", "z5", "prefix"]]
    s = sel_sum.set_index(["pipeline", "dataset"])
    lines = ["| Rung | " + " | ".join(f"{t} F1 | {t} PR-AUC" for _, t in DATASETS) + " |",
             "| --- |" + " --- | --- |" * len(DATASETS)]
    for p in order:
        cells = []
        for ds, _ in DATASETS:
            if (p, ds) in s.index:
                r = s.loc[(p, ds)]
                cells += [f"{r.f1_mean:.3f} ± {r.f1_sd:.3f}", f"{r.pr_auc_mean:.3f} ± {r.pr_auc_sd:.3f}"]
            else:
                cells += ["—", "—"]
        if any(c != "—" for c in cells):
            lines.append(f"| {label(p)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def figure_timesteps(sel_sum, metric="f1"):
    """F1 vs readouts included (cumulative prefix), one panel per dataset and encoding; noiseless / 240 shots /
    Aquila. Open markers: the single readout alone. Dotted line: the quadratic control (poly2i); dashed: FC."""
    panels = [("30000", "det"), ("2500", "det"), ("1000", "det"), ("1000", "pos")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=True)
    s = sel_sum.set_index(["dataset", "pipeline"])
    x = np.arange(1, 6)
    for ax, (ds, enc) in zip(axes, panels):
        ends = []
        for be in ["hw", "shot", "aquila"]:
            p = f"{enc}{be}"
            if p not in RESERVOIRS[ds]:
                continue
            cum = [s.loc[(ds, f"{p}:{k}")] if (ds, f"{p}:{k}") in s.index else None for k in PREFIXES]
            one = [s.loc[(ds, f"{p}:{k}")] if (ds, f"{p}:{k}") in s.index else None for k in SINGLES]
            m = np.array([np.nan if r is None else r[f"{metric}_mean"] for r in cum])
            sd = np.array([np.nan if r is None else r[f"{metric}_sd"] for r in cum])
            c = COLOURS[be]
            ax.fill_between(x, m - sd, m + sd, color=c, alpha=0.12, lw=0)
            ax.plot(x, m, marker=MARKERS[be], color=c, ms=5, mec="#fcfcfb", mew=1, label=f"{BACKEND[be]}, readouts 1..k")
            ax.plot(x, [np.nan if r is None else r[f"{metric}_mean"] for r in one], ls="none", marker=MARKERS[be],
                    mfc="none", mec=c, ms=5, mew=1, label=f"{BACKEND[be]}, readout k alone")
            if not np.isnan(m[-1]):
                ends.append([m[-1], BACKEND[be]])
        ends.sort(reverse=True)                     # direct labels, pushed apart to >= 0.028
        for i in range(1, len(ends)):
            ends[i][0] = min(ends[i][0], ends[i - 1][0] - 0.028)
        for yv, text in ends:
            ax.annotate(text, (5.15, yv), fontsize=6.5, color=bm.INK2, va="center")
        for ref, ls in [("FC", "--"), ("poly2i", ":")]:
            if (ds, ref) in s.index:
                ax.axhline(s.loc[(ds, ref)][f"{metric}_mean"], color=bm.INK2, ls=ls, lw=1,
                           label="FC (12 features)" if ref == "FC" else "12 + pairwise products (78)")
        ax.set_xticks(x, ["0.5", "1.0", "1.5", "2.0", "2.5"])
        ax.set_xlim(0.7, 5.9)
        ax.set_xlabel("readout k [µs]")
        ax.set_title(f"CARDS_{ds}, {ENCODING[enc]}", fontsize=8, color=bm.INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(f"test {'F1' if metric == 'f1' else 'PR-AUC'}, selected (mean ± SD)")
    h, l = axes[2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    stem = "ablation_timesteps" if metric == "f1" else "ablation_timesteps_pr_auc"
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=200, bbox_inches="tight"), plt.close(fig)


def figure_ladder(sel):
    """Validation-selected test F1 per rung (every seed + mean ± SD), one panel per dataset; the noiseless
    detuning reservoir's mean as a vertical reference."""
    rungs = ["FC24", "pca", "FC", "poly2i", "poly2", "poly3", "rff1", "tanh", "CRC",
             "dethw:z5", "dethw:z12345", "dethw:t5", "dethw:t12345", "detshot:t12345", "detaquila:t12345"]
    colour = {"dethw": COLOURS["hw"], "detshot": COLOURS["shot"], "detaquila": COLOURS["aquila"]}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), sharey=True)
    y = np.arange(len(rungs))[::-1]
    for ax, (ds, title) in zip(axes, DATASETS):
        s = sel[sel.dataset == ds]
        ref = s[s.pipeline == "dethw:t12345"].f1.mean()
        ax.axvline(ref, color=COLOURS["hw"], lw=1, ls="--", alpha=0.8)
        for yi, p in zip(y, rungs):
            v = s[s.pipeline == p].f1
            if not len(v):
                continue
            c = colour.get(p.split(":")[0], "#2a78d6" if p in ("FC", "FC24", "pca") else
                           "#eb6834" if p == "CRC" else "#6250d6")
            ax.plot(v, np.full(len(v), yi) + np.random.default_rng(0).uniform(-0.18, 0.18, len(v)), "o", ms=2.5,
                    color=c, alpha=0.35, mec="none")
            ax.errorbar(v.mean(), yi, xerr=v.std(), fmt="o", color=c, ms=5, elinewidth=1.2, capsize=0,
                        mec="#fcfcfb", mew=1)
        ax.set_title(title, fontsize=8.5, color=bm.INK)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [label(p).replace("QRC detuning, ", "QRC det., ") for p in rungs], fontsize=6.5)
    fig.supxlabel("test F1, selected on validation (every seed, mean ± SD; dashed: QRC detuning, statevector)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "ablation_ladder.pdf", bbox_inches="tight")
    fig.savefig(FIG / "ablation_ladder.png", dpi=200, bbox_inches="tight"), plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    bm.check_seeds(runs, ("dataset", "pipeline"), "ablation ")
    runs.drop(columns="_order").to_csv(OUT / "runs.csv", index=False)
    n = runs.groupby(["dataset", "pipeline"]).seed.nunique().unstack(0)
    print("seeds per (pipeline, dataset); incomplete rows:\n", n[(n < 10).any(axis=1)].to_string())
    sel = bm.select(runs)
    sel.drop(columns="_order").to_csv(OUT / "selected_per_seed.csv", index=False)
    sel_sum = bm.summarise(sel, ["dataset", "pipeline"])
    sel_sum["label"] = sel_sum.pipeline.map(label)
    sel_sum.to_csv(OUT / "summary_selected.csv", index=False)
    t = tests(sel)
    t.to_csv(OUT / "tests.csv", index=False)
    desc = descriptive(sel)
    desc.to_csv(OUT / "descriptive.csv", index=False)
    ts = sel_sum[sel_sum.pipeline.str.contains(":")].copy()
    ts[["reservoir", "subset"]] = ts.pipeline.str.split(":", expand=True)
    ts.to_csv(OUT / "timesteps.csv", index=False)
    (OUT / "ablation_table.md").write_text(ablation_table(sel_sum))
    figure_timesteps(sel_sum)
    figure_timesteps(sel_sum, "pr_auc")
    figure_ladder(sel)

    pd.set_option("display.width", 220)
    for ds, title in DATASETS:
        print(f"\n{title} — selected on validation\n",
              sel_sum[sel_sum.dataset == ds].set_index("pipeline")[["f1_mean", "f1_sd", "pr_auc_mean", "pr_auc_sd",
                                                                     "roc_auc_mean"]].round(3).to_string())
        pre = sel[(sel.dataset == ds) & sel.pipeline.str.endswith(":prefix")]
        print(" prefixes chosen on validation:", pre.groupby("pipeline").prefix.agg(lambda s: s.value_counts().to_dict()).to_dict())
    if len(t):
        print("\npre-registered tests (F1)\n", t[["dataset", "comparison", "a_mean", "b_mean", "diff", "diff_ci95",
                                                  "a_wins", "p", "p_holm", "pr_auc_diff", "pr_auc_p"]].round(4).to_string())


if __name__ == "__main__":
    main()
