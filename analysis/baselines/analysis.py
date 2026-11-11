"""Baseline analysis: classical baselines on the 12 aggregated features (fc12) and the 24 preprocessed
variables (fc24), CARDS_30000 / 2500 / 1000, 10 seeds each (CARDS_2500/1000: 1-10; CARDS_30000: the
pre-registered split seeds, preprocessing/split_seeds.py).

Selection is on validation only: for each (dataset, input, model, seed) the resampler with the best
validation F1; for a model group (paper's classifiers, trees, DNN) the (model, resampler) with the best
validation F1 among the group's members. Reads results/baselines/logs/ (run_classification.sh).
Writes results/baselines/{runs,summary_model_resampler,selected_model,selected_group,paired,
n_params}.csv and figures/baselines.pdf.

    python analysis/baselines/analysis.py
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "preprocessing"))
from split_seeds import SEEDS, check_seeds  # noqa: E402  per-dataset split seeds

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "baselines"
FIG = OUT / "figures"
NAME = re.compile(r"^c(30000|2500|1000)_(fc12|fc24)_(trees|rest|all)_(\d+)$")
GROUPS = {"paper": ["KNN", "SGD", "NBT", "LM", "P", "LSVM"], "trees": ["DTR", "RFR", "ET", "HGB"], "DNN": ["DNN"]}
MODELS = GROUPS["paper"] + GROUPS["DNN"] + GROUPS["trees"]
DATASETS = [("30000", "CARDS_30000"), ("2500", "CARDS_2500"), ("1000", "CARDS_1000")]
C = {"fc12": "#2a78d6", "fc24": "#eb6834"}
MARK = {"fc12": "o", "fc24": "s"}
LABEL = {"fc12": "12 aggregated features", "fc24": "24 preprocessed variables"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 4,
    "pdf.fonttype": 42,
})
KEYS = ["dataset", "input", "seed"]


def load_runs():
    rows = []
    for log in sorted((OUT / "logs").glob("*.log")):
        m = NAME.match(log.stem)
        if not m or not log.with_suffix(".predictions.csv.gz").exists():
            continue
        ds, inp, _, seed = m.groups()
        for line in log.read_text().splitlines():
            if "Extended metrics: " in line:
                d = json.loads(line.split("Extended metrics: ", 1)[1])
                rows.append(dict(dataset=ds, input=inp, seed=int(seed), model=d["model"], resampler=d["resampler"],
                                 f1=d["f1"], val_f1=d["val_f1"], precision=d["precision"], recall=d["recall"],
                                 roc_auc=d["roc_auc"], pr_auc=d["pr_auc"], balanced_acc=d["balanced_acc"],
                                 n_params=d["n_params"], grid_time_s=d["grid_time_s"],
                                 best_params=json.dumps(d["best_params"])))
    return pd.DataFrame(rows)


def best_on_val(df, by):
    """Row with the highest validation F1 per group (ties: first in log order)."""
    return df.loc[df.groupby(by, sort=False).val_f1.idxmax()].reset_index(drop=True)


def summarise(df, by):
    return df.groupby(by).agg(
        n=("f1", "size"), f1_mean=("f1", "mean"), f1_sd=("f1", "std"), pr_auc_mean=("pr_auc", "mean"),
        pr_auc_sd=("pr_auc", "std"), roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
        val_f1_mean=("val_f1", "mean")).reset_index()


def paired(sel_model, sel_group):
    """Paired over seeds (Wilcoxon signed-rank): fc24 - fc12 per model and group; trees - paper and
    trees - DNN per input. Descriptive, not part of the benchmark's pre-registered family."""
    rows = []

    def add(a, b, **kw):
        a, b = a.sort_index(), b.reindex(a.sort_index().index)
        d = a - b
        if d.notna().sum() < 2:
            return
        rows.append(dict(**kw, n=len(d), mean_diff=d.mean(), sd_diff=d.std(),
                         p_wilcoxon=wilcoxon(a, b).pvalue if (d != 0).any() else 1.0))

    for ds, _ in DATASETS:
        for name, sel, col in [(m, sel_model, "model") for m in MODELS] + [(g, sel_group, "group") for g in ["paper", "trees"]]:
            s = sel[(sel.dataset == ds) & (sel[col] == name)]
            add(s[s.input == "fc24"].set_index("seed").f1, s[s.input == "fc12"].set_index("seed").f1,
                dataset=ds, comparison=f"{name}: fc24 - fc12")
        for inp in ["fc12", "fc24"]:
            s = sel_group[(sel_group.dataset == ds) & (sel_group.input == inp)]
            for other in ["paper", "DNN"]:
                add(s[s.group == "trees"].set_index("seed").f1, s[s.group == other].set_index("seed").f1,
                    dataset=ds, comparison=f"{inp}: trees - {other}")
    return pd.DataFrame(rows)


def figure(sm, sg):
    rows = MODELS + ["best paper model", "best tree model"]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 3.4), sharey=True)
    y = np.arange(len(rows))[::-1]
    for j, (ds, title) in enumerate(DATASETS):
        for k, inp in enumerate(["fc12", "fc24"]):
            a = sm[(sm.dataset == ds) & (sm.input == inp)].set_index("model")
            g = sg[(sg.dataset == ds) & (sg.input == inp)].set_index("group")
            mean = [a.f1_mean.get(m, np.nan) for m in MODELS] + [g.f1_mean.get("paper", np.nan), g.f1_mean.get("trees", np.nan)]
            sd = [a.f1_sd.get(m, np.nan) for m in MODELS] + [g.f1_sd.get("paper", np.nan), g.f1_sd.get("trees", np.nan)]
            ax[j].errorbar(mean, y + (0.15 if k == 0 else -0.15), xerr=sd, fmt=MARK[inp], color=C[inp], ms=4.5,
                           elinewidth=1, capsize=0, label=LABEL[inp])
        for b in [len(rows) - len(GROUPS["paper"]) - 0.5, len(rows) - len(GROUPS["paper"]) - 1.5, 1.5]:
            ax[j].axhline(b, color=INK2, lw=0.5)
        ax[j].set_title(title, fontsize=8.5, color=INK)
        ax[j].set_xlabel("test F1 (mean ± SD, 10 seeds)")
        ax[j].grid(axis="y", visible=False)
    ax[0].set_yticks(y, rows)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "baselines.pdf", bbox_inches="tight")
    fig.savefig(FIG / "baselines.png", dpi=200, bbox_inches="tight"), plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    check_seeds(runs, ("dataset", "input", "model"), "baselines ")
    runs["group"] = runs.model.map({m: g for g, ms in GROUPS.items() for m in ms})
    runs.to_csv(OUT / "runs.csv", index=False)
    n_runs = runs.groupby(["dataset", "input"]).seed.nunique()
    print("seeds per (dataset, input):", n_runs.to_dict())

    summarise(runs, ["dataset", "input", "model", "resampler"]).to_csv(OUT / "summary_model_resampler.csv", index=False)
    sel_model = best_on_val(runs, KEYS + ["model"])
    sel_group = best_on_val(runs, KEYS + ["group"])
    sm = summarise(sel_model, ["dataset", "input", "model"])
    sg = summarise(sel_group, ["dataset", "input", "group"])
    sm.to_csv(OUT / "selected_model.csv", index=False)
    sg.to_csv(OUT / "selected_group.csv", index=False)
    sel_group.to_csv(OUT / "selected_group_per_seed.csv", index=False)
    pr = paired(sel_model, sel_group)
    pr.to_csv(OUT / "paired.csv", index=False)
    npar = sel_model.groupby(["dataset", "input", "model"]).n_params.agg(["median", "min", "max"]).reset_index()
    npar.to_csv(OUT / "n_params.csv", index=False)
    figure(sm, sg)

    pd.set_option("display.width", 200)
    for ds, title in DATASETS:
        t = sm[sm.dataset == ds].pivot(index="model", columns="input", values=["f1_mean", "f1_sd", "pr_auc_mean"])
        print(f"\n{title} — per model, resampler selected on validation\n{t.reindex(MODELS).round(3).to_string()}")
    print("\nper group, (model, resampler) selected on validation\n",
          sg.pivot(index=["dataset", "group"], columns="input", values=["f1_mean", "f1_sd", "pr_auc_mean"]).round(3).to_string())
    print("\nselected tree models:", sel_group[sel_group.group == "trees"].groupby(["dataset", "input"]).model
          .agg(lambda s: s.value_counts().to_dict()).to_dict())
    print("\n", pr.round(4).to_string())


if __name__ == "__main__":
    main()
