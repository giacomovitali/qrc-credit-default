"""Shot-count analysis: F1 vs shots (emulation and Aquila), feature error vs shots, figures.

Reads results/shots/logs/ (run_classification.sh) and feature_error.csv (make_features.py).
Writes results/shots/{f1_runs,f1_summary,f1_selected,f1_paired,mse_fit}.csv and figures/.

    python analysis/shots/analysis.py
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

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "shots"
FIG = OUT / "figures"
NAME = re.compile(r"^(detuning|position)_(sv|model|fc|emu|aquila)(?:_s(\d+))?_(\d+)$")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 4,
    "pdf.fonttype": 42,
})
PANELS = [("detuning", "Detuning, CARDS_2500"), ("position", "Position, CARDS_1000")]


def load_runs():
    rows = []
    for log in sorted((OUT / "logs").glob("*.log")):
        m = NAME.match(log.stem)
        if not m or not log.with_suffix(".predictions.csv.gz").exists():
            continue
        enc, src, s, seed = m.groups()
        for line in log.read_text().splitlines():
            if "Extended metrics: " in line:
                d = json.loads(line.split("Extended metrics: ", 1)[1])
                rows.append(dict(encoding=enc, source=src, shots=int(s) if s else 0, seed=int(seed),
                                 model=d["model"], resampler=d["resampler"], f1=d["f1"], val_f1=d["val_f1"],
                                 roc_auc=d["roc_auc"], pr_auc=d["pr_auc"], balanced_acc=d["balanced_acc"]))
    return pd.DataFrame(rows)


def selected(runs):
    """Per run, the (model, resampler) with the best validation F1 (ties: first in log order)."""
    i = runs.groupby(["encoding", "source", "shots", "seed"], sort=False).val_f1.idxmax()
    return runs.loc[i].reset_index(drop=True)


def summarise(df, by):
    g = df.groupby(by)
    return g.agg(n=("f1", "size"), f1_mean=("f1", "mean"), f1_sd=("f1", "std"), roc_auc_mean=("roc_auc", "mean"),
                 pr_auc_mean=("pr_auc", "mean"), val_f1_mean=("val_f1", "mean")).reset_index()


def paired(sel):
    """Selected-pipeline test F1 at each shot count vs the largest count of the same source
    (paired over seeds, Wilcoxon signed-rank), and Aquila vs emulation at equal shots."""
    rows = []
    for (enc, src), g in sel[sel.source.isin(["emu", "aquila"])].groupby(["encoding", "source"]):
        top = g.shots.max()
        ref = g[g.shots == top].set_index("seed").f1
        for s, h in g.groupby("shots"):
            x = h.set_index("seed").f1.reindex(ref.index)
            d = x - ref
            p = wilcoxon(x, ref).pvalue if (d != 0).any() else 1.0
            rows.append(dict(encoding=enc, comparison=f"{src} s vs {src} s={top}", shots=s,
                             mean_diff=d.mean(), sd_diff=d.std(), p_wilcoxon=p))
    for enc, g in sel.groupby("encoding"):
        for s in sorted(set(g[g.source == "aquila"].shots) & set(g[g.source == "emu"].shots)):
            a = g[(g.source == "aquila") & (g.shots == s)].set_index("seed").f1
            e = g[(g.source == "emu") & (g.shots == s)].set_index("seed").f1.reindex(a.index)
            d = a - e
            rows.append(dict(encoding=enc, comparison="aquila - emu", shots=s, mean_diff=d.mean(), sd_diff=d.std(),
                             p_wilcoxon=wilcoxon(a, e).pvalue if (d != 0).any() else 1.0))
    return pd.DataFrame(rows)


def mse_fit(err):
    """MSE(s) = b² + σ²/s, least squares in 1/s, on the readout-averaged MSE over all observables."""
    rows = []
    e = err[(err.obs == "all") & (err.shots > 0)]
    for (enc, src, ref), g in e.groupby(["encoding", "source", "reference"]):
        m = g.groupby("shots").mse.mean()
        if src == "aquila":
            m = m[m.index <= 160]  # uncapped counts only (position caps from 160 on)
        A = np.c_[np.ones(len(m)), 1.0 / m.index.to_numpy(float)]
        (b2, s2), *_ = np.linalg.lstsq(A, m.to_numpy(), rcond=None)
        rows.append(dict(encoding=enc, source=src, reference=ref, b2=b2, sigma2=s2,
                         s_star=s2 / b2 if b2 > 0 else np.inf, shots_used=",".join(map(str, m.index)),
                         mse_240=g[g.shots == 240].mse.mean(), mse_floor=b2))
    return pd.DataFrame(rows)


def fig_feature_error(err, fits):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    series = [("emu", "model", "shot emulation vs statevector", C[0], "-"), ("aquila", "model", "Aquila vs statevector", C[1], "-")]
    for j, (enc, title) in enumerate(PANELS):
        e = err[(err.encoding == enc) & (err.obs == "all")]
        for src, ref, lab, col, ls in series:
            g = e[(e.source == src) & (e.reference == ref) & (e.shots > 0)].groupby("shots").mse.mean()
            ax[j].plot(g.index, g.values, ls, marker="o", color=col, label=lab, lw=1.6 if ls == "-" else 1.1)
        f = fits[(fits.encoding == enc) & (fits.source == "aquila") & (fits.reference == "model")].iloc[0]
        ax[j].axhline(f.b2, color=C[1], ls=":", lw=1)
        ax[j].text(0.97, 0.93, f"Aquila floor b² = {f.b2:.4f}, s* = σ²/b² ≈ {f.s_star:.0f}",
                   transform=ax[j].transAxes, ha="right", va="top", color=INK2, fontsize=7)
        ax[j].axvline(240, color=GRID, lw=3, zorder=0)
        ax[j].set_xscale("log"), ax[j].set_yscale("log")
        ax[j].set_xlabel("shots per readout s")
        ax[j].set_title(title, fontsize=8.5, color=INK)
    ax[0].set_ylabel("feature MSE (mean over readouts)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "feature_error_vs_shots.pdf", bbox_inches="tight"), plt.close(fig)


def fig_f1(sel_sum):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)
    refs = [("model", "statevector emulation (∞ shots)", INK2), ("fc", "12 preprocessed features (FC)", "#9b9a97")]
    for j, (enc, title) in enumerate(PANELS):
        s = sel_sum[sel_sum.encoding == enc]
        for k, (src, lab) in enumerate([("emu", "shot emulation"), ("aquila", "Aquila (subsampled records)")]):
            g = s[s.source == src].sort_values("shots")
            ax[j].fill_between(g.shots, g.f1_mean - g.f1_sd, g.f1_mean + g.f1_sd, color=C[k], alpha=0.15, lw=0)
            ax[j].plot(g.shots, g.f1_mean, "-o", color=C[k], label=lab)
        for src, lab, col in refs:
            r = s[s.source == src].iloc[0]
            ax[j].axhline(r.f1_mean, color=col, ls="--", lw=1.2, label=lab)
        ax[j].axvline(240, color=GRID, lw=3, zorder=0)
        ax[j].set_xscale("log")
        ax[j].set_xlabel("shots per readout s")
        ax[j].set_title(title, fontsize=8.5, color=INK)
    ax[0].set_ylabel("test F1, validation-selected pipeline\n(mean ± SD, 10 seeds)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(FIG / "f1_vs_shots.pdf", bbox_inches="tight"), plt.close(fig)


def fig_f1_models(runs):
    """Per model: best resampler on validation per seed, test F1 mean vs shots."""
    i = runs.groupby(["encoding", "source", "shots", "seed", "model"], sort=False).val_f1.idxmax()
    m = runs.loc[i].groupby(["encoding", "source", "shots", "model"]).f1.mean().reset_index()
    models = ["KNN", "SGD", "NBT", "LM", "P", "LSVM"]
    fig, ax = plt.subplots(2, 6, figsize=(10, 3.6), sharex=True, sharey="row")
    for j, (enc, title) in enumerate(PANELS):
        for k, mod in enumerate(models):
            a = ax[j, k]
            for c, src in enumerate(["emu", "aquila"]):
                g = m[(m.encoding == enc) & (m.source == src) & (m.model == mod)].sort_values("shots")
                a.plot(g.shots, g.f1, "-o", color=C[c], lw=1.5, ms=3, label={"emu": "shot emulation", "aquila": "Aquila"}[src])
            for src, col in (("model", INK2), ("fc", "#9b9a97")):
                v = m[(m.encoding == enc) & (m.source == src) & (m.model == mod)].f1
                if len(v):
                    a.axhline(v.iloc[0], color=col, ls="--", lw=1, label={"model": "statevector emulation", "fc": "FC"}[src])
            a.set_xscale("log")
            if j == 0:
                a.set_title(mod, fontsize=8.5, color=INK)
            if j == 1:
                a.set_xlabel("shots")
        ax[j, 0].set_ylabel(f"{title.split(',')[0]}: test F1")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "f1_vs_shots_per_model.pdf", bbox_inches="tight"), plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    runs = load_runs()
    runs.to_csv(OUT / "f1_runs.csv", index=False)
    print(runs.groupby(["encoding", "source"]).seed.nunique().unstack(), "\n", len(runs), "rows")
    sel = selected(runs)
    sel.to_csv(OUT / "f1_selected.csv", index=False)
    sel_sum = summarise(sel, ["encoding", "source", "shots"])
    summ = summarise(runs, ["encoding", "source", "shots", "model", "resampler"])
    summ.to_csv(OUT / "f1_summary.csv", index=False)
    sel_sum.to_csv(OUT / "f1_selected_summary.csv", index=False)
    pr = paired(sel)
    pr.to_csv(OUT / "f1_paired.csv", index=False)
    err = pd.read_csv(OUT / "feature_error.csv")
    fits = mse_fit(err)
    fits.to_csv(OUT / "mse_fit.csv", index=False)
    fig_feature_error(err, fits)
    fig_f1(sel_sum)
    fig_f1_models(runs)
    pd.set_option("display.width", 200)
    print(sel_sum.round(3).to_string())
    print(pr.round(3).to_string())
    print(fits.round(4).to_string())


if __name__ == "__main__":
    main()
