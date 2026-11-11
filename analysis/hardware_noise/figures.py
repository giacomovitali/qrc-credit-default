"""Hardware-noise figures (vector PDF) from the CSVs written by analysis.py, postselection.py and
noise_model_eval.py. Output: results/hardware_noise/figures/.

    python analysis/hardware_noise/figures.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUT, READOUT_TIMES

FIG = OUT / "figures"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical slots 1-4, fixed order
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 4,
    "pdf.fonttype": 42,
})
PANELS = [("detuning", "CARDS_2500", "Detuning, CARDS_2500"),
          ("position", "CARDS_1000", "Position, CARDS_1000")]
SERIES = [(("aquila", "hw"), "Aquila vs statevector emulation", 0),
          (("shot", "hw"), "Shot emulation vs statevector (shot-noise ceiling)", 2)]


def agreement_vs_time():
    a = pd.read_csv(OUT / "agreement_per_sample.csv")
    r = pd.read_csv(OUT / "retention.csv")
    fig, ax = plt.subplots(3, 2, figsize=(7.0, 6.6), sharex=True)
    for j, (enc, sub, title) in enumerate(PANELS):
        for i, typ in enumerate(["Z", "ZZ"]):
            for (x, y), lab, k in SERIES:
                g = a[(a.encoding == enc) & (a.subset == sub) & (a.x == x) & (a.y == y) & (a.obs == typ) &
                      (a.metric == "pearson")].sort_values("readout_us")
                ax[i, j].fill_between(g.readout_us, g.q25, g.q75, color=C[k], alpha=0.15, lw=0)
                ax[i, j].plot(g.readout_us, g["median"], "-o", color=C[k], label=lab)
            ax[i, j].set_ylim(-0.2, 1.02)
            ax[i, j].set_ylabel(f"per-sample Pearson r, ⟨{'Zᵢ' if typ == 'Z' else 'ZᵢZⱼ'}⟩")
        for (x, y), lab, k in SERIES:
            g = r[(r.encoding == enc) & (r.subset == sub) & (r.x == x) & (r.y == y) & (r.obs == "all")].sort_values("readout_us")
            ax[2, j].errorbar(g.readout_us, g.slope, yerr=[g.slope - g.ci_lo, g.ci_hi - g.slope], fmt="o", color=C[k],
                              capsize=0, label=lab)
        # No decay fit drawn: the timescale depends on the agreement metric (decay_fit.csv, findings §2) and
        # the detuning series is not exponential; the text quotes it.
        ax[2, j].set_ylim(-0.05, 1.1)
        ax[2, j].set_ylabel("signal retention a(T)")
        ax[2, j].set_xlabel("evolution time T [µs]")
        ax[2, j].set_xticks(READOUT_TIMES)
        ax[0, j].set_title(title, fontsize=8.5, color=INK)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "noise_agreement_vs_time.pdf")
    plt.close(fig)


def bias_variance():
    c = pd.read_csv(OUT / "bias_variance_curves.csv")
    f = pd.read_csv(OUT / "bias_variance_fit.csv")
    panels = [("detuning", "CARDS_2500", "Detuning, CARDS_2500"), ("detuning", "fig8_10", "Detuning, 10 samples (≈950 records)"),
              ("position", "CARDS_1000", "Position, CARDS_1000")]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    for j, (enc, sub, title) in enumerate(panels):
        for k, (src, lab) in enumerate([("aquila", "Aquila"), ("shot", "Shot emulation")]):
            g = c[(c.encoding == enc) & (c.subset == sub) & (c.source == src) & (c.ref == "hw") & (c.obs == "all") &
                  (c.readout_us == "all")].sort_values("shots")
            ft = f[(f.encoding == enc) & (f.subset == sub) & (f.source == src) & (f.ref == "hw") & (f.obs == "all") &
                   (f.readout_us == "all")].iloc[0]
            s = np.geomspace(4, 1200, 200)
            ax[j].plot(s, max(ft.b2_fit, 0) + ft.sigma2_fit / s, "-", color=C[k], lw=1.2)
            ax[j].plot(g.shots, g.mse, "o", color=C[k], label=lab, markersize=5)
            if src == "aquila":
                ax[j].axhline(ft.b2_fit, color=C[k], ls=":", lw=1)
                ax[j].axvline(ft.s_star, color=INK2, ls="--", lw=0.8)
                ax[j].text(ft.s_star * 1.1, 0.12, f"s* = σ²/b² ≈ {ft.s_star:.0f}", color=INK2, fontsize=7)
                ax[j].text(5, ft.b2_fit * 0.75, f"b² = {ft.b2_fit:.4f}", color=INK2, fontsize=7, va="top")
        ax[j].axvline(240, color=GRID, lw=3, zorder=0)
        ax[j].set_xscale("log"); ax[j].set_yscale("log")
        ax[j].set_xlim(4, 1200); ax[j].set_ylim(5e-4, 0.4)
        ax[j].set_title(title, fontsize=8.5)
        ax[j].set_xlabel("shots per readout s")
    ax[0].set_ylabel("MSE vs statevector emulation")
    ax[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "noise_bias_variance.pdf")
    plt.close(fig)


def error_budget():
    e = pd.read_csv(OUT / "error_decomposition.csv")
    parts = [("offset", "per-feature offset"), ("attenuation", "signal attenuation"),
             ("residual", "sample-dependent residual"), ("shot_noise", "shot noise (median 222 / 195 records)")]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for j, (enc, sub, title) in enumerate(PANELS):
        g = e[(e.encoding == enc) & (e.subset == sub) & (e.source == "aquila") & (e.ref == "hw") & (e.obs == "all")].sort_values("readout_us")
        bottom = np.zeros(len(g))
        for k, (col, lab) in enumerate(parts):
            ax[j].bar(np.arange(len(g)), g[col].clip(lower=0), 0.6, bottom=bottom, color=C[k], label=lab,
                      edgecolor="white", linewidth=1)
            bottom += g[col].clip(lower=0).to_numpy()
        ax[j].plot(np.arange(len(g)), g.ref_signal_var, "_", color=INK, markersize=18, mew=1.5, label="model signal variance")
        ax[j].set_xticks(np.arange(len(g)), [f"{t:g}" for t in g.readout_us])
        ax[j].set_xlabel("evolution time T [µs]")
        ax[j].set_title(title, fontsize=8.5)
        ax[j].grid(axis="x", visible=False)
    ax[0].set_ylabel("mean squared error vs statevector")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(FIG / "noise_error_budget.pdf")
    plt.close(fig)


def postselection():
    t = pd.read_csv(OUT / "postselection_tasks.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    bins = np.linspace(0.5, 1.0, 51)
    for k, enc in enumerate(["detuning", "position"]):
        g = t[t.encoding == enc]
        ax.hist(g.kept_fraction, bins=bins, histtype="step", lw=1.5, color=C[k],
                label=f"{enc} (median {g.kept_fraction.median():.2f}, {len(g)} tasks)")
    ax.set_xlabel("fraction of register copies kept (all traps filled)")
    ax.set_ylabel("tasks")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "noise_postselection.pdf")
    plt.close(fig)


def noise_model():
    p = OUT / "noise_model_eval.csv"
    if not p.exists():
        return
    m = pd.read_csv(p)
    m = m[m.readout_us == "all"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 4.2))
    for j, enc in enumerate(["detuning", "position"]):
        g = m[(m.encoding == enc) & m.setting.str.endswith("+spam")].sort_values("b2_vs_aquila", ascending=False)
        base = m[(m.encoding == enc) & (m.setting == "nominal")].b2_vs_aquila.iloc[0]
        y = np.arange(len(g))
        cols = [C[1] if s.startswith("spec_combined") else C[2] if s.startswith("nominal") else C[0] for s in g.setting]
        ax[j].barh(y, g.b2_vs_aquila, 0.65, color=cols)
        ax[j].axvline(base, color=INK, ls="--", lw=1)
        ax[j].text(base, len(g) - 0.3, " statevector emulation", fontsize=6.5, color=INK2, va="bottom")
        ax[j].set_yticks(y, [s.replace("+spam", "") for s in g.setting], fontsize=6.5)
        ax[j].set_xlabel("b² between model and Aquila (100 samples)")
        ax[j].set_title(f"{enc} (all settings + measurement error)", fontsize=8.5)
        ax[j].grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "noise_model.pdf")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    agreement_vs_time()
    bias_variance()
    error_budget()
    postselection()
    noise_model()
    print("figures in", FIG)
