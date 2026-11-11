"""Hardware-protocol model vs ideal statevector, per readout time.

The model runs one program from the ground state per readout time T, with the submitted Aquila
waveforms (Ω ramped down over the last 0.06 µs before T; global detuning constant from t = 0).
The ideal statevector is one continuous pulse with Ω held through the intermediate readouts and
ramped down only at 2.5 µs. Each panel: density of (sv, model) over every observable and sample.

    python analysis/hardware_noise/fig_model_vs_sv.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from common import DATASETS, OUT, READOUT_TIMES, Records, blocks, load_features, obs_types
from figures import FIG, INK, INK2, GRID

# sequential single-hue ramp (reference blue, light -> dark)
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ROWS = [("detuning", "Detuning encoding, CARDS_2500"), ("position", "Position encoding, CARDS_1000")]


def stats(x, y):
    return np.corrcoef(x, y)[0, 1], np.sqrt(np.mean((x - y) ** 2))


def main():
    from figures import C
    fig, ax = plt.subplots(4, 5, figsize=(7.2, 6.4), gridspec_kw={"height_ratios": [1.15, 0.8, 1.15, 0.8]})
    bins = np.linspace(-0.4, 0.4, 61)
    for i, (enc, title) in enumerate(ROWS):
        d = DATASETS[enc]
        ids = Records(d["aquila"]).ids
        k = d["nsites"]
        SV = blocks(load_features(d["sv"], ids), k)
        HW = blocks(load_features(d["hw"], ids), k)
        typ = obs_types(k)
        rs, rh = 2 * i, 2 * i + 1
        for t, T in enumerate(READOUT_TIMES):
            a = ax[rs, t]
            x, y = SV[t].ravel(), HW[t].ravel()
            hb = a.hexbin(x, y, gridsize=40, extent=(-1, 1, -1, 1), cmap=SEQ, norm=LogNorm(vmin=1, vmax=3e4),
                          mincnt=1, linewidths=0)
            a.plot([-1, 1], [-1, 1], color=INK2, lw=0.7, ls="--")
            r, _ = stats(x, y)
            a.text(0.05, 0.95, f"r = {r:.3f}", transform=a.transAxes, va="top", fontsize=6.5, color=INK)
            a.set_xlim(-1, 1); a.set_ylim(-1, 1); a.set_aspect("equal"); a.grid(False)
            a.set_xticks([-1, 0, 1]); a.set_yticks([-1, 0, 1])
            a.tick_params(labelsize=6)
            if t:
                a.set_yticklabels([])
            if i == 0:
                a.set_title(f"T = {T:g} µs", fontsize=8)
            a.set_xlabel("ideal continuous evolution", fontsize=6)
            h = ax[rh, t]
            for c, (o, lab) in enumerate([("Z", "⟨Zᵢ⟩"), ("ZZ", "⟨ZᵢZⱼ⟩")]):
                dlt = (HW[t][typ == o] - SV[t][typ == o]).ravel()
                h.hist(np.clip(dlt, bins[0], bins[-1]), bins=bins, histtype="step", lw=1.3, color=C[c],
                       density=True, label=lab)
                _, e = stats(SV[t][typ == o].ravel(), HW[t][typ == o].ravel())
                h.text(0.97, 0.95 - 0.13 * c, f"{lab} rmse {e:.3f}", transform=h.transAxes, ha="right", va="top",
                       fontsize=5.5, color=INK)
            h.axvline(0, color=INK2, lw=0.6, ls="--")
            h.set_xlim(-0.4, 0.4); h.set_yticks([]); h.grid(False)
            h.tick_params(labelsize=6)
            h.set_xlabel("hardware protocol − ideal", fontsize=6)
        ax[rs, 0].set_ylabel(f"{title}\nhardware protocol", fontsize=6.5)
        ax[rh, 0].set_ylabel("density", fontsize=6.5)
    h, l = ax[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.03, 0.92, 0.97))
    cax = fig.add_axes([0.93, 0.56, 0.012, 0.36])
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label("pairs per bin", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    fig.suptitle("Hardware-protocol model vs ideal statevector: each point is one observable of one sample",
                 fontsize=8.5, color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"hardware_protocol_vs_ideal.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "hardware_protocol_vs_ideal.pdf")


if __name__ == "__main__":
    main()
