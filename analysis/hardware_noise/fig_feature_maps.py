"""SI figures of the features: QRC features of CARDS_2500 (detuning) from the
statevector emulation, the 240-shot emulation and Aquila, for the whole set (Fig 10) and for three
samples (Fig 11).

The inputs are those of the classification (benchmark): statevector emulation = hardware-protocol
emulation (`*_hw.csv`), shot emulation = its 240 shots (`*_hwshots_240.csv`), Aquila = the
hardware features. The three samples of Fig S7 are those
with the lowest, the closest-to-mean and the highest Pearson correlation between their Aquila and
statevector feature vectors (all 390 features).

    python analysis/hardware_noise/fig_feature_maps.py
Output: results/hardware_noise/figures/features_dataset.pdf, features_samples.pdf,
        results/hardware_noise/feature_maps_samples.csv
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATASETS, EMU, OUT, READOUT_TIMES, ROOT, load_features, n_obs

FIG = OUT / "figures"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "pdf.fonttype": 42})
D = DATASETS["detuning"]
SOURCES = [("Statevector emulation", D["hw"]),
           ("Shot emulation (240 shots)", EMU / "detuning_encoding_qpu_samples_12_delta_7.6_deltamax_-9.0_hwshots_240.csv"),
           ("Aquila", D["published"])]
NOBS = n_obs(D["nsites"])
CMAP, NORM = plt.cm.RdBu, plt.Normalize(-1, 1)


def load():
    ids = pd.read_csv(ROOT / "dataset_aquila" / "labels_qpu.csv").iloc[:, 0].to_numpy(int)
    F = [load_features(p, ids) for _, p in SOURCES]
    assert all(f.shape == (5 * NOBS, len(ids)) for f in F) and np.isfinite(np.stack(F)).all()
    return ids, F


def readout_axis(ax, rows):
    for k in range(1, 5):
        ax.axhline(k * rows - 0.5, color=INK, lw=0.5)
    ax.set_yticks([(k + 0.5) * rows for k in range(5)], [f"{t:g} µs" for t in READOUT_TIMES])
    ax.tick_params(axis="y", length=0)


def dataset_figure(ids, F):
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.4), sharex=True, constrained_layout=True)
    for ax, (name, _), f in zip(axes, SOURCES, F):
        im = ax.imshow(f, aspect="auto", interpolation="nearest", cmap=CMAP, norm=NORM)
        readout_axis(ax, NOBS)
        ax.set_title(name, fontsize=8.5, color=INK)
        ax.set_ylabel("readout (12 ⟨Zᵢ⟩, then 66 ⟨ZᵢZⱼ⟩)", fontsize=7)
    axes[-1].set_xlabel(f"datapoint (CARDS_2500, {len(ids)} samples, label-file order)")
    fig.colorbar(im, ax=axes, shrink=0.6, label="feature value")
    fig.savefig(FIG / "features_dataset.pdf")
    plt.close(fig)


def pick_samples(ids, F):
    sv, aq = F[0], F[2]
    r = np.array([np.corrcoef(sv[:, c], aq[:, c])[0, 1] for c in range(len(ids))])
    pick = [int(np.argmin(r)), int(np.argmin(np.abs(r - r.mean()))), int(np.argmax(r))]
    rows = [dict(panel=p, role=role, id=int(ids[c]), pearson_aquila_sv=r[c],
                 pearson_shot_sv=np.corrcoef(sv[:, c], F[1][:, c])[0, 1], mean_r=r.mean(), median_r=np.median(r))
            for p, role, c in zip("abc", ["lowest", "closest to mean", "highest"], pick)]
    return pick, pd.DataFrame(rows)


def samples_figure(F, pick, table):
    fig = plt.figure(figsize=(7.0, 4.4))
    outer = fig.add_gridspec(1, 3, wspace=0.28, left=0.07, right=0.9, top=0.86, bottom=0.12)
    for p, (c, row) in enumerate(zip(pick, table.itertuples())):
        inner = outer[p].subgridspec(1, 3, wspace=0.06)
        for k, ((name, _), f) in enumerate(zip(SOURCES, F)):
            ax = fig.add_subplot(inner[k])
            im = ax.imshow(f[:, c].reshape(5, NOBS).T, aspect="auto", interpolation="nearest", cmap=CMAP, norm=NORM)
            ax.set_xticks(range(5), [f"{t:g}" for t in READOUT_TIMES], fontsize=5.5)
            ax.set_title(name.split(" (")[0].replace(" emulation", "\nemulation").replace("Aquila", "Aquila\n"),
                         fontsize=6.5, color=INK)
            if k == 0:
                ax.set_yticks([5.5, (12 + NOBS) / 2], ["⟨Zᵢ⟩", "⟨ZᵢZⱼ⟩\n(i < j)"], fontsize=6.5)
                ax.tick_params(axis="y", length=0)
            else:
                ax.set_yticks([])
            ax.axhline(11.5, color=INK, lw=0.4)
            if k == 1:
                ax.set_xlabel("evolution time T [µs]", fontsize=7)
        fig.text((outer[p].get_position(fig).x0 + outer[p].get_position(fig).x1) / 2, 0.955,
                 f"({row.panel}) {row.role}: r = {row.pearson_aquila_sv:.2f}", ha="center", fontsize=7.5, color=INK)
    cax = fig.add_axes([0.92, 0.12, 0.015, 0.74])
    fig.colorbar(im, cax=cax, label="feature value")
    fig.savefig(FIG / "features_samples.pdf")
    plt.close(fig)


if __name__ == "__main__":
    ids, F = load()
    dataset_figure(ids, F)
    pick, table = pick_samples(ids, F)
    table.to_csv(OUT / "feature_maps_samples.csv", index=False)
    samples_figure(F, pick, table)
    print(table.to_string(index=False))
