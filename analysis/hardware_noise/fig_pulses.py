"""Figure 4: Ω(t) and the detuning of every site in the programs executed on Aquila, for one sample of
each encoding (read from the raw-record export of export_raw.py).

Each readout time T is a separate program of length T from the ground state. The figure shows the
T = 2.5 µs program; the dotted lines mark where Ω ramps down in the four shorter programs. The global
detuning is constant from t = 0, and only Ω and the local-detuning term are ramped (0.06 µs).

    python analysis/hardware_noise/fig_pulses.py
Output: results/hardware_noise/figures/fig4_pulses.pdf, results/hardware_noise/fig4_programs.csv
"""
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "hardware_noise"
RAW = OUT / "raw"
DET_ID, POS_ID = 1, 10021        # the samples drawn: one per encoding
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "pdf.fonttype": 42})


def wave(w):
    """(times in µs, values in rad/µs) of a piecewise-linear waveform in SI units."""
    return np.array(w["times"], float) * 1e6, np.array(w["values"], float) / 1e6


def programs(prefix, id_):
    """The executed programs (Hamiltonians) of one datapoint, from export_raw.py, in readout order."""
    with gzip.open(RAW / f"{prefix}_programs.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    return [r["program"]["effective_hamiltonian"]["rydberg"]
            for r in sorted((r for r in rows if r["id"] == id_), key=lambda r: r["readout"])]


def detuning_programs():
    out = []
    for h in programs("aquila_det", DET_ID):
        loc = h["detuning"]["local"]
        out.append(dict(omega=wave(h["rabi_frequency_amplitude"]["global"]), glob=wave(h["detuning"]["global"]),
                        local=wave(loc), coeff=np.array(loc["lattice_site_coefficients"], float)[:12]))
    return sorted(out, key=lambda p: p["omega"][0][-1])


def position_programs():
    out = []
    for h in programs("aquila_pos", POS_ID):
        assert "local" not in h["detuning"]
        out.append(dict(omega=wave(h["rabi_frequency_amplitude"]["global_"]), glob=wave(h["detuning"]["global_"])))
    return sorted(out, key=lambda p: p["omega"][0][-1])


def main():
    det, pos = detuning_programs(), position_programs()
    assert [p["omega"][0][-1] for p in det] == [p["omega"][0][-1] for p in pos] == [0.5, 1.0, 1.5, 2.0, 2.5]
    rows = []
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.2), sharex=True)
    for col, (name, progs) in enumerate([("(a) position encoding", pos), ("(b) detuning encoding", det)]):
        last = progs[-1]
        for p in progs[:-1]:
            ax[0, col].axvline(p["omega"][0][-1], color=INK2, lw=0.7, ls=":")
        ax[0, col].plot(*last["omega"], color="#2a78d6", lw=1.6)
        ax[0, col].set_title(name, fontsize=8.5, color=INK)
        ax[0, col].set_ylabel("Ω(t) [rad/µs]")
        tg, vg = last["glob"]
        if "local" in last:
            tl, vl = last["local"]
            assert np.array_equal(tg, tl)
            cmap = plt.cm.viridis
            for i, c in enumerate(last["coeff"]):
                ax[1, col].plot(tl, vg + c * vl, color=cmap(i / 11), lw=1.0)
            ax[1, col].plot(tg, vg, color=INK, lw=1.0, ls="--", label="global term (Δg = 7.64 rad/µs)")
            ax[1, col].plot([], [], color=cmap(0.5), lw=1.0, label="Δᵢ(t), one line per site (12)")
            ax[1, col].legend(loc="center", bbox_to_anchor=(0.5, 0.66), fontsize=6.5, frameon=False)
            ax[1, col].set_ylabel("Δᵢ(t) [rad/µs]")
        else:
            ax[1, col].plot(tg, vg, color=INK, lw=1.4, label="Δ(t), same on all sites")
            ax[1, col].legend(loc="center", bbox_to_anchor=(0.5, 0.66), fontsize=6.5, frameon=False)
            ax[1, col].set_ylabel("Δ(t) [rad/µs]")
        ax[1, col].set_xlabel("time t [µs]")
        for p in progs:
            rows.append(dict(encoding=name.split()[1], T_us=p["omega"][0][-1], omega_times_us=p["omega"][0].tolist(),
                             omega_rad_per_us=p["omega"][1].round(4).tolist(), global_detuning=p["glob"][1].round(4).tolist(),
                             local_detuning=p["local"][1].round(4).tolist() if "local" in p else "",
                             local_coefficients=p["coeff"].round(4).tolist() if "coeff" in p else ""))
    ax[1, 0].set_ylim(-0.2, 2.4)
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "fig4_pulses.pdf")
    pd.DataFrame(rows).to_csv(OUT / "fig4_programs.csv", index=False)
    print(pd.DataFrame(rows)[["encoding", "T_us", "omega_times_us", "global_detuning", "local_detuning"]].to_string(index=False))


if __name__ == "__main__":
    main()
