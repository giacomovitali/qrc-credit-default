"""Shared loaders for the hardware-noise analysis.

Records come from `export_bits.jl` (flat UInt32 bitstrings plus an index); feature
files are reservoir outputs (one column per datapoint, named by ID, one row per
feature). Features per readout: <Z_i> for each atom, then <Z_i Z_j> for i < j in
row-major order, the layout of the reservoir output.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "hardware_noise"
BITS = OUT / "bits"
EMU = ROOT / "results" / "emulation"
READOUT_TIMES = np.array([0.5, 1.0, 1.5, 2.0, 2.5])  # µs, total evolution per readout

DATASETS = {
    # name: (encoding, n atoms, Aquila records, emulated hardware-protocol records,
    #        ideal statevector, hardware-protocol exact, Aquila features, statevector with integrator-step readouts)
    "detuning": dict(
        nsites=12,
        aquila="aquila_det", hwemu="hwemu_det",
        sv=EMU / "detuning_encoding_qpu_samples_12_delta_7.6_deltamax_-9.0_sv_exact.csv",
        hw=EMU / "detuning_encoding_qpu_samples_12_delta_7.6_deltamax_-9.0_hw.csv",
        published=ROOT / "datasets" / "detuning_encoding_qpu_samples_240shots_real_aquila.csv",
        sv_integrator=ROOT / "dataset_aquila" / "qpu_samples_sv.csv",
    ),
    "position": dict(
        nsites=13,
        aquila="aquila_pos", hwemu="hwemu_pos",
        sv=EMU / "position_encoding_qpu_samples_pos_12_delta_2.0_lambda_1.0_sv_exact.csv",
        hw=EMU / "position_encoding_qpu_samples_pos_12_delta_2.0_lambda_1.0_hw.csv",
        published=ROOT / "datasets" / "position_encoding_qpu_samples_240shots_real_aquila_lambda1.0_delta2.0.csv",
        sv_integrator=ROOT / "datasets" / "position_encoding_qpu_samples_pos_12_delta_2.0_lambda_1.0_sv.csv",
    ),
}


def n_obs(nsites):
    return nsites + nsites * (nsites - 1) // 2


def obs_types(nsites):
    """'Z' or 'ZZ' for each observable of one readout block."""
    return np.array(["Z"] * nsites + ["ZZ"] * (nsites * (nsites - 1) // 2))


class Records:
    """Measurement records per (datapoint, readout)."""

    def __init__(self, prefix):
        self.bits = np.fromfile(BITS / f"{prefix}_bits.u32", dtype=np.uint32)
        self.index = pd.read_csv(BITS / f"{prefix}_index.csv")
        self.nsites = int(self.index.nsites.iloc[0])
        self._loc = {(r.id, r.readout): (r.offset, r.count) for r in self.index.itertuples()}
        self.ids = np.array(sorted(self.index.id.unique()))

    def get(self, id_, readout):
        off, cnt = self._loc[(int(id_), int(readout))]
        return self.bits[off:off + cnt]

    def counts(self):
        """DataFrame id × readout of the number of records."""
        return self.index.pivot(index="id", columns="readout", values="count")

    def spins(self, id_, readout):
        """records × atoms matrix of Z = ±1 (+1 = ground)."""
        r = self.get(id_, readout)
        return 1.0 - 2.0 * ((r[:, None] >> np.arange(self.nsites, dtype=np.uint32)) & 1)


def observables(z):
    """<Z_i> and <Z_i Z_j> (i < j, row-major) from a records × atoms ±1 matrix."""
    n, k = z.shape
    iu = np.triu_indices(k, 1)
    return np.concatenate([z.mean(0), (z.T @ z / n)[iu]])


def records_features(rec, ids=None, n=None, rng=None):
    """features × datapoints array from records; first `n` records, or a random
    subsample of `n` without replacement if `rng` is given."""
    ids = rec.ids if ids is None else ids
    out = np.empty((5 * n_obs(rec.nsites), len(ids)))
    for c, id_ in enumerate(ids):
        blocks = []
        for r in range(1, 6):
            z = rec.spins(id_, r)
            if n is not None:
                z = z[rng.choice(len(z), n, replace=False)] if rng is not None else z[:n]
            blocks.append(observables(z))
        out[:, c] = np.concatenate(blocks)
    return out


def load_features(path, ids=None):
    """features × datapoints array of a reservoir output, columns in `ids` order."""
    df = pd.read_csv(path)
    df.columns = df.columns.astype(int)
    return df.to_numpy(float) if ids is None else df[list(ids)].to_numpy(float)


def blocks(F, nsites):
    """Reshape features × datapoints to readout × observable × datapoint."""
    return F.reshape(5, n_obs(nsites), F.shape[1])
