"""Compare each noise-model setting (noise_model.jl) with the Aquila records.

For every setting, with and without measurement errors (Aquila whitepaper: P(g read as r) = 0.01,
P(r read as g) = 0.08, applied analytically to the expectation values), per readout and pooled:
  b2_vs_aquila   mean (Aquila - model)^2 minus the Aquila shot-noise term S^2/N: the squared
                 hardware-model discrepancy the setting leaves unexplained
  retention      slope of the centred Aquila features on the centred model features
  pearson        median per-sample Pearson r across observables

    python analysis/hardware_noise/noise_model_eval.py
"""
import numpy as np
import pandas as pd

from common import DATASETS, OUT, READOUT_TIMES, Records, blocks, records_features

EPS_G, EPS_R = 0.01, 0.08


def spam(F, k):
    """Apply independent per-atom readout flips to readout × observable × sample features."""
    c, d = 1 - EPS_G - EPS_R, EPS_R - EPS_G
    out = F.copy()
    Z = F[:, :k]
    out[:, :k] = c * Z + d
    i, j = np.triu_indices(k, 1)
    out[:, k:] = c ** 2 * F[:, k:] + c * d * (Z[:, i] + Z[:, j]) + d ** 2
    return out


def metrics(A, M, N):
    S2 = (1 - A ** 2) * N / (N - 1)
    b2 = ((A - M) ** 2 - S2 / N).mean()
    Ac, Mc = A - A.mean(-1, keepdims=True), M - M.mean(-1, keepdims=True)
    slope = (Ac * Mc).sum() / (Mc ** 2).sum()
    return b2, slope


def per_sample_r(A, M):
    a, m = A - A.mean(0), M - M.mean(0)
    return np.median((a * m).sum(0) / np.sqrt((a ** 2).sum(0) * (m ** 2).sum(0)))


def main():
    nm = OUT / "noise_model"
    rows = []
    for enc in ("detuning", "position"):
        d = DATASETS[enc]
        rec = Records(d["aquila"])
        k = rec.nsites
        files = sorted(nm.glob(f"{enc}_*.csv"))
        ids = pd.read_csv(files[0], nrows=0).columns.astype(int).to_numpy()
        A = blocks(records_features(rec, ids), k)
        N = rec.counts().loc[ids].to_numpy().T[:, None, :].astype(float)
        for f in files:
            setting = f.stem[len(enc) + 1:]
            M0 = blocks(pd.read_csv(f)[[str(i) for i in ids]].to_numpy(float), k)
            for label, M in ((setting, M0), (setting + "+spam", spam(M0, k))):
                for ti, T in list(enumerate(READOUT_TIMES)) + [(None, "all")]:
                    sl = slice(None) if ti is None else slice(ti, ti + 1)
                    b2, slope = metrics(A[sl], M[sl], N[sl])
                    r = np.mean([per_sample_r(A[t], M[t]) for t in range(5)][sl])
                    rows.append(dict(encoding=enc, setting=label, readout_us=T, b2_vs_aquila=b2,
                                     retention=slope, pearson_median=r, n_samples=len(ids)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "noise_model_eval.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.precision", 4):
        piv = df.pivot_table(index=["encoding", "setting"], columns="readout_us", values="b2_vs_aquila")
        print(piv.sort_values("all"))


if __name__ == "__main__":
    main()
