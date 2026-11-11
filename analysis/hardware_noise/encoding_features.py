"""Feature files for the encoding / noise-as-resource check on CARDS_1000.

Writes, for both encodings, on the 1,000 CARDS_1000 IDs (one column per datapoint, named by ID):
  <enc>_shotmatched.csv  shot emulation of the hardware-protocol model with, per datapoint and
                         readout, as many samples as Aquila kept (same shot statistics as Aquila)
  <enc>_aquila_resid.csv Aquila features minus their per-feature linear prediction from the model
                         (OLS across samples, label-free): the part of the hardware response the
                         noiseless model does not explain, plus shot noise
  <enc>_shot_resid.csv   the same for the matched shot emulation (pure sampling noise): control
Output: results/hardware_noise/features/.

    python analysis/hardware_noise/encoding_features.py
"""
import numpy as np
import pandas as pd

from analysis import matched_shot_features
from common import DATASETS, OUT, ROOT, Records, blocks, load_features, records_features


def residual(X, M):
    """X minus its per-feature (row) OLS prediction from M."""
    Mc = M - M.mean(1, keepdims=True)
    Xc = X - X.mean(1, keepdims=True)
    beta = (Xc * Mc).sum(1, keepdims=True) / np.maximum((Mc ** 2).sum(1, keepdims=True), 1e-15)
    return Xc - beta * Mc


def main():
    out = OUT / "features"
    out.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(ROOT / "dataset_aquila" / "labels_qpu_pos.csv").ID.to_numpy()
    for enc in ("detuning", "position"):
        d = DATASETS[enc]
        A_rec, H_rec = Records(d["aquila"]), Records(d["hwemu"])
        k = A_rec.nsites
        A = records_features(A_rec, ids)
        S = matched_shot_features(H_rec, ids, A_rec.counts()).reshape(A.shape)
        M = load_features(d["hw"], ids)
        cols = [str(i) for i in ids]
        for name, F in [("shotmatched", S), ("aquila_resid", residual(A, M)), ("shot_resid", residual(S, M))]:
            pd.DataFrame(F, columns=cols).to_csv(out / f"{enc}_{name}.csv", index=False)
        print(enc, "written", A.shape)


if __name__ == "__main__":
    main()
