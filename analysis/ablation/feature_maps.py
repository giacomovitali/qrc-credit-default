"""Run classification/run_classification.py on a fixed transform of its input features.

Every rung of the ablation ladder is a per-datapoint map of an existing feature file, applied in memory
right after loading (so no derived feature files are written) and recorded in the run's manifest under
"feature_map". The classification protocol itself is unchanged.

    python analysis/ablation/feature_maps.py --map MAP [--fit emulation|train] -- <run_classification.py arguments>

MAP (n = number of atoms, inferred from the 5-readout reservoir layout: 390 = 5 x 78, 455 = 5 x 91):
  qrc:t<k...>   reservoir readouts k (1-5 = 0.5 ... 2.5 us), all observables: <Z_i> and <Z_i Z_j>
  qrc:z<k...>   the same readouts, <Z_i> only (the first n rows of each block)
  pca           the 24 variables -> the 11 whitened principal components of the published
                preprocessing (PCA fitted on the emulation group's 24 variables, as the published preprocessing)
  poly:2i       12 features + the 66 products x_i x_j (i < j): one QRC readout's observable structure (78)
  poly:2        full degree 2 (90);  poly:3  full degree 3 (454)
  rff:<s>       390 random Fourier features cos(W x + b), W ~ N(0, 1/(s*sigma)^2), b ~ U(0, 2 pi);
                sigma = median pairwise distance of the emulation group's 12 features (fixed subsample)
  tanh          390 random tanh units tanh(a + b): a = w.(x - mu), w ~ N(0, 1), each unit scaled to unit
                standard deviation on the emulation group; b ~ U(-1, 1)
The random maps are drawn from the run's --seed, so each seed is a different map as well as a different
split. Their outputs lie in [-1, 1] like the reservoir observables (no 1/sqrt(D) factor), since the
readouts see the features unscaled. --fit train (used on CARDS_30000) fits the PCA, sigma,
mu and the tanh scales on the run's own training rows (the split run_classification.py draws) instead
of the emulation group.

Run from the repository root (conda env qrc_class).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "classification"))
import run_classification as rc  # noqa: E402

N_READOUTS, N_RANDOM = 5, 390
EMU12 = "dataset_aquila/emulation_samples.csv"
EMU24 = "results/baselines/features/emulation_samples_24.csv"
N_PCA = 11          # published rule on the emulation group (checked in make_map)
SIGMA_SUBSAMPLE, SIGMA_SEED = 2000, 0


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def qrc_rows(n_features, spec):
    """Row indices of a reservoir feature vector for spec 't<readouts>' or 'z<readouts>'."""
    block = n_features // N_READOUTS
    n = int(round((np.sqrt(8 * block + 1) - 1) / 2))
    if N_READOUTS * block != n_features or n * (n + 1) // 2 != block:
        raise SystemExit(f"{n_features} features are not {N_READOUTS} readouts of n + n(n-1)/2 observables")
    kind, steps = spec[0], [int(c) for c in spec[1:]]
    if kind not in "tz" or not steps or any(not 1 <= k <= N_READOUTS for k in steps) or steps != sorted(set(steps)):
        raise SystemExit(f"bad qrc map '{spec}'")
    width = block if kind == "t" else n
    return [(k - 1) * block + i for k in steps for i in range(width)], {"atoms": n, "readouts": steps,
                                                                        "observables": "Z, ZZ" if kind == "t" else "Z"}


def fit_rows(path, train):
    """The rows the fitted maps learn from, and their manifest entry: the emulation group's file at `path`,
    or the run's training rows `train` (the unmapped input features, one row per datapoint)."""
    if train is None:
        return pd.read_csv(path, index_col="ID").to_numpy(float), {"fit": path, "fit_sha256": _sha(path)}
    ids = ",".join(map(str, train.index)).encode()
    return train.to_numpy(float), {"fit": "training rows", "fit_rows": len(train),
                                   "fit_ids_sha256": hashlib.sha256(ids).hexdigest()}


def make_map(spec, seed, train=None):
    """Return (f, info): f maps a DataFrame (one row per datapoint) to the transformed DataFrame. The fitted
    maps learn from the emulation group, or from `train` (the run's training rows) when it is given."""
    name, _, arg = spec.partition(":")
    info = {"map": spec}
    if name == "qrc":
        def f(X):
            rows, extra = qrc_rows(X.shape[1], arg)
            info.update(extra)
            return X.iloc[:, rows]
        return f, info
    if name == "pca":
        emu, fit_info = fit_rows(EMU24, train)
        ev = PCA(n_components=emu.shape[1]).fit(emu).explained_variance_ratio_
        if ((ev[:-1] - ev[1:]) > 1e-3).sum() != N_PCA:
            raise SystemExit("the published PC-count rule no longer gives 11 components")
        pca = PCA(n_components=N_PCA, whiten=True).fit(emu)
        info |= fit_info | {"n_components": N_PCA,
                 "explained_variance": float(pca.explained_variance_ratio_.sum())}

        def f(X):
            if X.shape[1] != emu.shape[1]:
                raise SystemExit(f"pca takes the 24 variables, got {X.shape[1]} columns")
            return pd.DataFrame(pca.transform(X.to_numpy()), index=X.index)
        return f, info
    if name == "poly":
        degree, interaction = int(arg[0]), arg.endswith("i")
        poly = PolynomialFeatures(degree, interaction_only=interaction, include_bias=False)
        info |= {"degree": degree, "interaction_only": interaction}
        return (lambda X: pd.DataFrame(poly.fit_transform(X.to_numpy()), index=X.index)), info
    if name in ("rff", "tanh"):
        emu, fit_info = fit_rows(EMU12, train)
        d = emu.shape[1]
        rng = np.random.default_rng([4, seed])          # stream 4 = ablation maps; independent of the split's seed use
        info |= fit_info | {"n_out": N_RANDOM, "map_seed": [4, seed]}
        if name == "rff":
            sub = emu[np.random.default_rng(SIGMA_SEED).choice(len(emu), SIGMA_SUBSAMPLE, replace=False)]
            sigma = float(np.median(pdist(sub)))
            scale = float(arg or 1.0)
            W = rng.normal(0.0, 1.0 / (scale * sigma), size=(d, N_RANDOM))
            b = rng.uniform(0.0, 2 * np.pi, size=N_RANDOM)
            info |= {"sigma_median": sigma, "bandwidth_scale": scale}
            return (lambda X: pd.DataFrame(np.cos(X.to_numpy() @ W + b), index=X.index)), info
        mu = emu.mean(0)
        W = rng.normal(0.0, 1.0, size=(d, N_RANDOM))
        W /= ((emu - mu) @ W).std(0)
        b = rng.uniform(-1.0, 1.0, size=N_RANDOM)
        return (lambda X: pd.DataFrame(np.tanh((X.to_numpy() - mu) @ W + b), index=X.index)), info
    raise SystemExit(f"unknown map '{spec}'")


def main():
    if "--" not in sys.argv:
        raise SystemExit(__doc__)
    cut = sys.argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", required=True)
    ap.add_argument("--fit", choices=["emulation", "train"], default="emulation",
                    help="rows the fitted maps (pca, rff, tanh) learn from")
    args = ap.parse_args(sys.argv[1:cut])
    rest = sys.argv[cut + 1:]
    a = rc.parse_args(rest)
    train = None
    if args.fit == "train":     # the same split the driver draws below, on the unmapped features
        train = rc.make_split(a, rc.load_pairs(a.features, a.labels, a.layout)[0])[0]
    f, info = make_map(args.map, a.seed, train)
    load = rc.load_features

    def load_mapped(path, label_ids, layout="auto"):
        X, alignment = load(path, label_ids, layout)
        info["n_in"] = X.shape[1]
        Z = f(X)
        Z.columns = [str(c) for c in range(Z.shape[1])]
        info["n_out"] = Z.shape[1]
        return Z.astype(float), alignment

    rc.load_features = load_mapped
    sys.argv = [sys.argv[0]] + rest
    rc.main(rest)
    a = rc.parse_args(rest)
    path = os.path.join(a.log_dir, f"{a.run_name}.manifest.json")
    with open(path) as fh:
        manifest = json.load(fh)
    manifest["feature_map"] = info
    manifest["command"] = [os.path.relpath(__file__), "--map", args.map, "--fit", args.fit, "--"] + rest
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)


if __name__ == "__main__":
    main()
