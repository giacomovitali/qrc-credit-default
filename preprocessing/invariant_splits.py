"""CARDS_30000 split seeds whose train-only fit reproduces the published transforms.

For a split seed s (run_classification.py --split random --seed s, QPU pair first, as in the baseline, benchmark and ablation runs), every
transform that the published pipeline fitted on a group of rows is refitted, with the published code, on
the seed's 20,720 training rows only, and compared with the one the distributed feature files used:

  (a) MaxAbsScaler + PCA + clustering (preprocessing/utils.py): the same clusters in the same order (the
      order fixes the feature-to-atom mapping) and the 12 aggregated features of all 29,601 rows
      bit-identical to the published transform;
  (b) the scaler of the 24 variables (the *24 pipelines, the ablation pca rung input): bit-identical on all rows;
  (c) the reservoir input scaling of quantum_evolution/run_reservoir.jl (each set shifted by its global
      minimum and divided by the largest shifted value of the reference set, the QPU group): the
      training rows give the same minimum and the same factor;
  (d) the input scaling of "classical_rc/CRC.jl": max |x| of
      the emulation group; the training rows give the same value.

Pre-registered rule : CARDS_30000 is evaluated on the first K = 10 seeds, in
ascending order from 1, that pass (a)-(d). The rule looks only at the training rows' features, never at
labels, predictions or scores.

    python preprocessing/invariant_splits.py [--k 10] [--out results/preprocessing]

Writes <out>/invariant_splits.csv (every scanned seed and each check) and the seed list
configs/cards30000_seeds.txt, read by the baseline, benchmark and ablation and analysis scripts. Run from the
repository root (conda env qrc_class).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refit_sensitivity as R          # published-code preprocess() and the split driver

SEED_FILE = "configs/cards30000_seeds.txt"     # read with seeds.py


def reservoir_scaling(X):
    """(minimum, factor) of run_reservoir.jl: shift_to_zero and scale_factor on one set."""
    m = X.min()
    return m, np.abs(X - m).max()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--k", type=int, default=10, help="number of seeds to select")
    ap.add_argument("--out", default="results/preprocessing")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw = pd.read_csv("dataset_aquila/default_of_credit_card_clients.csv", index_col="ID")
    clean = R.pre.clean_dataset(raw)
    X_emu, X_qpu, _, _ = R.pre.split_dataset(clean, 2571)
    X_all = clean.loc[:, clean.columns[:-1]]

    # Published transforms (fit on the emulation group); the distributed files store them to 1e-16
    pub, pub_clusters, _ = R.preprocess(X_emu, X_all)
    pub24 = R.pre.scale_features(X_emu.copy(), X_all.copy(), R.CATEG)[1].astype(float)
    # The 12-feature files as the reservoir and CRC drivers read them
    qpu = pd.read_csv("dataset_aquila/qpu_samples.csv", index_col="ID", float_precision="round_trip")
    emu = pd.read_csv("dataset_aquila/emulation_samples.csv", index_col="ID", float_precision="round_trip")
    files = pd.concat([qpu, emu])
    res_min, res_factor = reservoir_scaling(qpu.values)          # reference set = qpu_samples (configs/reservoir/)
    if reservoir_scaling(emu.values)[0] != res_min:
        raise SystemExit("the two groups no longer share their minimum")
    crc_scale = max(abs(emu.values.max()), abs(emu.values.min()))

    a = R.rc.parse_args(["--features", "dataset_aquila/qpu_samples.csv", "dataset_aquila/emulation_samples.csv",
                         "--labels", "dataset_aquila/labels_qpu.csv", "dataset_aquila/labels_emu.csv",
                         "--seed", "1", "--run-name", "refit"])
    pairs = R.rc.load_pairs(a.features, a.labels, a.layout)[0]
    rows, selected, seed = [], [], 0
    while len(selected) < args.k:
        seed += 1
        a.seed = a.split_seed = seed
        X_tr = R.rc.make_split(a, pairs)[0]
        refit, clusters, npc = R.preprocess(X_all.loc[X_tr.index], X_all)
        same_order = clusters == pub_clusters
        same_sets = {frozenset(v) for v in clusters.values()} == {frozenset(v) for v in pub_clusters.values()}
        delta = np.abs(refit.loc[pub.index, pub.columns].values - pub.values).max() if same_order else np.nan
        refit24 = R.pre.scale_features(X_all.loc[X_tr.index].copy(), X_all.copy(), R.CATEG)[1].astype(float)
        t = files.loc[X_tr.index].values
        t_min, t_factor = reservoir_scaling(t)
        row = {"seed": seed, "n_pc": npc, "n_clusters": len(clusters), "same_clusters": same_sets,
               "same_order": same_order, "max_abs_delta_12": delta,
               "scaler24_identical": bool((refit24.loc[pub24.index, pub24.columns].values == pub24.values).all()),
               "reservoir_min_same": t_min == res_min, "reservoir_factor_same": t_factor == res_factor,
               "crc_scale_same": max(abs(t.max()), abs(t.min())) == crc_scale}
        row["passes"] = bool(same_order and delta == 0 and row["scaler24_identical"] and row["reservoir_min_same"]
                             and row["reservoir_factor_same"] and row["crc_scale_same"])
        rows.append(row)
        if row["passes"]:
            selected.append(seed)
            print(f"seed {seed}: passes ({len(selected)}/{args.k})")

    df = pd.DataFrame(rows)
    out = os.path.join(args.out, "invariant_splits.csv")
    df.to_csv(out, index=False)
    print(f"scanned seeds 1-{seed}: same clusters {df.same_clusters.sum()}, same order {df.same_order.sum()}, "
          f"identical 12 features {(df.max_abs_delta_12 == 0).sum()}, all checks {df.passes.sum()}; wrote {out}")
    with open(SEED_FILE, "w") as f:
        f.write("# CARDS_30000 split seeds (pre-registered): the first "
                f"{args.k} seeds whose train-only\n# fit reproduces the published transforms "
                "(preprocessing/invariant_splits.py). CARDS_2500/1000 use seeds 1-10.\n")
        f.write(" ".join(map(str, selected)) + "\n")
    print(f"wrote {SEED_FILE}: {' '.join(map(str, selected))}")


if __name__ == "__main__":
    main()
