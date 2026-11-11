"""The 24 preprocessed variables before feature aggregation.

The published preprocessing (preprocessing/preprocessing.py) cleans the dataset, splits it into the
emulation and QPU groups, scales the 21 numerical columns with a MaxAbsScaler fitted on the emulation
group, one-hot encodes SEX and MARRIAGE (dropping SEX_2 and MARRIAGE_3), and then averages
clusters of these 24 columns into the 12 aggregated features. Only the aggregated features were
saved. This script writes the 24-column table with the same fit scope (as the published preprocessing), one row per
datapoint keyed by ID, for the same three datasets:

  emulation_samples_24.csv (27,030), qpu_samples_24.csv (CARDS_2500), qpu_samples_pos_24.csv (CARDS_1000)

and checks that aggregating it with the published clusters gives the distributed 12-feature files.

    python preprocessing/make_features24.py [--out results/baselines/features]

Run from the repository root (conda env qrc_class). Nothing outside --out is written.
"""

import argparse
import contextlib
import importlib.util
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("preprocessing_utils", "preprocessing/utils.py")
pre = importlib.util.module_from_spec(spec); spec.loader.exec_module(pre)
plt.savefig = lambda *a, **k: None       # pca_correlation_graph would overwrite results/preprocessing/corr_circle.png


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="results/baselines/features")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw = pd.read_csv("dataset_aquila/default_of_credit_card_clients.csv", index_col="ID")
    X_emu, X_qpu, _, _ = pre.split_dataset(pre.clean_dataset(raw), 2571)
    emu, qpu = pre.scale_features(X_emu.copy(), X_qpu.copy(), ["SEX", "MARRIAGE"])
    emu, qpu = emu.astype(float), qpu.astype(float)
    pos_ids = pd.read_csv("dataset_aquila/labels_qpu_pos.csv", index_col="ID").index
    tables = {"emulation_samples_24.csv": emu, "qpu_samples_24.csv": qpu, "qpu_samples_pos_24.csv": qpu.loc[pos_ids]}

    # Aggregating with the published clusters must give the distributed 12-feature files
    with contextlib.redirect_stdout(io.StringIO()):
        clusters = pre.pca_correlation_graph(emu, emu.columns)
    agg_emu, agg_qpu = pre.aggregate_features(clusters, emu.copy(), qpu.copy())
    for name, agg in (("emulation_samples.csv", agg_emu), ("qpu_samples.csv", agg_qpu)):
        disk = pd.read_csv(os.path.join("dataset_aquila", name), index_col="ID")
        agg.columns = disk.columns
        print(f"re-aggregated vs dataset_aquila/{name}: max |Δ| = "
              f"{np.abs(agg.loc[disk.index].values - disk.values).max():.1e}")

    for name, X in tables.items():
        X.index.name = "ID"
        X.to_csv(os.path.join(args.out, name))
        print(f"wrote {os.path.join(args.out, name)}: {X.shape[0]} rows × {X.shape[1]} columns, "
              f"range [{X.values.min():.2f}, {X.values.max():.2f}]")
    print("columns:", " ".join(emu.columns))


if __name__ == "__main__":
    main()
