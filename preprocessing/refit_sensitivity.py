"""Sensitivity of the preprocessing fit scope on CARDS_30000.

The published preprocessing fits the MaxAbsScaler, the PCA and the agglomerative feature clustering
on the emulation group (27,030 rows), while CARDS_30000 validation and test splits contain
emulation-group rows. This script refits the same three transforms on each seed's *training* rows
only and compares with the published transforms:

  (a) number of principal components retained (criterion of preprocessing/utils.py);
  (b) cluster memberships and their order (the order fixes the feature-to-atom mapping);
  (c) the 12 aggregated features, on all rows and on the validation+test rows.

It also writes, per seed, the refit aggregated features of all 29,601 rows, so the classification
can be repeated with train-only transforms (run_fc_dnn.sh).

    python preprocessing/refit_sensitivity.py [--seeds 1 2 ... 10] [--out results/preprocessing]

Run from the repository root (conda env qrc_class). Nothing outside --out is written.
"""

import argparse
import contextlib
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import importlib.util

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

pre = _load("preprocessing_utils", "preprocessing/utils.py")   # both stages have a utils.py
sys.path.insert(0, "classification")
import run_classification as rc

plt.savefig = lambda *a, **k: None       # pca_correlation_graph would overwrite results/preprocessing/corr_circle.png

CATEG = ["SEX", "MARRIAGE"]


def n_components(X):
    """Number of PCs retained by preprocessing/utils.pca_correlation_graph."""
    ev = PCA(n_components=X.shape[1]).fit(np.asarray(X, dtype=float)).explained_variance_ratio_
    return int(((ev[:-1] - ev[1:]) > 1e-3).sum())


def preprocess(X_fit, X_apply):
    """Fit scaler + PCA clustering on X_fit, aggregate X_apply. Returns (features, clusters, n_pc)."""
    fit_s, app_s = pre.scale_features(X_fit.copy(), X_apply.copy(), CATEG)
    with contextlib.redirect_stdout(io.StringIO()):
        clusters = pre.pca_correlation_graph(fit_s, fit_s.columns)
    _, agg = pre.aggregate_features(clusters, fit_s.copy(), app_s.copy())
    agg.columns = [str(c) for c in agg.columns]
    return agg, {int(k): list(v) for k, v in clusters.items()}, n_components(fit_s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("--out", default="results/preprocessing")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "features"), exist_ok=True)

    raw = pd.read_csv("dataset_aquila/default_of_credit_card_clients.csv", index_col="ID")
    clean = pre.clean_dataset(raw)
    X_emu, X_qpu, _, _ = pre.split_dataset(clean, 2571)
    X_all = clean.loc[:, clean.columns[:-1]]

    # Published transforms (fit on the emulation group), checked against the distributed files
    pub, pub_clusters, pub_npc = preprocess(X_emu, X_all)
    disk = pd.concat([pd.read_csv("dataset_aquila/qpu_samples.csv", index_col="ID"),
                      pd.read_csv("dataset_aquila/emulation_samples.csv", index_col="ID")])
    disk.columns = [str(c) for c in disk.columns]
    reproduced = float(np.abs(pub.loc[disk.index, disk.columns].values - disk.values).max())
    print(f"published transforms re-created: {pub_npc} PCs, {len(pub_clusters)} clusters; "
          f"max |Δ| vs distributed files = {reproduced:.1e}")

    pub_sets = {frozenset(v) for v in pub_clusters.values()}
    rows = []
    for seed in args.seeds:
        a = rc.parse_args(["--features", "dataset_aquila/qpu_samples.csv", "dataset_aquila/emulation_samples.csv",
                           "--labels", "dataset_aquila/labels_qpu.csv", "dataset_aquila/labels_emu.csv",
                           "--seed", str(seed), "--run-name", "refit"])
        X_tr, X_va, X_te, *_ = rc.make_split(a, rc.load_pairs(a.features, a.labels, a.layout)[0])
        refit, clusters, npc = preprocess(X_all.loc[X_tr.index], X_all)
        same_sets = {frozenset(v) for v in clusters.values()} == pub_sets
        same_order = clusters == pub_clusters
        eval_ids = list(X_va.index) + list(X_te.index)
        row = {"seed": seed, "train_rows": len(X_tr), "n_pc": npc, "n_pc_published": pub_npc,
               "n_clusters": len(clusters), "memberships_identical": same_sets, "order_identical": same_order}
        if same_order:
            d_all = np.abs(refit.loc[pub.index, pub.columns].values - pub.values)
            d_eval = np.abs(refit.loc[eval_ids, pub.columns].values - pub.loc[eval_ids].values)
            row.update({"max_abs_delta_all": d_all.max(), "mean_abs_delta_all": d_all.mean(),
                        "max_abs_delta_eval": d_eval.max(), "mean_abs_delta_eval": d_eval.mean(),
                        "features_changed": int((d_all.max(axis=0) > 1e-12).sum())})
            for j, c in enumerate(pub.columns):
                row[f"max_abs_delta_f{c}"] = d_all[:, j].max()
        else:
            row["clusters"] = json.dumps(clusters)
        rows.append(row)
        refit.index.name = "ID"
        refit.to_csv(os.path.join(args.out, "features", f"refit_seed{seed}.csv"))
        print(f"seed {seed}: {npc} PCs, memberships identical {same_sets}, order identical {same_order}"
              + (f", max |Δ| {row['max_abs_delta_all']:.2e} (eval rows {row['max_abs_delta_eval']:.2e}), "
                 f"{row['features_changed']}/12 features changed" if same_order else ""))

    out = os.path.join(args.out, "refit_comparison.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
