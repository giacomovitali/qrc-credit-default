"""Figures of the preprocessing stage: PCA explained variance (Figure 2) and the correlation circle of
the feature aggregation (Figure 3, main panel; the zoomed inset of the article is not drawn).

Recomputes the scaled emulation group exactly as preprocessing.py does, fits the PCA on it, and selects
the number of components with the same rule (drops in the explained-variance ratio larger than 1e-3).

    python preprocessing/fig_pca.py

Writes results/preprocessing/pca_explained_variance.pdf and results/preprocessing/corr_circle.png.
Run from the repository root (conda env qrc_class).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils import clean_dataset, split_dataset, scale_features, pca_correlation_graph

OUT = "results/preprocessing"


def main():
    input_df = pd.read_csv("dataset_aquila/default_of_credit_card_clients.csv", index_col="ID")
    cleaned_df = clean_dataset(input_df)
    X_emulation, X_qpu, _, _ = split_dataset(cleaned_df, 2571)
    X_emu_scaled, _ = scale_features(X_emulation, X_qpu, ["SEX", "MARRIAGE"])

    X = np.array(X_emu_scaled).astype(float)
    ratio = PCA(n_components=X.shape[1]).fit(X).explained_variance_ratio_
    n_pc = int(((ratio[:-1] - ratio[1:]) > 1e-3).sum())
    cumulative = np.cumsum(ratio)
    pcs = np.arange(1, len(ratio) + 1)

    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pcs, ratio, "o-", label="Explained Variance")
    ax.plot(pcs, cumulative, "s-", label="Cumulative Variance")
    ax.axhline(cumulative[n_pc - 1], color="black", linestyle="--", label=f"Threshold at PC {n_pc}")
    ax.set_xticks(pcs)
    ax.set_xlabel("Principal Components")
    ax.set_ylabel("Variance")
    ax.grid(True)
    ax.legend(loc="center right")
    fig.savefig(os.path.join(OUT, "pca_explained_variance.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"{n_pc} components, cumulative explained variance {cumulative[n_pc - 1]:.4f}")

    pca_correlation_graph(X_emu_scaled, X_emu_scaled.columns)     # writes corr_circle.png
    print(f"wrote {OUT}/pca_explained_variance.pdf, {OUT}/corr_circle.png")


if __name__ == "__main__":
    main()
