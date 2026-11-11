"""Train/validation/test IDs of every split used in the article.

The split is drawn by classification/run_classification.py (`load_pairs` and `make_split`, called here
unchanged): stratified 70/15/15 over the concatenated pairs, random_state = seed. It depends only on
the datapoint IDs, their order and the labels, so the 12 aggregated features stand in for any input.
CARDS_30000 takes the QPU group first, as every run of the article.

Seeds: preprocessing/split_seeds.py (CARDS_30000: the pre-registered seeds; CARDS_2500/1000: 1-10),
plus seeds 1-10 for CARDS_30000 (the comparison of SI §1.1).

    python preprocessing/make_splits.py [--out results/splits]

Writes <out>/splits_CARDS_<n>.csv.gz with columns seed, ID, split (train / val / test).
Run from the repository root (conda env qrc_class).
"""
import argparse
import os
import sys
from argparse import Namespace

import pandas as pd

sys.path.insert(0, "classification")
import run_classification as rc  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_seeds import SEEDS  # noqa: E402

D = "dataset_aquila"
DATASETS = {
    "30000": ([f"{D}/qpu_samples.csv", f"{D}/emulation_samples.csv"], [f"{D}/labels_qpu.csv", f"{D}/labels_emu.csv"],
              sorted(set(SEEDS["30000"]) | set(range(1, 11)))),
    "2500": ([f"{D}/qpu_samples.csv"], [f"{D}/labels_qpu.csv"], SEEDS["2500"]),
    "1000": ([f"{D}/qpu_samples_pos.csv"], [f"{D}/labels_qpu_pos.csv"], SEEDS["1000"]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="results/splits")
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)
    for ds, (features, labels, seeds) in DATASETS.items():
        pairs, _ = rc.load_pairs(features, labels, "auto")
        rows = []
        for seed in seeds:
            parts = rc.make_split(Namespace(split="random", split_seed=seed), pairs)
            for name, X in zip(("train", "val", "test"), parts[:3]):
                rows.append(pd.DataFrame({"seed": seed, "ID": X.index.astype(int), "split": name}))
        df = pd.concat(rows, ignore_index=True)
        path = os.path.join(out, f"splits_CARDS_{ds}.csv.gz")
        df.to_csv(path, index=False)
        sizes = df[df.seed == seeds[0]].split.value_counts().to_dict()
        print(f"CARDS_{ds}: {len(seeds)} seeds, sizes {sizes} -> {path}")


if __name__ == "__main__":
    main()
