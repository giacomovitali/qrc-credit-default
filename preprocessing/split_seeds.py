"""Split seeds per dataset (pre-registered).

CARDS_30000 is evaluated on the pre-registered seeds of cards30000_seeds.txt (written by
invariant_splits.py: the first 10 seeds whose train-only fit reproduces the published transforms);
CARDS_2500 and CARDS_1000 on seeds 1-10 (their pool was never seen by any fit). Keys are the dataset
tags of the study analyses ("30000", "2500", "1000").

    from split_seeds import SEEDS, check_seeds      (after sys.path.insert(0, "preprocessing"))
    bash: grep -v '^#' configs/cards30000_seeds.txt
"""
from pathlib import Path

SEED_FILE = Path(__file__).resolve().parents[1] / "configs" / "cards30000_seeds.txt"


def read_seeds(path=SEED_FILE):
    """The pre-registered CARDS_30000 seeds (lines starting with # are comments)."""
    with open(path) as f:
        return [int(t) for line in f if not line.startswith("#") for t in line.split()]


SEEDS = {"30000": read_seeds(), "2500": list(range(1, 11)), "1000": list(range(1, 11))}


def check_seeds(runs, by=("dataset",), where=""):
    """Stop unless every group of `runs` (a DataFrame with a dataset and a seed column) has exactly the
    seeds of its dataset."""
    for key, g in runs.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        ds = dict(zip(by, key))["dataset"]
        got = sorted(set(g.seed))
        if got != SEEDS[ds]:
            raise SystemExit(f"{where}{dict(zip(by, key))}: seeds {got}, expected {SEEDS[ds]}")
