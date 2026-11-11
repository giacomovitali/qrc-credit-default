"""CARDS_30000 on the pre-registered split seeds vs on seeds 1-10 (SI §1.1).

The test sets of seeds 1-10 contain rows of the preprocessing fit pool; the pre-registered seeds
(configs/cards30000_seeds.txt) do not. Per pipeline, the validation-selected test F1 (the benchmark's
select()) is compared between the two seed sets: means and a two-sided Mann-Whitney U test.

Logs: the pre-registered seeds in results/benchmark/logs and results/baselines/logs; seeds 1-10 other
than seed 8 (which is also pre-registered) in results/benchmark/seeds_1-10/logs and
results/baselines/seeds_1-10/logs, with the same naming.

    python analysis/benchmark/seed_comparison.py

Writes results/benchmark/seed_comparison.csv. Run from the repository root (conda env qrc_class).
"""
import importlib.util
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("benchmark_analysis", ROOT / "analysis" / "benchmark" / "analysis.py")
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)

ALT = "seeds_1-10"
PIPES = ["dethw", "detshot", "CRC", "FC", "DNN", "trees24"]


def load(benchmark_dir, baseline_logs):
    """The benchmark's load_runs() on <benchmark_dir>/logs and baseline_logs, CARDS_30000 only."""
    saved = bm.OUT, bm.BASELINE_LOGS
    bm.OUT, bm.BASELINE_LOGS = benchmark_dir, baseline_logs
    try:
        runs = bm.load_runs()
    finally:
        bm.OUT, bm.BASELINE_LOGS = saved
    return runs[runs.dataset == "30000"]


def main():
    current = load(bm.OUT, bm.BASELINE_LOGS)
    alt = load(bm.OUT / ALT, ROOT / "results" / "baselines" / ALT / "logs")
    sel = bm.select(pd.concat([current, alt], ignore_index=True))
    pre = sel[sel.seed.isin(bm.SEEDS["30000"])]
    one_ten = sel[sel.seed.isin(range(1, 11))]
    rows = []
    for p in PIPES:
        a, b = pre[pre.pipeline == p].f1, one_ten[one_ten.pipeline == p].f1
        rows.append(dict(pipeline=p, label=bm.PIPELINES[p][0], n_pre=len(a), f1_pre=a.mean(), n_1_10=len(b),
                         f1_1_10=b.mean(), diff=a.mean() - b.mean(), p_mannwhitney=mannwhitneyu(a, b).pvalue))
    df = pd.DataFrame(rows)
    df.to_csv(bm.OUT / "seed_comparison.csv", index=False)
    print(df.round(4).to_string(index=False))
    print(f"max |difference| {df['diff'].abs().max():.4f}, min p {df.p_mannwhitney.min():.2f}")


if __name__ == "__main__":
    main()
