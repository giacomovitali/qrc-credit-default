# Reproducing the results

This document maps each figure and table of the article to the commands that produce it. Figure and
table numbers are those of the article and its Supporting Information (SI).

## Before you start

1. Install the environments — [installation.md](installation.md) — and activate `qrc_class` for
   every Python command and sweep script.
2. Download the data deposit and unpack it at the repository root — [data.md](data.md).
3. Run everything from the repository root.

## Reproduction levels

| Level | Starting point | Reproduces | Cost |
| --- | --- | --- | --- |
| **A. Analysis** | Logs in the deposit (`results/<topic>/logs/`) | Every figure and table | Minutes |
| **B. Classification** | Feature files and records in the deposit | The logs, then level A | Hours |
| **C. Emulation** | Raw UCI dataset | The features, then level B — except the hardware arm | Hours to a day |

The Aquila results depend on measurements from the device. They are reproducible at levels A and B,
from the recorded measurements in the deposit, and not at level C.

Every sweep script skips runs whose predictions file already exists. Deleting a topic's `logs/`
directory (or pointing the script elsewhere) re-runs it.

## Level C — features from the raw dataset

```bash
python preprocessing/preprocessing.py                 # 12 aggregated features, Table S1
python preprocessing/fig_pca.py                       # Figs 2-3
python preprocessing/make_splits.py                   # train/val/test IDs of every split (results/splits/)
python preprocessing/make_features24.py               # the 24 preprocessed variables
python preprocessing/invariant_splits.py              # CARDS_30000 split seeds (configs/cards30000_seeds.txt)
python preprocessing/refit_sensitivity.py             # refit on seeds 1-10: components and clusters (Section 2, SI §1.1)

# quantum reservoir: hardware protocol (with 240- and 1,000-shot features) and ideal evolution
julia --project=. -t 27 --heap-size-hint=40G quantum_evolution/run_reservoir.jl configs/reservoir/detuning_hardware.toml output_dir=results/emulation
julia --project=. -t 27 --heap-size-hint=40G quantum_evolution/run_reservoir.jl configs/reservoir/position_hardware.toml output_dir=results/emulation
julia --project=. -t 27 --heap-size-hint=40G quantum_evolution/run_reservoir.jl configs/reservoir/detuning_statevector.toml output_dir=results/emulation
julia --project=. -t 27 --heap-size-hint=40G quantum_evolution/run_reservoir.jl configs/reservoir/position_statevector.toml output_dir=results/emulation

julia --project=. classical_rc/CRC.jl                 # classical reservoir

python preprocessing/make_subsets.py                  # independent subsamples, balance (Table S3)
bash analysis/subsets/run_position.sh                 # position-encoding emulation of the subsamples
python analysis/shots/make_features.py                # features at reduced shot counts (emulation and Aquila records)
```

The four reservoir runs take about 1 h 45 min together on a 30-core AMD EPYC 7763.

## Level B — classification sweeps

| Script | Runs | Time |
| --- | --- | --- |
| `bash analysis/baselines/run_classification.sh` | classifiers, DNN, tree models on 12 and 24 inputs; 3 datasets × 10 seeds | ≈ 2–3 h |
| `bash analysis/benchmark/run_classification.sh` | CRC and QRC (emulation, 240 shots, Aquila) pipelines; 3 datasets × 10 seeds | ≈ 1–2 h |
| `bash analysis/ablation/run_classification.sh` | ablation rungs; 3 datasets × 10 seeds | ≈ 3–4 h |
| `bash analysis/shots/run_classification.sh` | F1 vs shots, detuning CARDS_2500 and position CARDS_1000 | ≈ 30 min |
| `bash analysis/subsets/run_classification.sh` | 20 independent subsamples × 10 seeds | ≈ 1.5 h |
| `python analysis/hardware_noise/encoding_features.py`, then `bash analysis/hardware_noise/run_encoding_check.sh` | encodings with and without shot noise, CARDS_1000 | ≈ 10 min |
| `bash analysis/timing/run_timing.sh` | timing table: one process at a time with all cores (`THREADS=27 P=1`) | ≈ 1–2 h |

The benchmark and ablation analyses use the baseline runs (same splits and seeds), so run the
baselines first. Times are wall-clock on a 30-core AMD EPYC 7763 with the parallelism set in each
script.

## Level A — figures and tables

Run the analyses in this order (later ones read the CSVs of earlier ones):

```bash
python analysis/baselines/analysis.py
python analysis/benchmark/analysis.py
python analysis/benchmark/seed_comparison.py           # CARDS_30000: pre-registered seeds vs seeds 1-10 (SI §1.1)
python analysis/ablation/analysis.py
python analysis/shots/analysis.py
python analysis/subsets/analysis.py
python analysis/hardware_noise/analysis.py
python analysis/hardware_noise/postselection.py        # reads results/hardware_noise/raw/ (export_raw.py)
julia --project=. -t 12 analysis/hardware_noise/noise_model.jl     # ≈ 45 min
python analysis/hardware_noise/noise_model_eval.py
python analysis/hardware_noise/figures.py
python analysis/hardware_noise/fig_feature_maps.py
python analysis/hardware_noise/fig_model_vs_sv.py
python analysis/hardware_noise/fig_pulses.py
python analysis/hardware_noise/tables.py
python analysis/tables/master.py
python analysis/tables/latex_tables.py
python analysis/timing/timing.py
python analysis/timing/table_rows_from_logs.py
```

Each writes into `results/<topic>/` (CSVs, `figures/`, and the printed tests in `analysis_output.txt`
where applicable).

### Main text

| Item | Content | Produced by | Output |
| --- | --- | --- | --- |
| Fig 1, Fig 7 | Pipeline and register schematics | — (drawings) | — |
| Figs 2, 3 | PCA explained variance; feature aggregation (main panel; the zoomed inset is not drawn) | `preprocessing/fig_pca.py` | `results/preprocessing/pca_explained_variance.pdf`, `corr_circle.png` |
| Fig 4 | Pulse sequences of the executed programs | `analysis/hardware_noise/fig_pulses.py` | `results/hardware_noise/figures/fig4_pulses.pdf` |
| Table 1 | Classifiers and hyperparameter grids | `analysis/tables/latex_tables.py` | `results/tables/latex/` |
| Fig 5 | F1 vs number of shots | `analysis/shots/analysis.py` | `results/shots/figures/f1_vs_shots.pdf` |
| Fig 6 | Benchmark, CARDS_30000 | `analysis/benchmark/analysis.py` | `results/benchmark/figures/benchmark_CARDS_30000.pdf` |
| Table 2 | Test performance of every pipeline | `analysis/tables/latex_tables.py` | `results/tables/latex/tab_main_selected.tex` |
| Table 3 | Training times: KNN, SGD, NB, LM, P and DNN rows from 100 repeated runs of one split; random-forest and linear-SVM rows over the 10 splits | `analysis/timing/table_rows_from_logs.py`; `analysis/timing/timing.py` | `results/timing/repeated_runs_rows.csv`; `results/timing/summary.csv`, `table.md` |
| Fig 8 | Benchmark, CARDS_1000 | `analysis/benchmark/analysis.py` | `…/benchmark_CARDS_1000.pdf` |
| Fig 9 | Benchmark, CARDS_2500 | `analysis/benchmark/analysis.py` | `…/benchmark_CARDS_2500.pdf` |
| Fig 10 | Agreement between Aquila and emulation vs time | `analysis/hardware_noise/figures.py` | `results/hardware_noise/figures/noise_agreement_vs_time.pdf` |
| Fig 11 | Feature error vs number of records (bias–variance) | `analysis/hardware_noise/figures.py` | `…/noise_bias_variance.pdf` |
| Fig 12 | Ablation of the reservoir's contribution | `analysis/ablation/analysis.py` | `results/ablation/figures/ablation_ladder.pdf` |
| Statistical tests in the text | Paired Wilcoxon tests, Holm-corrected | `analysis/benchmark/analysis.py`, `analysis/ablation/analysis.py` | `tests.csv`, `analysis_output.txt` |

### Supporting Information

| Item | Content | Produced by | Output |
| --- | --- | --- | --- |
| Table S1 | Feature clusters | `preprocessing/preprocessing.py` | printed |
| Table S2 | Data preparation, splits and seeds | `analysis/tables/latex_tables.py` | `tab_si_data.tex` |
| Table S3 | Balance of the subsets | `preprocessing/make_subsets.py`, `latex_tables.py` | `tab_si_subsets_balance.tex` |
| Table S4 | Resampling methods and class counts | `analysis/tables/latex_tables.py` | `tab_si_resampling.tex` |
| Table S5 | Classifiers, settings and grids | `analysis/tables/latex_tables.py` | `tab_si_classifiers.tex` |
| Fig S1 | Every model on 12 and 24 inputs | `analysis/baselines/analysis.py` | `results/baselines/figures/baselines.pdf` |
| Table S6 | Reservoir, emulation and hardware parameters | `analysis/tables/latex_tables.py` | `tab_si_reservoir.tex` |
| Table S7 | Software versions | `analysis/tables/latex_tables.py` | `tab_si_software.tex` |
| Fig S2 | Post-selection on Aquila | `analysis/hardware_noise/postselection.py`, `figures.py` | `noise_postselection.pdf` |
| Fig S3 | Hardware protocol vs ideal evolution | `analysis/hardware_noise/fig_model_vs_sv.py` | `hardware_protocol_vs_ideal.pdf` |
| Table S8 | Selected test metrics of every pipeline | `analysis/tables/latex_tables.py` | `tab_si_selected.tex` |
| Table S9 | Confusion matrices | `analysis/tables/latex_tables.py` | `tab_si_confusion.tex` |
| Tables S10–S12 | Every pipeline × resampler × classifier | `analysis/tables/latex_tables.py` | `tab_si_full_CARDS_{30000,2500,1000}.tex` |
| Table S13 | Agreement per readout | `analysis/hardware_noise/tables.py` | `results/hardware_noise/tables/tab_si_noise_agreement.tex` |
| Fig S4 | Error budget | `analysis/hardware_noise/figures.py` | `noise_error_budget.pdf` |
| Fig S5 | Hardware bias vs noise models | `noise_model.jl`, `noise_model_eval.py`, `figures.py` | `noise_model.pdf` |
| Figs S6, S7 | Feature maps, all samples and three samples | `analysis/hardware_noise/fig_feature_maps.py` | `features_dataset.pdf`, `features_samples.pdf` |
| Fig S8 | F1 vs shots per classifier | `analysis/shots/analysis.py` | `f1_vs_shots_per_model.pdf` |
| Fig S9 | Feature error vs shots | `analysis/shots/analysis.py` | `feature_error_vs_shots.pdf` |
| Figs S10, S11 | Ablation by readout time, F1 and PR-AUC | `analysis/ablation/analysis.py` | `ablation_timesteps.pdf`, `ablation_timesteps_pr_auc.pdf` |
| Table S14 | Complete ablation | `analysis/tables/latex_tables.py` | `tab_si_ablation.tex` |
| Table S15 | QPU subsets among the subsamples | `analysis/tables/latex_tables.py` | `tab_si_subsets_placement.tex` |
| Figs S12, S13 | Subset robustness: F1 and paired differences | `analysis/subsets/analysis.py` | `subsets_f1.pdf`, `subsets_differences.pdf` |

The LaTeX tables are in `results/tables/latex/`; `tables_preview.tex` compiles all of them into one
document. The CSVs behind every table (`runs_master.csv.gz` and the summaries) are in `results/tables/`.

## Determinism

- **Preprocessing.** Deterministic; re-running it reproduces the distributed files to 1e-16.
- **Statevector and hardware-protocol emulation.** Deterministic, independent of the thread count and
  of how a set is split into ranges. Shots are seeded per datapoint from `(seed, ID)`.
- **Classification.** `--seed` fixes the split, the resampler and model initialisation, including
  the DNN; the models carry fixed internal seeds. Results are bit-identical when the BLAS/OpenMP
  thread counts are the same; the sweep scripts fix them, and
  [pipeline.md](pipeline.md#stage-3--classification) lists the steps that change at the last digit
  with the thread count.
- **Timings** depend on the machine and its load: run the timing sweep alone on an idle machine.
