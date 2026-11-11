# The pipeline

Four stages, each communicating with the next only through files on disk. Every command is run
**from the repository root**.

```
raw UCI CSV
    │  preprocessing/preprocessing.py                     (qrc_class)
    ▼
aggregated features + labels, split into an emulation group and a QPU group
    │  quantum_evolution/run_reservoir.jl                 (Julia)
    │  classical_rc/CRC.jl                                (Julia, classical reservoir)
    │  quantum_evolution/reduce_experiment.jl             (Julia, recorded hardware runs)
    ▼
reservoir readouts: one feature vector per datapoint
    │  classification/run_classification.py               (qrc_class)
    ▼
run logs with metrics per classifier × resampling method, a manifest and test predictions per run
    │  result_analysis/runs.py, analysis/                 (qrc_class)
    ▼
figures and tables
```

Every driver writes a JSON **manifest** beside its output: resolved settings, arguments, git commit
and whether the tree was dirty, software versions and input checksums. See
[architecture.md](architecture.md#randomness-and-provenance). Which command produces which figure and
table of the article is in [reproducing.md](reproducing.md).

---

## Stage 1 — Preprocessing

```bash
conda activate qrc_class
python preprocessing/preprocessing.py
```

Cleans the raw dataset, scales and one-hot-encodes it, aggregates correlated columns into a smaller
set of features, and splits the result into two disjoint groups.

**Settings** (in `main()`): the input `filename`, `categ_cols` (columns to one-hot encode) and
`num_samples_qpu`, the size of the group reserved for quantum-hardware processing (2571).

**What it does.** Rows with undocumented `MARRIAGE` or `EDUCATION` codes are dropped and the `PAY_*`
columns are shifted by +1. Numerical columns are scaled to [-1, 1] with a `MaxAbsScaler` fitted on
the emulation group only; `SEX` and `MARRIAGE` are one-hot encoded with one redundant level removed.
Features are then clustered by their loadings on the principal components — the correlation-circle
method — and each cluster is replaced by its mean, giving **12 aggregated features**. The
emulation/QPU split is stratified on the label, so both groups keep the class balance.

**Outputs** (in `dataset_aquila/`): `emulation_samples.csv`, `qpu_samples.csv`, `labels_emu.csv`,
`labels_qpu.csv`, and the correlation circle as `results/preprocessing/corr_circle.png`. Feature
files carry an `ID` column that keys every downstream artefact.

Further scripts in `preprocessing/`:

| Script | Writes |
| --- | --- |
| `make_features24.py` | The 24 preprocessed variables before aggregation (same fitted scaler), for the emulation group, CARDS_2500 and CARDS_1000 |
| `make_subsets.py` | Re-derives CARDS_2500 and CARDS_1000 from the raw file, draws the independent stratified subsamples of the same sizes, and checks the balance of all of them |
| `invariant_splits.py` | Re-derives `configs/cards30000_seeds.txt`, the CARDS_30000 split seeds ([data.md](data.md#datasets-splits-and-fitted-transforms)) |
| `refit_sensitivity.py` | Refits the preprocessing on the training rows of given split seeds and compares the components and clusters with the published fit |
| `make_splits.py` | The train/validation/test IDs of every dataset and seed (`results/splits/`) |
| `fig_pca.py` | The PCA explained-variance figure and the correlation circle |
| `split_seeds.py` | Not a script: the split seeds per dataset (`SEEDS`), read by the analyses |

---

## Stage 2 — Reservoir

### Quantum reservoir

```bash
julia --project=. -t auto quantum_evolution/run_reservoir.jl CONFIG.toml [key=value ...]
```

Settings come from the TOML config, then from `key=value` overrides on the command line; nested
settings use dots (`layer.delta=2.5`), lists may be comma-separated (`sets=qpu_samples,emulation_samples`).
An unknown key is an error, and `dry_run=true` prints the resolved settings without running.
The configurations of the article are in `configs/reservoir/` (what each backend evolves:
[architecture.md](architecture.md#what-each-backend-evolves)):

| Config | Encoding | Backend | Samples |
| --- | --- | --- | --- |
| `detuning_hardware.toml` | detuning | hardware protocol + shots | CARDS_30000 (both groups) |
| `position_hardware.toml` | position | hardware protocol + shots | CARDS_1000 |
| `detuning_statevector.toml` | detuning | statevector, exact readout times | CARDS_30000 |
| `position_statevector.toml` | position | statevector, exact readout times | CARDS_1000 |

The article's noiseless emulation ("statevector emulation") is the hardware-protocol backend
(`*_hardware.toml`, files `…_hw.csv`), and its shot-based emulation is the shots sampled by the same
backend (`…_hwshots_<s>.csv`). The `*_statevector.toml` outputs (ideal continuous evolution, `…_sv_exact.csv`)
are used only to compare the protocol with the continuous evolution.

#### Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `encoding` | `"position"` | `"position"` or `"detuning"` |
| `backend` | `"statevector"` | `"statevector"`, `"hardware"` (Aquila protocol + sampled shots) or `"shots"` (record-based shot emulation, below) |
| `readout_timing` | `"exact"` | Statevector: `"exact"` readout times, or `"integrator"` (readouts at the integrator's step ends) |
| `shot_counts` | `[240, 1000]` | Hardware: shot counts to write features for (each ≤ `n_shots`) |
| `save_bitstrings` | `true` | Hardware: also store the `n_shots` sampled bitstrings per readout |
| `input_dir` | `"dataset_aquila"` | Directory with the preprocessed CSVs |
| `sets` | `["qpu_samples"]` | Input files to process, without `.csv` |
| `scale_reference` | first of `sets` | Set whose range fixes the feature scaling |
| `first`, `last` | `1`, `-1` | Datapoint range within each set, 1-based and inclusive; `-1` = to the end |
| `n_shots` | `1000` | Hardware, shots: samples per datapoint and readout time |
| `seed` | `1234` | Hardware, shots: per-datapoint seeds derive from it |
| `output_dir` | `"datasets"` | Statevector, hardware: where outputs are written |
| `experiment_root` | `"data/experiments"` | Shots: where experiment directories are created |
| `resume` | `""` | Shots: an experiment directory to append to |
| `checkpoint_every` | `100` | Shots: save the experiment every N datapoints |
| `tag` | `""` | Appended to output and experiment names |
| `dry_run` | `false` | Print the resolved settings and exit |
| `layer.t_start`, `layer.t_end`, `layer.step` | `0.0`, `2.5`, `0.5` | Evolution window and readout spacing, µs |
| `layer.omega`, `layer.scale` | `2π`, `10.0` | Rabi frequency (rad/µs), chain spacing (µm) |
| `layer.delta` | `2.0` | Position: uniform detuning. Detuning: offset Δ |
| `layer.lambda` | `1.0` | Position only: displacement scale λ |
| `layer.delta_max` | `-9.0` | Detuning only: detuning window Δ_max |

**Scaling.** Each set is shifted by its own global minimum and divided by the largest magnitude of
the shifted reference set. For the preprocessed data both groups span the same range, so the choice
of reference does not change the result.

#### Hardware-protocol backend

Emulates, noiselessly and on a single chain, the programs executed on Aquila: one fresh program from
the ground state per readout time, with the same waveforms. For each datapoint it writes, under
`output_dir`:

| File | Contents |
| --- | --- |
| `…_hw.csv` | Exact (infinite-shot) features |
| `…_hwshots_<s>.csv` | Features from the first `s` sampled shots, for each `s` in `shot_counts` |
| `…_hwbits.jld2` | The `n_shots` sampled bitstrings per datapoint and readout (`save_bitstrings`) |
| `…_hw.manifest.json` | Settings, git commit, input checksums, output paths |

Shots are sampled from the exact states with an RNG seeded from `(seed, datapoint ID)`, so results
do not depend on threads, ranges or order. A datapoint run as part of a different input set gets the
same exact features to the integrator tolerance, not always to the last bit (312 CARDS_1000
datapoints re-run in a set of 8,633: 298 bit-identical, the rest within 1.7e-6). Features at other
shot counts come from the stored bitstrings, without re-running:

```bash
julia --project=. -t auto quantum_evolution/reduce_bitstrings.jl \
    input=results/emulation/<stem>_hwbits.jld2 shots=10,20,40,100 output=results/emulation/<stem>_hwshots_{shots}.csv
```

The counts of one sweep are nested subsamples (the first `s` of the same samples), which keeps a
shot-count curve free of resampling jitter.

#### Statevector backend

Writes `<output_dir>/<encoding>_encoding_<set>_<n>_delta_<Δ>..._sv_exact.csv` (or `…_sv.csv` with
`readout_timing = "integrator"`) — one column per datapoint, named by ID — plus a `.manifest.json`
beside it. Datapoints are distributed over Julia threads; results do not depend on the thread count.

#### Shot-based backend with experiment records

`backend = "shots"` records an *experiment* in the same layout as a hardware run
([architecture.md](architecture.md#experiments-persistence-for-shot-based-runs)), saving it every
`checkpoint_every` datapoints. It evolves a different pulse sequence from the hardware protocol and
is not used for the results of the article; the hardware-protocol backend is. Each datapoint's
shots are seeded from `(seed, ID)`, so splitting a set into ranges, interrupting and resuming
(`resume=DIR`) do not change any record.

#### Reducing recorded experiments to features

The Aquila runs are stored as experiments; `reduce_experiment.jl` turns one or more of them into a
feature file:

```bash
julia --project=. quantum_evolution/reduce_experiment.jl \
    experiments=DIR1,DIR2 shots=100,240 seed=7 output=results/<name>_{shots}.csv
```

For each shot count the stored records are subsampled (seeded per datapoint) and reduced to ⟨Zᵢ⟩
and ⟨ZᵢZⱼ⟩ per readout time; `shots=-1` uses every stored record. Several experiments are merged,
and a datapoint appearing in two of them is an error unless `duplicates=first`. The output has one
column per datapoint in ascending ID order.

#### Long runs

Throughput on a 30-core AMD EPYC 7763 at 27 threads: about 10–14 datapoints/s for the detuning
encoding and 7–8/s for the position encoding, plus about 80 s of start-up. For runs over a few
thousand datapoints, cap Julia's heap: `julia --project=. -t 27 --heap-size-hint=40G …`. Without it,
the garbage collector falls behind on long multithreaded runs: memory grows to 70–90 GB and fewer
than half of the threads do useful work. The driver warns when a large run starts without the cap
(measured on 12,000 detuning samples: peak 37 GB instead of 74–86 GB, identical outputs). Even
capped, one process keeps only about half of 27 threads busy (stop-the-world garbage collection), so
for large sets run 2–3 processes on disjoint `first`/`last` ranges with fewer threads each; outputs
do not depend on the split. Progress lines are flushed as they happen.

Jobs that take more than a few minutes are best started detached (`tmux`, `screen` or `nohup`), so
that they survive a closed terminal or SSH session.

### Classical reservoir

```bash
julia --project=. classical_rc/CRC.jl
```

Replaces the quantum evolution with a mean-field approximation on the same atom geometry and writes
`*_crc.csv` feature files that Stage 3 consumes like the quantum ones. Its settings are the constants
at the top of the script (Table S6 of the article lists them). The layer is adapted from QuEra's
[QRC-tutorials](https://github.com/QuEraComputing/QRC-tutorials) (`Classical Spin Reservoir Demo.ipynb`);
see [classical_rc/README.md](../classical_rc/README.md).

---

## Stage 3 — Classification

```bash
conda activate qrc_class
python classification/run_classification.py --features F [F ...] --labels L [L ...] --run-name NAME [options]
```

Features and labels are given as pairs. Each feature file is matched to its label file **by
datapoint ID**, whichever layout it has: one column per datapoint (reservoir output), or one row per
datapoint with an `ID` column (preprocessed data). Files whose columns carry no IDs can only be
matched by position; that is reported, and recorded in the manifest.

For each resampling strategy (none, SMOTE, k-means SMOTE, cluster centroids), each selected
classifier is trained with its own grid search, selected on validation F1, and evaluated on the test
set.

| Option | Default | Meaning |
| --- | --- | --- |
| `--split random` | ✓ | Stratified 70/15/15 split over all pairs together |
| `--split qpu-mix` | | Two pairs, emulation first and QPU second. The test set has `--test-size` samples (4441), a `--qpu-perc` fraction of the QPU group included; `--val-size` (4440) for validation |
| `--train-features F …` | | Take training-set features from these files instead (same datapoints) |
| `--eval-features F …` | | Take validation- and test-set features from these files instead |
| `--classifiers` | `KNN SGD NBT LM P LSVM DT RF` | Any of `KNN SGD NB NBT LM P LSVM SVM DT RF DTR RFR ET HGB DNN` |
| `--resampling` | all four | Any of `None SMOTE kSMOTE CC` |
| `--seed` | `7` | Seeds the split, the resampler, and model initialisation (numpy, PyTorch) |
| `--split-seed`, `--resample-seed` | `--seed` | Override the seed of one of those individually |
| `--run-name` | required | Names the log, the manifest and the figures |
| `--log-dir`, `--figure-dir` | `results/logs`, `results/logs/fig` | Output directories |
| `--n-jobs` | `-1` (all cores) | Cores for the tree ensembles; results do not depend on it |
| `--no-predictions` | | Don't write the per-datapoint test predictions |

The article's six classifiers are `KNN SGD NBT LM P LSVM`.

`LSVM` is a linear SVM fitted with liblinear (`LinearSVC`, squared hinge loss solved in the primal),
with the C grid and class weights of `SVM`: it converges at every C and searches the CARDS_30000 grid
in minutes. `SVM` is libsvm's `SVC(kernel="linear")` (hinge loss), which costs hours per fit on
CARDS_30000.

`NBT` is Gaussian naive Bayes fitted on the training set only, like every other model. `NB` fits it
on **train + validation** (it has no hyperparameter to select).

`DTR`, `RFR`, `ET` and `HGB` are the reference baselines (decision tree, random forest, extra trees,
histogram gradient boosting), all with `class_weight="balanced"` and grids of 15–18 points that
include depth and leaf-size limits; `HGB` runs without internal early stopping, so it is selected on
the same validation set as every other model. `DT` and `RF` are smaller-grid variants. Their
`n_params` is the total number of tree nodes. `DNN` is the fully connected network of
`classification/dnn.py`.

`--train-features` / `--eval-features` give the heterogeneous setting — for example training on
emulated features and testing on hardware features of the same datapoints.

**CARDS_30000 pair order.** With `--split random` the split of the concatenated pairs depends on
their order. All CARDS_30000 runs of the article put the **QPU group first**:
`--features <qpu file> <emulation file> --labels dataset_aquila/labels_qpu.csv dataset_aquila/labels_emu.csv`.

**Thread count.** Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` explicitly and
keep them fixed within a comparison. Two steps vary at the last floating-point digit with the BLAS/OpenMP
thread count and, for ClusterCentroids, with concurrent load: the logistic-regression solver (`LM`), which
can then pick a different `C` when two grid points are nearly tied on validation, and the k-means
inside ClusterCentroids (`CC`), which changes the resampled training set and with it every model under
`CC`. Measured on 50 runs repeated at 1 vs 4 threads: about 1 % of the (model, resampler) rows changed,
by up to 0.09 F1 in single rows; the validation-selected pipeline F1 averaged over 10 seeds did not change
at the third decimal. The other models are bit-identical across thread counts. Two more cases show up
between 4 and 27 threads (tree models on the 24 variables, CARDS_30000): SMOTE's neighbour search picks a
different neighbour for a few synthetic samples (4 of 11,474, seed 8), so models under `SMOTE` can change;
and `HGB` bins features by quantiles, so it also follows the last-digit changes of the `CC` set that the
other tree models ignore. Models under `None` and `kSMOTE` were identical. The sweep scripts in
`analysis/` fix the thread counts they were run with.

**Timing.** The training times of the article's timing table are `grid_time_s / grid_size`: the mean
time per hyperparameter configuration (fit on the resampled training set + validation prediction),
measured with multithreaded libraries. To time a model on the same footing, run one process at a time
with all cores (`OMP_NUM_THREADS` etc. and `--n-jobs` set to the core count): on one core per process,
KNN is 10–16×, LM 2–4×, the random forest 10–21× and extra trees 3–12× slower than with 27 threads,
while P, SGD, the DNN and the single decision tree barely change. `analysis/timing/run_timing.sh` does
this.

**Outputs.** `<log-dir>/<run-name>.log`, `<log-dir>/<run-name>.manifest.json`,
`<log-dir>/<run-name>.predictions.csv.gz`, and a confusion matrix per model and resampling method in
`<figure-dir>`.

For every model and resampling method the log has, in this order:

- `Best parameters:` and `Best validation F1-score:` (grid-searched models);
- a metrics line — accuracy, precision, recall, `AUC`, F1 (class 1). **`AUC` there is the area under
  the precision–recall curve of the hard 0/1 predictions**, not a score-based AUC;
- `Extended metrics: {json}`: precision, recall, F1, balanced accuracy, `roc_auc` and `pr_auc`
  (`average_precision_score`) computed from continuous scores, `score_source` (`predict_proba` for
  KNN, NB, LM, DT, RF; `decision_function` for SGD, P, SVM, LSVM; `softmax` for the DNN), the
  confusion matrix (`tn fp fn tp`), the selected `best_params` and `val_f1`, `grid_size`, `fit_data`
  (`train`, or `train+val` for `NB`), `fit_time_s` (one fit of the selected configuration),
  `grid_time_s` (the whole search), `predict_time_s` (test set), `n_train` (after resampling),
  `n_test` and `n_params` (weights + biases for linear models and the DNN, means + variances +
  priors for NB, tree nodes for the tree models, none for KNN).

Each resampling method also logs `Resampler used: {json}` with the parameters imblearn actually
resolved (SMOTE `k_neighbors = 5`; KMeansSMOTE `k_neighbors = 2`, `MiniBatchKMeans` with 8 clusters,
`cluster_balance_threshold = 1e-5`; ClusterCentroids `KMeans`, `voting = soft`), its
`random_state`, run time and class counts before and after; the same goes into the manifest under
`resamplers`. The seeds are logged at the top (`Run seeds:`).

The predictions file (read it with `runs.load_predictions`, which keeps the resampler name `None`
from becoming NaN) has one row per test datapoint, model and resampling method: `resampler, model,
ID, y_true, y_pred, score`. Test-set bootstraps and ROC/PR curves are computed from it without
re-running.

---

## Stage 4 — Analysis

```python
import sys; sys.path.insert(0, "result_analysis")
from runs import load_runs
df = load_runs("results/benchmark/logs", "*")     # one row per run × resampling × model
```

`result_analysis/runs.py` parses every run log into a table and, for runs that have a manifest, adds
what the run was — split, seeds, input files, git commit — as columns, together with the extended
metrics (`runs.EXTENDED`). `runs.best_params(value)` turns the `best_params` column back into a
dict. `python result_analysis/runs.py <log dir>` prints a summary.

The analyses of the article are in `analysis/`, one directory per topic. Each has a sweep script
(`run_classification.sh`, which skips runs whose predictions already exist) and an `analysis.py` that
reads the logs, selects a pipeline per seed on validation F1, runs the tests and draws the figures:

| Directory | Topic |
| --- | --- |
| `analysis/baselines/` | The classifiers, the DNN and the tree models on the 12 aggregated features and on the 24 preprocessed variables |
| `analysis/benchmark/` | Every pipeline over 10 seeds, paired tests with Holm correction, the benchmark figures |
| `analysis/ablation/` | Feature-map ablation of the reservoir's contribution (`feature_maps.py` applies a fixed transform to the inputs of `run_classification.py`) |
| `analysis/hardware_noise/` | Agreement between Aquila and emulation, bias–variance and error budget, post-selection, noise-model scan, feature maps, pulse figure |
| `analysis/shots/` | F1 and feature error as a function of the number of shots, for emulation and Aquila |
| `analysis/subsets/` | Robustness to the choice of the QPU subsets (independent stratified subsamples) |
| `analysis/tables/` | Master tables of every run (`master.py`) and the LaTeX tables (`latex_tables.py`) |
| `analysis/timing/` | The timing table |

**Split seeds.** CARDS_2500 and CARDS_1000 run on seeds 1–10. CARDS_30000 runs on the
pre-registered seeds of `configs/cards30000_seeds.txt` ([data.md](data.md#datasets-splits-and-fitted-transforms));
Python code reads them with `preprocessing/split_seeds.py` (`SEEDS[dataset]`). The analyses call
`check_seeds()`, so they stop if a pipeline's logs cover any other seed set. The sweep scripts take
`S30000=...` to override the CARDS_30000 list (e.g. for a pilot). The ablation's fitted feature maps
take `--fit emulation` (the emulation group, used for CARDS_2500/1000, whose rows it never contains)
or `--fit train` (the run's own training rows, used on CARDS_30000).

**Selection.** For each dataset, pipeline and seed, the (model, resampling method) with the highest
validation F1 is selected and its test metrics are reported; ties go to the first in resampling
order, then in model order.

**Master tables.** `analysis/tables/master.py` gathers every classification run into one table
(`runs_master.csv.gz`), one row per run × model × resampling method, with explicit descriptors
(study, dataset, subset, input, encoding, backend, shot count, readouts, observables, number of
features) and the whole extended-metrics line of its log, confusion matrix included. It summarises
them per configuration (`summary_configurations.csv`: mean, SD and 95 % CI over seeds, the selected
hyperparameters' most frequent value and full distribution) and per pipeline after validation
selection (`selected_per_seed.csv`, `summary_selected.csv`). A configuration with fewer seeds than
its pipeline is one where no grid point reached a validation F1 above 0 in the missing seeds, so the
driver logged no test result (`missing_seeds`; `n_seeds = 0` when that happened in every seed).
`regression.csv` records the checks against each study's own tables. `latex_tables.py` writes the
booktabs tables and checks every parameter it prints (grids, resampler settings, reservoir settings,
package versions) against the logs, manifests and configuration files.
