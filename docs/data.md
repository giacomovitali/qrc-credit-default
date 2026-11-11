# Data

## Where the data comes from

- **Raw dataset.** The UCI *default of credit card clients* dataset (I.-C. Yeh, UCI Machine Learning
  Repository, 2009, [doi:10.24432/C55S3H](https://doi.org/10.24432/C55S3H), CC BY 4.0).
- **Everything derived from it** — preprocessed features, split indices, reservoir features,
  measurement records of Aquila and of the emulation, classification logs — is in the data deposit
  on Zenodo, [doi:10.5281/zenodo.24131386](https://doi.org/10.5281/zenodo.24131386) (CC BY 4.0).
  Its archives unpack **at the repository root** into the directories the code reads
  (`dataset_aquila/`, `datasets/`, `results/…`); the deposit's README lists them.

None of these directories is under version control.

## Input dataset

30,000 credit-card holders in Taiwan, 23 explanatory attributes, and a binary label
(`default_payment_next_month`) indicating default in the following month. The classes are
imbalanced, roughly 22 % positive, which is why the classification stage evaluates several
resampling strategies.

- Source: <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>
- Expected location: `dataset_aquila/default_of_credit_card_clients.csv`
- Expected header: `ID, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_1…PAY_6,
  BILL_AMT1…BILL_AMT6, PAY_AMT1…PAY_AMT6, default_payment_next_month`

> The repayment-status columns are named `PAY_1`…`PAY_6` here. The file distributed by UCI names
> the first of them `PAY_0`; rename it before running the preprocessing stage.

## Preprocessed data

| File | Contents |
| --- | --- |
| `dataset_aquila/emulation_samples.csv` | Aggregated features of the emulation group (27,030 rows) |
| `dataset_aquila/qpu_samples.csv` | Aggregated features of the QPU group, CARDS_2500 (2,571 rows) |
| `dataset_aquila/qpu_samples_pos.csv` | Aggregated features of CARDS_1000 (1,000 rows) |
| `dataset_aquila/labels_emu.csv`, `labels_qpu.csv`, `labels_qpu_pos.csv` | Labels, keyed by `ID` |
| `results/baselines/features/{emulation,qpu,qpu_pos}_samples_24.csv` | The 24 preprocessed variables before aggregation |

Twelve aggregated features per row, each the mean of a cluster of correlated original columns. The
`ID` column is the join key for everything downstream. The 24 variables are the 21 scaled numerical
columns plus the one-hot `SEX_1`, `MARRIAGE_1`, `MARRIAGE_2`, written by
`preprocessing/make_features24.py` with the same fitted scaler; aggregating them with the published
clusters gives the 12 features to 1e-16. `preprocessing/make_subsets.py` re-derives the IDs of
CARDS_2500 and CARDS_1000 from the raw file (IDs and order match the files above).

## Datasets, splits and fitted transforms

| Name in the article | Rows | Definition |
| --- | --- | --- |
| CARDS_30000 | 29,601 | All rows after cleaning (both groups) |
| CARDS_2500 | 2,571 | The QPU group: `train_test_split(cleaned, test_size=2571, random_state=7, stratify=label)`, test side |
| CARDS_1000 | 1,000 | `train_test_split(CARDS_2500 IDs, test_size=1000, random_state=7, stratify=label)`, test side |

Every fitted transform of the preprocessing stage — the `MaxAbsScaler`, the PCA and the
agglomerative clustering — is fitted on the **emulation group only** (27,030 rows) and then applied
to both groups. Re-running the preprocessing reproduces the distributed files to 1e-16. The QPU
group, and hence CARDS_2500 and CARDS_1000, is never seen by any fit. CARDS_30000 evaluation splits
do contain emulation-group rows, so CARDS_30000 is evaluated only on split seeds for which that does
not matter (next paragraph).

**CARDS_30000 split seeds.** For most seeds, refitting the transforms on the 20,720 training rows
alone gives different features: the clustering of the correlated `PAY_*` and `BILL_AMT*` columns
changes in about 60 % of seeds, and the scaler is set by a handful of customers (8 customers alone
hold 16 of the 21 column maxima, one of them 7). `preprocessing/invariant_splits.py` keeps the seeds
for which the published code, run on the training rows only, reproduces every fitted quantity
exactly:
- the 12 aggregated features of all 29,601 rows, bit for bit, cluster order included;
- the 24-variable scaler;
- the input scaling of `run_reservoir.jl` (global minimum and range);
- the input scaling of `CRC.jl` (max |x|).

With these seeds, no reported test row was seen by any fit, and the distributed feature files
(emulated, shot and CRC) are exactly what a train-only pipeline would have produced. The rule was
fixed before any run: the first 10 passing seeds in ascending order (64 of seeds 1–371 keep the
cluster order; 10 pass every check). They are listed in `configs/cards30000_seeds.txt`: 8, 95, 170,
197, 242, 254, 265, 277, 287, 371. CARDS_2500 and CARDS_1000 use seeds 1–10.

The rule looks only at the training rows' features. It does constrain the split slightly: the few
customers that alone hold a column maximum always fall in the training set. The fitted maps of the
ablation (`pca`, `rff`, `tanh`) cannot match the emulation-group fit, so on CARDS_30000 they are
fitted on each run's training rows (`feature_maps.py --fit train`).

**Splits.** Train/validation/test splits are drawn per dataset and per seed by the classification
stage: stratified 70/15/15 (two successive `train_test_split` calls with `random_state = seed`),
giving 20,720 / 4,440 / 4,441, 1,799 / 386 / 386 and 700 / 150 / 150 rows. For CARDS_30000 the split
depends on the order in which the two groups are concatenated; every run puts the **QPU group first**
([pipeline.md](pipeline.md#stage-3--classification)). The CARDS_1000 split is drawn independently of
the CARDS_2500 split, so it is **not** nested in it: at a given seed most CARDS_1000 test rows belong
to the CARDS_2500 training set. The IDs of every split are in the deposit (`results/splits/`).

**Independent subsamples.** `preprocessing/make_subsets.py` draws further subsets of 2,571 and 1,000
samples with the same call and other seeds (`random_state` 101–110 and 201–210), writing per-subset
labels and verbatim column/row selections of the feature files. Different sizes need different
seeds: at equal `random_state`, sklearn's stratified split puts the smaller test side inside the
larger one.

## Reservoir features

One file per encoding, input group and backend, written with **one column per datapoint** (column
name = datapoint ID) and one row per feature — the transpose of the usual layout. The Python stages
transpose them on load and align them to the labels by ID. Feature-vector lengths are 390 for the
detuning encoding and 455 for the position encoding; the row order is in
[architecture.md](architecture.md#readouts-and-the-feature-vector).

| File (in `results/emulation/` unless stated) | Contents |
| --- | --- |
| `<enc>_encoding_<set>_12_delta_<Δ>…_hw.csv` | Statevector emulation of the hardware protocol (noiseless) |
| `…_hwshots_<s>.csv` | Shot-based emulation: the first `s` shots sampled from the same states |
| `…_hwbits.jld2` | The 1,000 sampled bitstrings per datapoint and readout |
| `…_sv_exact.csv` | Ideal continuous evolution, exact readout times |
| `dataset_aquila/{qpu,emulation}_samples_crc_labeled.csv` | Classical (mean-field) reservoir |
| `datasets/detuning_encoding_qpu_samples_240shots_real_aquila.csv` | Aquila, detuning encoding, CARDS_2500 |
| `datasets/position_encoding_qpu_samples_240shots_real_aquila_lambda1.0_delta2.0.csv` | Aquila, position encoding, CARDS_1000 |

`<enc>` is `detuning` or `position`, `<set>` the input file (`emulation_samples`, `qpu_samples`,
`qpu_samples_pos`). The emulation covers the detuning encoding on CARDS_30000 (both groups) and the
position encoding on CARDS_1000; the position encoding of the size-1,000 subsamples is in
`results/subsets/emulation/`. Each emulated file has a `.manifest.json` beside it.

## Hardware runs

Encoding read from the stored bitstrings (12 bits: detuning, 13: position), group from the datapoint
IDs:

| Runs | Encoding | Datapoints | Records per readout |
| --- | --- | --- | --- |
| 4 runs, July–August 2024 | detuning | CARDS_2500 (2,571) | ≈ 220 after the filling filter |
| 7 runs, February–March 2025 (the first two with 1 and 2 datapoints) | position | CARDS_1000 (1,000) | ≈ 160–215 after the filling filter |

Each program ran with six register copies and 40 shots per task, i.e. 240 records per readout before
the filling filter — except the first detuning run (10 datapoints), which used 166 shots per task
(≈ 1,000 records). The filling filter keeps a copy only if every trap was filled
([hardware.md](hardware.md)); kept records per readout: detuning median 222 (198–238; ≈ 950 for the
10 datapoints of the first run), position median 195 (124–230).

The Aquila feature files above are the reduction of every kept record, for all datapoints except the
10 of the first detuning run, which the detuning file holds as a 240-record subsample. The deposit
holds the records themselves (`results/hardware_noise/bits/`): one flat array of bitstrings per
encoding and an index giving, for each datapoint and readout, the offset and count of its records.
`analysis/hardware_noise/common.py:Records` reads them. The raw records are there too (`results/hardware_noise/raw/`): for
every task (datapoint × readout) the executed program and, per shot, the pre-sequence (traps filled
before the pulse), the post-sequence (measured state) and the shot status, written by
`analysis/hardware_noise/export_raw.py` without device or account identifiers. The post-selection
statistics (Fig S2) and the pulse figure (Fig 4) are computed from them.

## Experiment records

Shot-based runs — emulated with `backend = "shots"`, or on hardware — are recorded as a directory
per run under `data/experiments/`, holding the shot records, the serialised pulse programs and the
reduced feature CSVs. The layout is in [architecture.md](architecture.md#experiments-persistence-for-shot-based-runs).
`analysis/hardware_noise/export_bits.jl` converts such directories, or a `…_hwbits.jld2` file, into
the flat record format above.

## Run logs

Classification writes one log, one manifest and one predictions file per run (format in
[pipeline.md](pipeline.md#stage-3--classification)). The analyses read them from
`results/<topic>/logs/`. The deposit holds the logs behind every figure and table, and the master
tables of all runs (`results/tables/`).
