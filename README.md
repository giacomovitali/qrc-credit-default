# Quantum reservoir computing for credit-card default prediction

Code for the article *Quantum Reservoir Computing for Credit Card Default Prediction on a Neutral
Atom Platform* (G. Vitali et al., Advanced Quantum Technologies; see [Citation](#citation)).

Tabular credit-risk features are encoded into the Hamiltonian of a chain of Rydberg atoms, either
as local detunings or as atom positions. The chain evolves, its single-site and two-site
correlations ⟨Zᵢ⟩ and ⟨ZᵢZⱼ⟩ are read out at five times, and these readouts are the input of
standard classifiers. The reservoir runs in three ways:

- **statevector emulation** of the programs executed on the device (exact expectation values);
- **shot-based emulation**, sampling measurements from the same states;
- **QuEra Aquila**, the 256-atom analog quantum computer, accessed through the cloud.

The task is the UCI *default of credit card clients* dataset. The quantum reservoir is compared with
a classical (mean-field) reservoir, the same classifiers on the original features, a deep neural
network and tree ensembles.

Reproducing the results does **not** need quantum hardware: the emulation runs on a workstation,
and the measurements taken on Aquila are in the data deposit (see [Data](#data)).

## Documentation

| Document | Contents |
| --- | --- |
| [docs/installation.md](docs/installation.md) | Julia and Python environments |
| [docs/pipeline.md](docs/pipeline.md) | The pipeline stages and every setting of their drivers |
| [docs/architecture.md](docs/architecture.md) | Encodings, backends, registers, readouts, seeding |
| [docs/data.md](docs/data.md) | Input data, datasets and splits, file formats, provenance of the hardware runs |
| [docs/reproducing.md](docs/reproducing.md) | Which command produces which figure and table, and how long it takes |
| [docs/hardware.md](docs/hardware.md) | How the hardware runs were built, submitted and post-selected |

## Repository layout

```
preprocessing/       cleaning, scaling, PCA-based feature aggregation, subsets, split seeds   (Python)
quantum_evolution/   drivers: run the reservoir, reduce measurement records to features      (Julia)
qrc_module/          the reservoir: encodings, backends, hardware programs, experiments        (Julia package)
classical_rc/        classical (mean-field) reservoir                                          (Julia)
classification/      classifiers, resampling, evaluation, run logs                             (Python)
result_analysis/     reading run logs and manifests into tables                                (Python)
analysis/            the analyses of the article, one directory per topic                      (Python, Julia)
configs/             reservoir configurations and the CARDS_30000 split seeds
docs/                documentation
```

Outputs go to `results/` (not versioned).

## Quick start

With the environments installed ([docs/installation.md](docs/installation.md)) and the data in
place ([docs/data.md](docs/data.md)), from the repository root:

```bash
conda activate qrc_class

# 1. clean, scale and aggregate the raw dataset
python preprocessing/preprocessing.py

# 2. reservoir: statevector emulation of the hardware protocol, detuning encoding, first 50 samples
julia --project=. -t auto quantum_evolution/run_reservoir.jl configs/reservoir/detuning_hardware.toml \
    sets=qpu_samples first=1 last=50 output_dir=results/quickstart

# 3. classification on the full emulated features of CARDS_2500 (from the data deposit), one seed
python classification/run_classification.py \
    --features results/emulation/detuning_encoding_qpu_samples_12_delta_7.6_deltamax_-9.0_hw.csv \
    --labels dataset_aquila/labels_qpu.csv --classifiers LM LSVM --seed 1 \
    --run-name quickstart --log-dir results/quickstart/logs

# 4. tabulate
python result_analysis/runs.py results/quickstart/logs
```

Every command runs **from the repository root**. Drivers are configured by TOML files plus
`key=value` overrides, and every run writes a manifest with its settings, the git commit and the
checksums of its inputs.

## Data

The UCI dataset is available from the UCI Machine Learning Repository
([doi:10.24432/C55S3H](https://doi.org/10.24432/C55S3H), CC BY 4.0). The derived data — preprocessed
features, split indices, reservoir features, the Aquila and emulated measurement records, and the
classification logs behind every figure and table — are on Zenodo:
[doi:10.5281/zenodo.24131386](https://doi.org/10.5281/zenodo.24131386) (CC BY 4.0).
[docs/data.md](docs/data.md) describes the files and where to put them.

## Citation

Please cite the article and, for the code, this repository; the metadata is in
[`CITATION.cff`](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE). The data deposit is licensed separately under CC BY 4.0.
