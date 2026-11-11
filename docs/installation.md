# Installation

The pipeline spans two languages and three environments. This is deliberate: the classification
stack (PyTorch, scikit-learn, imbalanced-learn) and the quantum stack (`bloqade`, Amazon Braket)
have incompatible dependency sets, so they are installed separately and never activated together.

| Environment | Used by | Specification |
| --- | --- | --- |
| Julia project at the repository root | `quantum_evolution/`, `qrc_module/`, `classical_rc/`, `analysis/hardware_noise/*.jl` | `Project.toml`, `Manifest.toml` |
| Conda env, here called `qrc_class` | `preprocessing/`, `classification/`, `result_analysis/`, `analysis/` | `requirements.txt` |
| Conda env, here called `bloqade_py` | Python-side hardware tooling (Braket SDK); not needed to reproduce the results | `requirements_bloqade.txt` |

The environment names are conventions used throughout the documentation; any name works as long as
you are consistent.

## 1. Julia

Install **Julia 1.9.3**. [`juliaup`](https://github.com/JuliaLang/juliaup) makes the version pin
straightforward:

```bash
curl -fsSL https://install.julialang.org | sh
juliaup add 1.9.3
juliaup default 1.9.3
julia --version        # expect: julia version 1.9.3
```

Instantiate the project from the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.status()'
```

`Pkg.status()` should list `qrc_module v0.1.0 \`qrc_module\``, resolved by relative path, alongside
`Bloqade`, `BloqadeSchema`, `CUDA` and the rest. `Manifest.toml` pins exact versions of the whole
dependency graph; do not delete it if you want to reproduce published numbers.

The first `using qrc_module` compiles the dependency tree and can take several minutes. It also
provisions a private Conda environment under `qrc_module/.CondaPkg/env`, driven by
`qrc_module/CondaPkg.toml`, which supplies the Python `bloqade` package that Julia calls through
`PythonCall`. This environment is managed automatically — it is **not** one of the two conda
environments below, and you never activate it yourself.

**GPU (optional).** Shot-based emulation moves the evolution to the GPU when `CUDA.functional()`
reports a usable device, and runs on the CPU otherwise. No configuration is needed either way.

**Threads.** The statevector backend distributes datapoints over Julia threads; start Julia with
`-t auto` to use every core. Results do not depend on the thread count.

## 2. Python environment for classification and analysis

```bash
conda create -n qrc_class python=3.11
conda activate qrc_class
pip install -r requirements.txt
```

This provides NumPy, SciPy, pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn and PyTorch. It has no
`bloqade` and cannot submit quantum jobs.

`requirements.txt` pins the direct dependencies to the versions used for the article. PyTorch is
used only by the DNN; the default wheel brings CUDA libraries, which are not needed (no GPU is used):
`pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu` installs the CPU build.

## 3. Python environment for the quantum tooling

Needed only to work with Aquila directly — skip it if you are reproducing the results from the
measurement records in the data deposit.

```bash
conda create -n bloqade_py python=3.11
conda activate bloqade_py
pip install -r requirements_bloqade.txt
```

This provides `bloqade` 0.15.14 and `amazon-braket-sdk` 1.86.1.

## 4. Cloud credentials (optional)

Only required for running programs on the device. Two independent routes exist, described in
[hardware.md](hardware.md):

- **Amazon Braket.** Standard AWS credentials, picked up from the default profile of the AWS CLI.

- **QuEra direct access.** An endpoint URL and a request key, supplied either through the
  environment or through an untracked config file:

  ```bash
  cp direct_access.conf.example direct_access.conf
  # then edit direct_access.conf, or instead export:
  export QRC_DIRECT_ACCESS_URL="..."
  export QRC_DIRECT_ACCESS_KEY="..."
  ```

  `direct_access.conf` is excluded from version control and must stay that way. Emulation never
  reads these values.

## 5. Verifying the installation

```bash
# Julia side: the reservoir package loads, and a two-datapoint statevector run completes
julia --project=. -e 'using qrc_module; println("qrc_module OK")'
julia --project=. quantum_evolution/run_reservoir.jl configs/reservoir/detuning_hardware.toml \
    sets=qpu_samples first=1 last=2 output_dir=results/install_check

# Python side
conda activate qrc_class && python -c "import torch, sklearn, imblearn, seaborn; print('qrc_class OK')"
conda activate bloqade_py && python -c "import bloqade, braket; print('bloqade_py OK')"
```
