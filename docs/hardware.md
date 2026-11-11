# The hardware runs on QuEra Aquila

This document describes how the hardware data of the article was produced. **None of it is needed
to reproduce the results**: the measurement records are in the data deposit ([data.md](data.md)),
and the emulation reproduces the executed programs without cloud access.

## Programs

Each datapoint is one program per readout time T = 0.5, 1.0, …, 2.5 µs, each starting from the
ground state (waveforms in [architecture.md](architecture.md#what-each-backend-evolves)). The chain
is replicated six times across the lattice area, so each shot measures six copies of the same
datapoint ([architecture.md](architecture.md#register-embedding-and-replication)). The programs are
built by `build_waveform` in `qrc_module` with the Python `bloqade` package (called from Julia through
`PythonCall`), one batch per datapoint with `run_time` = T − 0.12 µs (the two 0.06 µs ramps). The
hardware-protocol emulation backend evolves exactly these programs, on a single chain and without
noise.

## Submission and retrieval

The runs went through two routes, both implemented in `qrc_module`:

| | Amazon Braket | QuEra direct access |
| --- | --- | --- |
| Credentials | AWS CLI default profile | Endpoint URL + request key |
| Submission (`shot_exec!`) | `bloqade` `….braket.aquila().run_async(…)`, `direct_access = false` | `bloqade` `….quera.custom().submit(…)` to the endpoint, `direct_access = true` (default) |
| Retrieval (`save_task`) | Through Braket, from the saved task handle | From result files downloaded from the job export |

Direct access reads its endpoint and key from the environment (`QRC_DIRECT_ACCESS_URL`,
`QRC_DIRECT_ACCESS_KEY`) or from `direct_access.conf`, a copy of `direct_access.conf.example`, which
must never be committed. `Qrcutils.directaccess_config()` raises a descriptive error if neither
provides both values; the emulation never calls it. The Braket and direct-access tooling of the
Python side is pinned in `requirements_bloqade.txt` ([installation.md](installation.md)).

A run is recorded as an experiment directory (`QRC_<dim>_emu_false_<timestamp>`): the serialised
program of each task under `waveforms/`, a submission handle under `tasks/`, and the results under
`results/` ([architecture.md](architecture.md#experiments-persistence-for-shot-based-runs)).
`Qrcutils.save_task(exp, id)` collects the results of a datapoint once all its readout times have
completed, stores the raw records and reduces them to per-readout bitstring files; datapoints with
tasks still pending are skipped, so retrieval can be repeated. A retried task reuses the serialised
program, so it is identical to the original. The repository has no driver for new hardware runs:
submitting needs device access and is outside the scope of reproducing the article.

## Post-selection

Aquila reports, for each shot, a *pre-sequence* (which traps were filled before the pulse) and a
*post-sequence* (the measured state). `bitstrings_filtered` splits each shot into the six copies and
keeps a copy only if its pre-sequence shows every trap filled. Imperfect filling is the main source
of discarded records: the median fraction of copies kept per task is 0.93 for the detuning runs and
0.81 for the position runs (`analysis/hardware_noise/postselection.py`).
