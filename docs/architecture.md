# Architecture of the reservoir

This document describes how a row of tabular data becomes a feature vector. The driver scripts in
`quantum_evolution/` are thin; everything below lives in the Julia package `qrc_module/`.

## Module layout

```
qrc_module
└── Qrcutils              encodings, waveforms, backends, register embedding
    └── ExperimentUtils   experiment objects, persistence, feature reduction
        ├── qcutils       ⟨Zᵢ⟩ and ⟨ZᵢZⱼ⟩ from measurement records
        └── HardwareUtils  Braket submission, schema translation, shot filtering
```

Only `Qrcutils` is exported, so everything is reached as `Qrcutils.<name>`.

## The physical system

A chain of Rydberg atoms under a driving laser, governed by

```
H(t)/ħ = Ω(t)/2 · Σᵢ (|gᵢ⟩⟨rᵢ| + h.c.)  −  Σᵢ Δᵢ(t) nᵢ  +  Σᵢ<ⱼ C₆/|rᵢ − rⱼ|⁶ · nᵢnⱼ
```

with Rabi frequency Ω, detunings Δᵢ and the van der Waals interaction that couples atoms according
to their separation. Atoms start in the ground state; the pulse ramps Ω up over 0.06 µs, holds it,
and ramps down over 0.06 µs. The reservoir is *fixed* — nothing about it is trained. Only the
classifier that reads it out is fitted.

Data enters the Hamiltonian in one of two ways.

## Encoding 1 — detuning

`DetuningLayer` places `dim` atoms in a chain of fixed spacing (10 µm by default) and maps each
feature onto the **local detuning** of one atom. Feature values, scaled to a fixed range, are
rescaled to a detuning window set by `Δ_max` and `Δ`. Geometry is identical for every datapoint;
only the detuning pattern changes.

## Encoding 2 — position

`PositionLayer` keeps the detuning uniform and maps each feature onto **atom separation**. Starting
from an evenly spaced chain of `dim + 1` atoms, feature *k* displaces every atom from index *k + 1*
onward, so feature *k* controls the gap between atoms *k* and *k + 1*. The displacement is
`scale/6 · λ · xₖ`, with `λ` a global scale knob.

Because the interaction falls off as 1/r⁶, the features modulate the interaction strengths directly
rather than the local fields. This needs one more atom than the detuning encoding, which is why a
12-feature dataset produces 13-atom registers here and 12-atom registers there.

## Backends

`apply_layer` dispatches on two flags, `Aquila` and `emu`.

### Statevector emulation (`Aquila = false`)

`statevector_sim` builds the Hamiltonian with `rydberg_h`, integrates the Schrödinger equation
(`SchrodingerProblem` with the `Vern8` integrator), and evaluates each readout operator exactly at
the requested times. There is no sampling noise, so this is the idealised limit of the reservoir.

### Shot-based emulation (`Aquila = true`, `emu = true`)

`shot_exec!` performs Krylov time evolution — on the GPU when CUDA is available, otherwise on the
CPU — and draws projective measurements at each readout time, reproducing the finite-shot
statistics of a real device. Results are stored as an *experiment* (below) rather than returned, so
the shot records can be re-reduced later at different shot counts.

### Hardware (`Aquila = true`, `emu = false`)

The same waveforms are translated to a device schema and submitted to QuEra Aquila, either
asynchronously through Amazon Braket or through the QuEra direct-access route. Results arrive
out-of-band and are retrieved separately; see [hardware.md](hardware.md).

### What each backend evolves

The three backends do not implement exactly the same pulse sequence, and their features differ by
more than sampling noise:

- **Hardware.** Each readout time *T* is a separate program starting from the ground state: Ω ramps
  up over 0.06 µs, holds, and ramps down over the final 0.06 µs, after which the atoms are measured.
  The detuning has a global part that is constant from t = 0 (7.64 rad/µs for the detuning
  encoding, 2.0 rad/µs for the position encoding). For the detuning encoding it adds a local part
  (1 − fᵢ)·Δ_max, Δ_max = −9 rad/µs, ramped in and out over 0.06 µs like Ω. During the hold this
  gives the same Δᵢ as the emulation backends; the ramps differ, since the emulators ramp the total
  detuning up from zero.
- **Statevector.** A single pulse to `t_end`, with Ω held through the intermediate readouts and
  ramped down only at the end. The integrator is adaptive, and each intermediate readout is taken at
  the integrator's step endpoint at or just after the requested time — for the first QPU datapoint
  0.517, 1.005, 1.510 and 2.016 µs instead of 0.5, 1.0, 1.5 and 2.0; the final readout is at
  exactly `t_end`. Individual observables differ by up to about 0.09 from their values at the exact
  times, mostly at the first readout.
- **Shot-based emulation.** One Hamiltonian per readout time *T*, each with its own ramp-down ending
  at *T*, but the evolution continues from the previous readout's state instead of restarting from
  the ground state, and each interval is advanced by a coarse Krylov step with the Hamiltonian
  sampled at the interval's clock points (0, 0.05 and 0.5 µs for the first, then *T* − 0.5 and *T*).
  Its exact (infinite-shot) expectation values differ from the statevector backend's by up to about
  0.35 in individual observables — well above the sampling noise at the shot counts used. This
  backend is not used for the results of the article.

Figures measured on the first QPU datapoints with the position encoding. Each backend is a
deterministic function of its input and therefore a legitimate reservoir, but they are three
different reservoirs: differences between them in downstream classification metrics are not
attributable to sampling noise alone.

## Register embedding and replication

A hardware register is a 2-D area, while both encodings produce a 1-D chain. The chain is laid
along the longer axis of the area — 128 µm, using the extended lattice height — and for the
configurations in the paper it fits in a straight line: 110 µm for the 12-atom detuning chain, and
between roughly 100 and 120 µm for the 13-atom position chain, whose gaps shrink by up to
`scale/6 · λ`. Only a chain longer than that axis is folded by `register_embedding!`, which then
lays it out as a serpentine: along one axis until the next atom would leave the area, one step
across, and back.

A single chain would leave most of the area unused, and unused area is wasted shots. `replicate!`
therefore tiles as many **copies** of the chain as fit side by side, 1.5 chain spacings (15 µm)
apart, and the register is centred in the area. In the paper's configuration that is **six
parallel copies** — 72 atoms for the detuning encoding, 78 for the position encoding. Each shot
then measures six independent instances of the same datapoint at once, which is why the submitted
shot count is

```
shots per task = ⌊requested shots / copies⌋
```

On readout, `bitstrings_filtered` slices each shot into per-copy bitstrings. Aquila reports both a
*pre-sequence* (which traps were actually filled before the pulse) and a *post-sequence* (the state
after it); a copy is kept only if its pre-sequence shows every trap filled, and discarded otherwise.
Imperfect filling is the dominant source of discarded data on hardware.

Emulation skips replication entirely (`filling = !emu`): one copy, no filtering. The emulated
register is therefore the same straight chain as the statevector backend's, merely translated.

Replicas 15 µm apart are not fully isolated: the van der Waals coupling there is
C₆/(15 µm)⁶ ≈ 0.48 rad/µs, against 5.4 rad/µs between nearest neighbours at 10 µm and Ω = 2π rad/µs.
A single-chain emulation does not include this residual inter-replica coupling.

## Readouts and the feature vector

At each readout time the reservoir is measured in the computational basis and reduced to

- the single-site magnetisations ⟨Zᵢ⟩, one per atom, and
- the two-site correlators ⟨ZᵢZⱼ⟩ for every pair *i < j*.

These are concatenated across the readout times into one vector per datapoint. With the default
schedule — evolve to 2.5 µs, read out every 0.5 µs, giving five readout times:

| Encoding | Atoms | Per readout time | Feature vector |
| --- | --- | --- | --- |
| Detuning | 12 | 12 + 66 = 78 | **390** |
| Position | 13 | 13 + 78 = 91 | **455** |

(66 and 78 are the pair counts for 12 and 13 atoms.)

In the feature files (one column per datapoint), readout *k* = 1…5 (0.5…2.5 µs) occupies rows
(*k* − 1)·*b* … *k*·*b* − 1, with *b* = 78 or 91. Within a block come the *n* values ⟨Zᵢ⟩, then ⟨ZᵢZⱼ⟩ in
row-major upper-triangle order: (1,2), (1,3), …, (1,*n*), (2,3), …. This holds for the statevector,
hardware-protocol, shot-emulation and *Aquila* files: with this pairing every joint probability
(1 ± ⟨Zᵢ⟩ ± ⟨Zⱼ⟩ ± ⟨ZᵢZⱼ⟩)/4 is non-negative. The mean-field (CRC) files have the same length but are
not ±1-spin moments.

For shot-based backends the expectation values are estimated from the measurement counts
(`compute_zi`, `compute_zij`); for the statevector backend they are computed exactly.

## Experiments: persistence for shot-based runs

Any shot-based run is recorded as an `Experiment`, which owns a directory:

```
data/experiments/QRC_<dim>_emu_<bool>_<timestamp>/
├── experiment.jld2                      the Experiment object; reload with resume_experiment
├── tasks/task_<id>.{json,jls}           submission handles
├── waveforms/task_<id>.jls              the serialised pulse program
├── results/task_<id>/
│   ├── raw_data/results_<id>.json       raw shot records as returned
│   └── task_<id>_step_<t>.jld2          filtered bitstrings, one file per readout time
└── datasets_<shots>/task_<id>.csv       reduced feature vector, one file per datapoint
```

`<id>` is the datapoint ID carried from preprocessing, so hardware records stay joinable to labels.

Because the raw bitstrings are kept, the shot-count sweep costs nothing extra: the stored records
are subsampled and re-reduced. `experiment_features(exp; n_shots, seed)` returns the whole
experiment as one matrix and is what `quantum_evolution/reduce_experiment.jl` uses. The older
`compute_exp_values(exp; n_shots, seed, dataset_dir)` writes one CSV per datapoint into
`dataset_dir` (default `datasets/`) and **skips datapoints whose CSV already exists**.

## Randomness and provenance

Two random draws enter a shot-based run, and both are seeded per datapoint so that results do not
depend on iteration order, on how a run is split into ranges, or on interruption and resumption:

- **Measurement** — `run_reservoir.jl` seeds the RNG from `(seed, datapoint ID)` before each
  datapoint's evolution.
- **Shot subsampling** — `experiment_features` draws each datapoint's subsample from an RNG seeded
  with `(seed, n_shots, datapoint ID)`.

With `seed = nothing` the library functions fall back to the global RNG, and results are then
reproducible statistically, not bit for bit; the drivers always pass a seed.
The statevector backend involves no randomness.

Every driver writes a JSON manifest beside its output — resolved settings, script arguments, git
commit and whether the working tree was dirty, Julia or Python and package versions, input
checksums — so each output can be traced to the exact code and inputs that produced it.

> **Naming quirk.** Per-timestep files are labelled with `experiment.timesteps[i]` for *i* = 1…5,
> i.e. 0.0, 0.5, 1.0, 1.5, 2.0 — while the *i*-th record actually corresponds to an evolution of
> total duration 0.5, 1.0, 1.5, 2.0, 2.5 µs. The label is an index, offset by one from the physical
> duration. (The submitted `run_time` is the duration minus 0.12 µs, the two ramps.)

## Orientation of the feature files

Reservoir output CSVs are written with **one column per datapoint**, the column name being the
datapoint ID, and one row per feature. The Python stages therefore transpose them on load. Files
whose name ends in `_ordered` have already had their columns reordered to match the ID order of the
label file.
