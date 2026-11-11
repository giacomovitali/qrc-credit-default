#!/usr/bin/env bash
# Position encoding, hardware-protocol emulation (configs/reservoir/position_hardware.toml) of the union
# of the size-1,000 subsamples (position_input/pos1000.csv, written by make_subsets.py; scaled by
# qpu_samples like CARDS_1000). No bitstrings stored. Output: results/subsets/emulation/. Then
# make_subsets.py again, which checks the IDs shared with CARDS_1000 against the published features and
# writes the per-subsample <tag>_poshw.csv files.
cd "$(dirname "$0")/../.."
julia --project=. -t 27 --heap-size-hint=40G quantum_evolution/run_reservoir.jl configs/reservoir/position_hardware.toml \
  input_dir=results/subsets/position_input sets=pos1000 scale_reference=qpu_samples \
  output_dir=results/subsets/emulation save_bitstrings=false || exit 1
# Run inside the qrc_class environment (docs/installation.md).
python preprocessing/make_subsets.py
