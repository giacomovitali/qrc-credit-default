#!/usr/bin/env bash
# Baseline classification sweep: the 12 aggregated features (fc12) and the 24 preprocessed variables
# (fc24, make_features24.py) on CARDS_30000 (the pre-registered split seeds,
# configs/cards30000_seeds.txt), CARDS_2500 and CARDS_1000 (seeds 1-10), every model
# (paper's linear models + DNN, and the tree baselines DTR RFR ET HGB) x four resamplers.
# CARDS_30000 takes the QPU pair first; its
# runs are split in two processes (trees / the rest). 6 processes x 4 threads, longest jobs first.
# Logs: results/baselines/logs/<set>_<input>_<part>_<seed>.log; a run is skipped once its
# predictions exist.
cd "$(dirname "$0")/../.."
# Run inside the qrc_class environment (docs/installation.md).
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
S30000=${S30000:-$(grep -v '^#' configs/cards30000_seeds.txt)}   # CARDS_2500/1000: seeds 1-10
F=results/baselines/features D=dataset_aquila L=results/baselines/logs
mkdir -p $L/fig
TREES="DTR RFR ET HGB" REST="KNN SGD NBT LM P LSVM DNN"
{
  for part in trees rest; do
    [ $part = trees ] && models=$TREES || models=$REST
    for s in $S30000; do
      echo "c30000_fc12_${part}_$s|$D/qpu_samples.csv $D/emulation_samples.csv|$D/labels_qpu.csv $D/labels_emu.csv|$models|$s"
      echo "c30000_fc24_${part}_$s|$F/qpu_samples_24.csv $F/emulation_samples_24.csv|$D/labels_qpu.csv $D/labels_emu.csv|$models|$s"
    done
  done
  for s in $(seq 1 10); do
    echo "c2500_fc12_all_$s|$D/qpu_samples.csv|$D/labels_qpu.csv|$TREES $REST|$s"
    echo "c2500_fc24_all_$s|$F/qpu_samples_24.csv|$D/labels_qpu.csv|$TREES $REST|$s"
    echo "c1000_fc12_all_$s|$D/qpu_samples_pos.csv|$D/labels_qpu_pos.csv|$TREES $REST|$s"
    echo "c1000_fc24_all_$s|$F/qpu_samples_pos_24.csv|$D/labels_qpu_pos.csv|$TREES $REST|$s"
  done
} | xargs -P 6 -I{} bash -c 'IFS="|" read -r name feat lab models seed <<< "{}"
  [ -f results/baselines/logs/$name.predictions.csv.gz ] || python classification/run_classification.py \
    --features $feat --labels $lab --split random --classifiers $models --seed $seed \
    --run-name $name --log-dir results/baselines/logs --figure-dir results/baselines/logs/fig --n-jobs 4 \
    > /dev/null 2>&1 || echo "FAILED $name"
  echo "finished $name"'
echo "done $(ls $L/*.predictions.csv.gz | wc -l) runs"
