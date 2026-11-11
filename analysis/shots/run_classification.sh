#!/usr/bin/env bash
# Shot-count classification sweep: every feature set of make_features.py plus the references (ideal sv,
# hardware-protocol model, 12 preprocessed features), seeds 1-10, the paper's six models x four
# resamplers. Detuning on CARDS_2500, position on CARDS_1000. 6 processes x 4 BLAS threads.
# Logs: results/shots/logs/<enc>_<set>_<seed>.log; a run is skipped once its predictions exist.
cd "$(dirname "$0")/../.."
# Run inside the qrc_class environment (docs/installation.md).
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
F=results/shots/features E=results/emulation L=results/shots/logs
mkdir -p $L/fig
{
  for enc in detuning position; do
    if [ $enc = detuning ]; then
      lab=dataset_aquila/labels_qpu.csv stem=$E/detuning_encoding_qpu_samples_12_delta_7.6_deltamax_-9.0 fc=dataset_aquila/qpu_samples.csv
    else
      lab=dataset_aquila/labels_qpu_pos.csv stem=$E/position_encoding_qpu_samples_pos_12_delta_2.0_lambda_1.0 fc=dataset_aquila/qpu_samples_pos.csv
    fi
    echo "${enc}_sv ${stem}_sv_exact.csv $lab"
    echo "${enc}_model ${stem}_hw.csv $lab"
    echo "${enc}_fc $fc $lab"
    for f in $F/${enc}_*_s*.csv; do echo "$(basename "$f" .csv) $f $lab"; done
  done
} | while read -r name feat lab; do for s in $(seq 1 10); do echo "$name $feat $lab $s"; done; done |
xargs -P 6 -L 1 bash -c '[ -f results/shots/logs/$0_$3.predictions.csv.gz ] || python classification/run_classification.py \
  --features $1 --labels $2 --split random --classifiers KNN SGD NBT LM P LSVM --seed $3 \
  --run-name $0_$3 --log-dir results/shots/logs --figure-dir results/shots/logs/fig --n-jobs 1 > /dev/null 2>&1 \
  || echo "FAILED $0 seed $3"'
echo "done $(ls results/shots/logs/*.predictions.csv.gz | wc -l) runs"
