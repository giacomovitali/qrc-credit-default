#!/usr/bin/env bash
# CARDS_1000 classification of every pipeline in results/hardware_noise/enc_pipelines.txt, seeds 1-10,
# 6 processes in parallel. Logs: results/hardware_noise/enc_logs/<pipeline>_s<seed>.log (skips existing).
cd "$(dirname "$0")/../.."
# Run inside the qrc_class environment (docs/installation.md).
while read -r name feat; do for s in $(seq 1 10); do echo "$name $feat $s"; done; done < results/hardware_noise/enc_pipelines.txt |
xargs -P 6 -L 1 bash -c '[ -f results/hardware_noise/enc_logs/$0_s$2.log ] || python classification/run_classification.py \
  --features $1 --labels dataset_aquila/labels_qpu_pos.csv --classifiers KNN SGD NBT LM P LSVM --seed $2 \
  --run-name $0_s$2 --log-dir results/hardware_noise/enc_logs --figure-dir results/hardware_noise/enc_logs/fig --n-jobs 1 > /dev/null 2>&1 \
  || echo "FAILED $0 seed $2"'
echo "done $(ls results/hardware_noise/enc_logs/*.log | wc -l) logs"
