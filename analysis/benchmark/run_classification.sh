#!/usr/bin/env bash
# Benchmark classification sweep: the reservoir pipelines — CRC, QRC-hw (noiseless hardware-protocol emulation),
# QRC-shot (240 shots from the same model) and QRC-Aquila (the published hardware features) — on
# CARDS_30000 (detuning; no hardware), CARDS_2500 (detuning) and CARDS_1000 (detuning and position),
# seeds 1-10 (CARDS_30000: the pre-registered split seeds), the paper's six models x four resamplers. FC, DNN and the tree baselines come from the baselines sweep
# (same splits and seeds). CARDS_30000 takes the QPU pair first (as the baselines sweep) and
# runs one process per model, LSVM first (liblinear, single-core: ~15 min per run there). 24 processes x 1
# thread.
# Logs: results/benchmark/logs/<set>_<pipeline>[_<model>]_<seed>.log; a run is skipped once its
# predictions exist.
cd "$(dirname "$0")/../.."
# Run inside the qrc_class environment (docs/installation.md).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S30000=${S30000:-$(grep -v '^#' configs/cards30000_seeds.txt)}   # CARDS_2500/1000: seeds 1-10
E=results/emulation D=dataset_aquila A=datasets L=results/benchmark/logs
DET=$E/detuning_encoding_qpu_samples_12_delta_7.6_deltamax_-9.0 DETE=$E/detuning_encoding_emulation_samples_12_delta_7.6_deltamax_-9.0
POS=$E/position_encoding_qpu_samples_pos_12_delta_2.0_lambda_1.0
mkdir -p $L/fig
MODELS="KNN SGD NBT LM P LSVM"
{
  for m in LSVM SGD KNN LM P NBT; do
    for s in $S30000; do
      echo "c30000_crc_${m}_$s|$D/qpu_samples_crc_labeled.csv $D/emulation_samples_crc_labeled.csv|$D/labels_qpu.csv $D/labels_emu.csv|$m|$s"
      echo "c30000_dethw_${m}_$s|${DET}_hw.csv ${DETE}_hw.csv|$D/labels_qpu.csv $D/labels_emu.csv|$m|$s"
      echo "c30000_detshot_${m}_$s|${DET}_hwshots_240.csv ${DETE}_hwshots_240.csv|$D/labels_qpu.csv $D/labels_emu.csv|$m|$s"
    done
  done
  for s in $(seq 1 10); do
    for lab in qpu qpu_pos; do
      [ $lab = qpu ] && set=c2500 || set=c1000
      echo "${set}_crc_$s|$D/qpu_samples_crc_labeled.csv|$D/labels_$lab.csv|$MODELS|$s"
      echo "${set}_dethw_$s|${DET}_hw.csv|$D/labels_$lab.csv|$MODELS|$s"
      echo "${set}_detshot_$s|${DET}_hwshots_240.csv|$D/labels_$lab.csv|$MODELS|$s"
      echo "${set}_detaquila_$s|$A/detuning_encoding_qpu_samples_240shots_real_aquila.csv|$D/labels_$lab.csv|$MODELS|$s"
    done
    echo "c1000_poshw_$s|${POS}_hw.csv|$D/labels_qpu_pos.csv|$MODELS|$s"
    echo "c1000_posshot_$s|${POS}_hwshots_240.csv|$D/labels_qpu_pos.csv|$MODELS|$s"
    echo "c1000_posaquila_$s|$A/position_encoding_qpu_samples_240shots_real_aquila_lambda1.0_delta2.0.csv|$D/labels_qpu_pos.csv|$MODELS|$s"
  done
} | xargs -P 24 -I{} bash -c 'IFS="|" read -r name feat lab models seed <<< "{}"
  [ -f results/benchmark/logs/$name.predictions.csv.gz ] || python classification/run_classification.py \
    --features $feat --labels $lab --split random --classifiers $models --seed $seed \
    --run-name $name --log-dir results/benchmark/logs --figure-dir results/benchmark/logs/fig --n-jobs 1 \
    > /dev/null 2>&1 || echo "FAILED $name"
  echo "finished $name"'
echo "done $(ls $L/*.predictions.csv.gz | wc -l) runs"
