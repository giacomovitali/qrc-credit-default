#!/usr/bin/env bash
# Classification sweep on the independent stratified subsamples (make_subsets.py): 10 of 2,571 and 10 of
# 1,000 samples, each with the benchmark protocol (--split random, seeds 1-10, four resamplers, validation
# selection, 1 BLAS thread per process) and the same inputs and models as the baseline/benchmark runs on the actual
# subsets: fc12 = the paper's six models + DNN on the 12 features (FC, DNN), fc24 = the tree baselines on
# the 24 variables (trees24), crc = CRC, dethw = QRC detuning, hardware-protocol emulation, and on the
# size-1,000 subsamples poshw = QRC position, hardware-protocol emulation (run_position.sh).
# Regression: the actual subsets written through the same path (tags cards2500, cards1000), seed 1, must
# reproduce the baseline/benchmark runs.
# Logs: results/subsets/logs/<tag>_<input>_<seed>.log; a run is skipped once its predictions exist.
# SEEDS / TAGS / NPROC restrict the sweep (pilot: SEEDS=1 TAGS="n2571s01 n1000s01").
cd "$(dirname "$0")/../.."
# Run inside the qrc_class environment (docs/installation.md).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S=results/subsets/subsets L=results/subsets/logs
SEEDS=${SEEDS:-$(seq 1 10)} NPROC=${NPROC:-24}
TAGS=${TAGS:-$(for n in 2571 1000; do for k in $(seq -w 1 10); do echo n${n}s$k; done; done)}
mkdir -p $L/fig
{
  # trees first (the slowest runs), larger subsamples first
  for inp in fc24 fc12 crc dethw poshw; do
    case $inp in
      fc24) models="DTR RFR ET HGB" ;;
      fc12) models="KNN SGD NBT LM P LSVM DNN" ;;
      *) models="KNN SGD NBT LM P LSVM" ;;
    esac
    for tag in $TAGS; do
      [ $inp = poshw ] && [[ $tag != n1000s* ]] && continue
      for s in $SEEDS; do
        echo "${tag}_${inp}_$s|$S/${tag}_$inp.csv|$S/${tag}_labels.csv|$models|$s"
      done
    done
  done
  [ -n "$NO_REGRESSION" ] || for tag in cards2500 cards1000; do
    echo "${tag}_fc24_1|$S/${tag}_fc24.csv|$S/${tag}_labels.csv|DTR RFR ET HGB|1"
    echo "${tag}_fc12_1|$S/${tag}_fc12.csv|$S/${tag}_labels.csv|KNN SGD NBT LM P LSVM DNN|1"
    echo "${tag}_crc_1|$S/${tag}_crc.csv|$S/${tag}_labels.csv|KNN SGD NBT LM P LSVM|1"
    echo "${tag}_dethw_1|$S/${tag}_dethw.csv|$S/${tag}_labels.csv|KNN SGD NBT LM P LSVM|1"
  done
} | xargs -P $NPROC -I{} bash -c 'IFS="|" read -r name feat lab models seed <<< "{}"
  [ -f results/subsets/logs/$name.predictions.csv.gz ] || python classification/run_classification.py \
    --features $feat --labels $lab --split random --classifiers $models --seed $seed \
    --run-name $name --log-dir results/subsets/logs --figure-dir results/subsets/logs/fig --n-jobs 1 \
    > /dev/null 2>&1 || echo "FAILED $name"
  echo "finished $name"'
echo "done $(ls $L/*.predictions.csv.gz | wc -l) runs"
