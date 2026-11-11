#!/usr/bin/env bash
# Training times for the timing table of the paper (CARDS_30000, per resampler), measured as the other rows
# of the table: time of the hyperparameter search / number of grid points = mean time per configuration
# (fit on the resampled training set + validation prediction). THREADS cores per process (BLAS/OpenMP threads
# and --n-jobs; default 1), CARDS_30000 pre-registered split seeds, QPU pair first (= baseline/benchmark).
#   trees: DTR RFR ET HGB on the 24 variables (the "best tree model" of the paper's comparisons)
#   ref:   the rows already in the table, re-timed on this machine under the same conditions:
#          KNN SGD NB LM P on the QRC features (noiseless emulation, detuning) and the DNN on the 12 features;
#          LSVM (LinearSVC) on the QRC features: PART=ref MODELS=LSVM
# One process per (input, model, seed), longest first, P at a time; a run is skipped once its predictions exist.
# Logs: results/timing/logs/c30000_<input>_<model>_<seed>.log
#
#   PART=trees|ref|all  MODELS="LSVM"  SEEDS="8 95"  P=10  THREADS=1  OUT=<log dir>  bash analysis/timing/run_timing.sh
cd "$(dirname "$0")/../.."
# Run inside the qrc_class environment (docs/installation.md).
THREADS=${THREADS:-1}
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
SEEDS=${SEEDS:-$(grep -v '^#' configs/cards30000_seeds.txt)}
PART=${PART:-all} P=${P:-10} OUT=${OUT:-results/timing/logs}
E=results/emulation D=dataset_aquila F=results/baselines/features
DET=$E/detuning_encoding_qpu_samples_12_delta_7.6_deltamax_-9.0 DETE=$E/detuning_encoding_emulation_samples_12_delta_7.6_deltamax_-9.0
LAB="$D/labels_qpu.csv $D/labels_emu.csv"
mkdir -p $OUT/fig
echo "start $(date -Is) part=$PART P=$P threads=$THREADS seeds=$SEEDS; $(uptime)"
{
  for m in ${MODELS:-RFR ET DNN HGB SGD DTR KNN LM P NB}; do
    for s in $SEEDS; do
      case $m in
        DTR|RFR|ET|HGB) [ $PART = ref ] || echo "c30000_fc24_${m}_$s|$F/qpu_samples_24.csv $F/emulation_samples_24.csv|$m|$s" ;;
        DNN)            [ $PART = trees ] || echo "c30000_fc12_${m}_$s|$D/qpu_samples.csv $D/emulation_samples.csv|$m|$s" ;;
        *)              [ $PART = trees ] || echo "c30000_dethw_${m}_$s|${DET}_hw.csv ${DETE}_hw.csv|$m|$s" ;;
      esac
    done
  done
} | xargs -P $P -I{} bash -c 'IFS="|" read -r name feat model seed <<< "{}"
  [ -f '$OUT'/$name.predictions.csv.gz ] || python classification/run_classification.py \
    --features $feat --labels '"$LAB"' --split random --classifiers $model --seed $seed \
    --run-name $name --log-dir '$OUT' --figure-dir '$OUT'/fig --n-jobs '$THREADS' > /dev/null 2>&1 || echo "FAILED $name"
  echo "finished $name $(date -Is) load $(cut -d" " -f1 /proc/loadavg)"'
echo "done $(ls $OUT/*.predictions.csv.gz | wc -l) runs $(date -Is); $(uptime)"
