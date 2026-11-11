#!/usr/bin/env bash
# Ablation sweep: the rungs that baseline/benchmark did not already run, with the benchmark protocol (the paper's six models x
# four resamplers, --split random, seeds 1-10, QPU pair first on CARDS_30000, 1 BLAS thread per process).
# CARDS_30000 uses the pre-registered split seeds (configs/cards30000_seeds.txt), and its
# pca / rff / tanh maps are fitted on each run's training rows (feature_maps.py --fit train).
# Reused, not re-run: the 24 variables and FC (baselines), CRC and the full QRC feature vectors (benchmark).
#   classical maps (feature_maps.py) of the 12 features: poly2i (78), poly2 (90), poly3 (454),
#     rff1 (390 random Fourier features, median bandwidth), tanh (390); pca (11 PCs of the 24 variables);
#     rff05 / rff2 (bandwidth x 0.5 / x 2, descriptive) on CARDS_2500 and CARDS_1000 only
#   reservoir subsets of QRC-hw, QRC-shot (240) and QRC-Aquila: single readouts t1..t5, cumulative t12,
#     t123, t1234, <Z> only (z12345) and <Z> at 2.5 us (z5); detuning on every dataset (Aquila: CARDS_2500
#     and CARDS_1000), position on CARDS_1000
# Logs: results/ablation/logs/c<set>_<pipeline>[-<subset>][_<model>]_<seed>.log; a run is skipped once
# its predictions exist. SEEDS (CARDS_2500/1000), S30000 / SETS / NPROC restrict the sweep (pilot: S30000=8 SETS=30000).
cd "$(dirname "$0")/../.."
# Run inside the qrc_class environment (docs/installation.md).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SEEDS=${SEEDS:-$(seq 1 10)} S30000=${S30000:-$(grep -v '^#' configs/cards30000_seeds.txt)} SETS=${SETS:-30000 2500 1000} NPROC=${NPROC:-24}
E=results/emulation D=dataset_aquila A=datasets F=results/baselines/features L=results/ablation/logs
DET=$E/detuning_encoding_qpu_samples_12_delta_7.6_deltamax_-9.0 DETE=$E/detuning_encoding_emulation_samples_12_delta_7.6_deltamax_-9.0
POS=$E/position_encoding_qpu_samples_pos_12_delta_2.0_lambda_1.0
DETAQ=$A/detuning_encoding_qpu_samples_240shots_real_aquila.csv
POSAQ=$A/position_encoding_qpu_samples_240shots_real_aquila_lambda1.0_delta2.0.csv
SUBSETS="t1 t2 t3 t4 t5 t12 t123 t1234 z12345 z5"
CLASSICAL="poly2i:poly:2i poly2:poly:2 poly3:poly:3 rff1:rff:1 tanh:tanh pca:pca"
SENSITIVITY="rff05:rff:0.5 rff2:rff:2"
MODELS="KNN SGD NBT LM P LSVM"
mkdir -p $L/fig
has() { [[ " $SETS " == *" $1 "* ]]; }
{
  # CARDS_30000: one process per model, LSVM first (liblinear is single-core and the slowest)
  if has 30000; then
    for m in LSVM SGD KNN LM P NBT; do
      for s in $S30000; do
        for c in $CLASSICAL; do
          name=${c%%:*} map=${c#*:}
          case $name in pca|rff*|tanh) fit=" --fit train" ;; *) fit= ;; esac
          if [ $name = pca ]; then feat="$F/qpu_samples_24.csv $F/emulation_samples_24.csv"; else feat="$D/qpu_samples.csv $D/emulation_samples.csv"; fi
          echo "c30000_${name}_${m}_$s|$map$fit|$feat|$D/labels_qpu.csv $D/labels_emu.csv|$m|$s"
        done
        for sub in $SUBSETS; do
          echo "c30000_dethw-${sub}_${m}_$s|qrc:$sub|${DET}_hw.csv ${DETE}_hw.csv|$D/labels_qpu.csv $D/labels_emu.csv|$m|$s"
          echo "c30000_detshot-${sub}_${m}_$s|qrc:$sub|${DET}_hwshots_240.csv ${DETE}_hwshots_240.csv|$D/labels_qpu.csv $D/labels_emu.csv|$m|$s"
        done
      done
    done
  fi
  for s in $SEEDS; do
    for set in 2500 1000; do
      has $set || continue
      [ $set = 2500 ] && lab=$D/labels_qpu.csv || lab=$D/labels_qpu_pos.csv
      [ $set = 2500 ] && in12=$D/qpu_samples.csv || in12=$D/qpu_samples_pos.csv
      [ $set = 2500 ] && in24=$F/qpu_samples_24.csv || in24=$F/qpu_samples_pos_24.csv
      for c in $CLASSICAL $SENSITIVITY; do
        name=${c%%:*} map=${c#*:}
        [ $name = pca ] && feat=$in24 || feat=$in12
        echo "c${set}_${name}_$s|$map|$feat|$lab|$MODELS|$s"
      done
      for sub in $SUBSETS; do
        echo "c${set}_dethw-${sub}_$s|qrc:$sub|${DET}_hw.csv|$lab|$MODELS|$s"
        echo "c${set}_detshot-${sub}_$s|qrc:$sub|${DET}_hwshots_240.csv|$lab|$MODELS|$s"
        echo "c${set}_detaquila-${sub}_$s|qrc:$sub|$DETAQ|$lab|$MODELS|$s"
        if [ $set = 1000 ]; then
          echo "c1000_poshw-${sub}_$s|qrc:$sub|${POS}_hw.csv|$lab|$MODELS|$s"
          echo "c1000_posshot-${sub}_$s|qrc:$sub|${POS}_hwshots_240.csv|$lab|$MODELS|$s"
          echo "c1000_posaquila-${sub}_$s|qrc:$sub|$POSAQ|$lab|$MODELS|$s"
        fi
      done
    done
  done
} | xargs -P $NPROC -I{} bash -c 'IFS="|" read -r name map feat lab models seed <<< "{}"
  [ -f results/ablation/logs/$name.predictions.csv.gz ] || python analysis/ablation/feature_maps.py --map $map -- \
    --features $feat --labels $lab --split random --classifiers $models --seed $seed \
    --run-name $name --log-dir results/ablation/logs --figure-dir results/ablation/logs/fig --n-jobs 1 \
    > /dev/null 2>&1 || echo "FAILED $name"
  echo "finished $name"'
echo "done $(ls $L/*.predictions.csv.gz | wc -l) runs"
