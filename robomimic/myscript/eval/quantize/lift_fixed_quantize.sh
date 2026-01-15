#!/bin/bash

# 共通パラメータ
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15_8.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=0

UNITS_LIST=(64 128 256)
QUANTIZES=(2 3 4 5 6 7 8)

for quantize_bit in "${QUANTIZES[@]}"; do
  echo "=========================================="
  echo "Running experiments for quantize bit: ${quantize_bit}"
  echo "=========================================="

  for U in "${UNITS_LIST[@]}"; do
    echo "=========================================="
    echo "Running experiments for unit${U}"
    echo "=========================================="

    CSV_BASE="/work/robomimic/csv/result/lift/quantize/RRAM-memory/unit${U}/"
    mkdir -p ${CSV_BASE}
    MODEL_DIR="/work/robomimic/trained_models/lift/u${U}"
    LOG_PATH="/work/robomimic/logs/quantize/best/calibration/u${U}"
    units="unit${U}"
    seed=1

    for model_path in ${MODEL_DIR}/seed*_model_epoch_*_low_dim_v15_success_*; do
      if [[ -f "$model_path" ]]; then
        filename=$(basename "$model_path")

        name="u${U}_seed${seed}"

        echo "Running inference for ${name}..."

        python /work/robomimic/robomimic/scripts/run_trained_agent.py \
          --agent "$model_path" \
          --n_rollouts "$N_ROLLOUTS" \
          --horizon "$HORIZON" \
          --seed "$SEED" \
          --dataset_path "$DATASET_PATH" \
          --name "$name" \
          --calibration_times 3 \
          --calibration_path "$LOG_PATH/Seed${seed}.json" \
          --calibration_percentile 99.9 \
          --digital_RRAM_quantization "${quantize_bit}" \
          --csv_path "$CSV_BASE/${quantize_bit}bit.csv"

        echo "Completed: ${name}"
        echo "----------------------------------------"
        seed=$((seed + 1))
      fi
    done
  done
done

echo "=========================================="
echo "All experiments completed!"
echo "Results saved in ${CSV_BASE}"
echo "=========================================="
