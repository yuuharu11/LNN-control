#!/bin/bash

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15_2.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=1
gaussian=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)
CSV_BASE="/work/robomimic/csv/eval/lift/error/proposal/gaussian/"
LOG_PATH="/work/robomimic/logs/quantize/best/calibration/u64"
mkdir -p ${CSV_BASE}
MODEL_DIR="/work/robomimic/trained_models/lift/u64"
seed=0
for model_path in ${MODEL_DIR}/seed*_model_epoch_*_low_dim_v15_success_*; do
  for g in "${gaussian[@]}"; do
    if [[ -f "$model_path" ]]; then
      name="u64_${seed}"
      units="unit64"
      echo "Running inference for ${name}..."
      python /work/robomimic/robomimic/scripts/run_trained_agent.py \
        --agent "$model_path" \
        --n_rollouts "$N_ROLLOUTS" \
        --horizon "$HORIZON" \
        --seed "$SEED" \
        --dataset_path "$DATASET_PATH" \
        --name "${name}_gaussian${g}" \
        --calibration_times 3 \
        --calibration_path "$LOG_PATH/Seed${seed}.json" \
        --calibration_percentile 99.9 \
        --digital_SRAM_quantization 8 \
        --digital_RRAM_quantization 8 \
        --weight_quantization 5 \
        --LUT_quantization 4 \
        --CAM_quantization 5 \
        --ADC_quantization 5 \
        --DAC_quantization 8 \
        --gaussian "${g}" \
        --csv_path "${CSV_BASE}${units}/gaussian${g}.csv" 
      echo "Completed: ${name}"
      echo "----------------------------------------"
    fi
  done
  seed=$((seed + 1))
done
echo "=========================================="
echo "All experiments completed!"
echo "Results saved in ${CSV_BASE}"
echo "=========================================="