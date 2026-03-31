#!/bin/bash
set -euo pipefail

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15_15_8.hdf5"
N_ROLLOUTS=10
HORIZON=400
SEED=0
model_path="/work/robomimic/trained_models/LNN/u256/01_model_epoch_400_low_dim_v15_success_1.0.pth"
name="test"
LOG_PATH="/work/robomimic/logs/quantize/best/calibration/LNN/u256"

echo "Running inference for ${name}..."
python /work/robomimic/robomimic/scripts/run_trained_agent.py \
  --agent "$model_path" \
  --n_rollouts "$N_ROLLOUTS" \
  --horizon "$HORIZON" \
  --seed "$SEED" \
  --dataset_path "$DATASET_PATH" \
  --calibration_times 3 \
  --calibration_path "$LOG_PATH/Seed7.json" \
  --calibration_percentile 99.9 \
  --digital_SRAM_quantization 8 \
  --digital_RRAM_quantization 8 \
  --weight_quantization 6 \
  --LUT_quantization 6 \
  --CAM_quantization 6 \
  --ADC_quantization 8 \
  --DAC_quantization 6 \
  --name "$name" \
  --gaussian 0.0 \

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="