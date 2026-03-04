#!/bin/bash
set -euo pipefail

# ===== 共通設定（必要に応じて変更）=====
MODEL_ROOT="/work/robomimic/trained_models/LNN_standardization"
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15_11.hdf5"
OUT_ROOT="/work/robomimic/csv/result/quantize/LNN_standardization"
LOG_ROOT="/work/robomimic/logs/quantize/best/calibration/LNN_standardization"

N_ROLLOUTS=100
HORIZON=400
CALIBRATION_TIMES=3
CALIBRATION_PERCENTILE=99.9

QUANTIZES=(2 3 4 5 6 7 8)

mkdir -p "$OUT_ROOT/CAM" "$LOG_ROOT"

shopt -s nullglob globstar
models=(
  "$MODEL_ROOT"/u256/*.pth
)

if [[ ${#models[@]} -eq 0 ]]; then
  echo "[ERROR] model not found: $MODEL_ROOT"
  exit 1
fi

seed_idx=1
for model_path in "${models[@]}"; do
  [[ -f "$model_path" ]] || continue

  base_name=$(basename "$model_path")
  prefix_num="${base_name%%_*}"
  
  name="seed${prefix_num}"
  # 先頭が数字でないファイルはスキップ
  if [[ ! "$prefix_num" =~ ^[0-9]+$ ]]; then
    echo "[SKIP] invalid prefix: $base_name"
    continue
  fi

  # 10進数として比較（先頭0対策: 10#）
  if (( 10#$prefix_num <= 10 )); then
    echo "[SKIP] prefix <= 10: $base_name"
    continue
  fi
  calib_path="${LOG_ROOT}/Seed${seed_idx}.json"
  name="seed${prefix_num}"
  echo "=========================================="
  echo "Running model: ${name}"
  echo "model path: ${model_path}"
  echo "=========================================="

  for quantize_bit in "${QUANTIZES[@]}"; do
    csv_path="${OUT_ROOT}/CAM/${quantize_bit}bit.csv"

    echo "[RUN] ${name} / quantize=${quantize_bit}"

    python /work/robomimic/robomimic/scripts/run_trained_agent.py \
      --agent "$model_path" \
      --name "$name" \
      --n_rollouts "$N_ROLLOUTS" \
      --horizon "$HORIZON" \
      --seed "$seed_idx" \
      --dataset_path "$DATASET_PATH" \
      --digital_SRAM_quantization 8 \
      --digital_RRAM_quantization 8 \
      --weight_quantization 6 \
      --LUT_quantization 5 \
      --CAM_quantization 6 \
      --ADC_quantization 5 \
      --DAC_quantization 8 \
      --CAM_quantization "$quantize_bit" \
      --calibration_times "$CALIBRATION_TIMES" \
      --calibration_path "$calib_path" \
      --calibration_percentile "$CALIBRATION_PERCENTILE" \
      --csv_path "$csv_path"

    echo "[DONE] ${name} / quantize=${quantize_bit}"
    echo "------------------------------------------"
  done

  seed_idx=$((seed_idx + 1))
done

echo "=========================================="
echo "All experiments completed."
echo "CSV: $OUT_ROOT/CAM"
echo "=========================================="