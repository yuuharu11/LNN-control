#!/bin/bash

# --- Environment / working directory setup ---
# Enable conda in non-interactive shells and activate the expected env.
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate robomimic_venv
  else
    echo "conda is available, but conda.sh was not found (base=${CONDA_BASE})." >&2
    exit 1
  fi
else
  echo "conda command not found. Please install/enable conda before running." >&2
  exit 1
fi

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15_8.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=0
gaussian=(0.0)
CSV_BASE="/work/robomimic/csv/result/error/LNN_standardization/6-6-6/tmp"
LOG_PATH="/work/robomimic/logs/quantize/best/calibration/LNN/u256"
mkdir -p ${CSV_BASE}
MODEL_DIR="/work/robomimic/trained_models/LNN/u256"
for model_path in ${MODEL_DIR}/*_model_epoch_*_low_dim_v15_success_*; do
  if [[ -f "$model_path" ]]; then
    filename=$(basename "$model_path")
    base_name="$filename"
    prefix_num="${base_name%%_*}"

    seed=""
    if [[ "$filename" =~ seed([0-9]+) ]]; then
      seed="${BASH_REMATCH[1]}"
    elif [[ "$prefix_num" =~ ^[0-9]+$ ]]; then
      seed="$((10#$prefix_num))"
    else
      echo "[SKIP] seed could not be parsed: $base_name"
      continue
    fi
  fi

  for g in "${gaussian[@]}"; do
    if [[ -f "$model_path" ]]; then
      name="u256_${seed}"
      units="unit256"
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
        --weight_quantization 6 \
        --LUT_quantization 6 \
        --CAM_quantization 6 \
        --ADC_quantization 8 \
        --DAC_quantization 6 \
        --gaussian "${g}" \
        --cell_bits 3 \
        --csv_path "${CSV_BASE}/gaussian${g}.csv" 
      echo "Completed: ${name}"
      echo "----------------------------------------"
    fi
  done
done
echo "=========================================="
echo "All experiments completed!"
echo "Results saved in ${CSV_BASE}"
echo "=========================================="