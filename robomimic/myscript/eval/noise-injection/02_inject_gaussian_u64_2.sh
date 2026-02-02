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
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15_3.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=10
gaussian=(0.05 0.06 0.07)
CSV_BASE="/work/robomimic/csv/result/error/proposal/6bit/gaussian/u64"
LOG_PATH="/work/robomimic/logs/quantize/best/calibration/u64"
mkdir -p ${CSV_BASE}
MODEL_DIR="/work/robomimic/trained_models/lift/u64"
for model_path in ${MODEL_DIR}/seed*_model_epoch_*_low_dim_v15_success_*; do
  seed=$(grep -oP 'seed\K[0-9]+' <<<"$model_path" | head -n 1)

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
        --weight_quantization 6 \
        --LUT_quantization 5 \
        --CAM_quantization 6 \
        --ADC_quantization 8 \
        --DAC_quantization 5 \
        --cell_bits 6 \
        --gaussian "${g}" \
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