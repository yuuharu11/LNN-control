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
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15_15.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=0
shift=(0.09 0.10)
CSV_BASE="/work/robomimic/csv/result/error/LNN/pure/shift/u64"
LOG_PATH="/work/robomimic/logs/quantize/calibration/old/u64"
mkdir -p ${CSV_BASE}
MODEL_DIR="/work/robomimic/trained_models/old/lift/u64"
for model_path in ${MODEL_DIR}/*_model_epoch_*_low_dim_v15_success_*; do
  if [[ -f "$model_path" ]]; then
    # ファイル名からseed番号を抽出
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

  for s in "${shift[@]}"; do
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
        --name "${name}_shift${s}" \
        --shift "${s}" \
        --csv_path "${CSV_BASE}/shift${s}.csv" 
      echo "Completed: ${name}"
      echo "----------------------------------------"
    fi
  done
done
echo "=========================================="
echo "All experiments completed!"
echo "Results saved in ${CSV_BASE}"
echo "=========================================="