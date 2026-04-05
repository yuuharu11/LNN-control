#!/bin/bash
set -euo pipefail

shopt -s nullglob

# --- Environment / working directory setup ---
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

# --- Sweep settings ---
MODEL_DIR="/work/robomimic/trained_models/LNN/u256"
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15_10.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=3

CALIBRATION_PERCENTILES=(100.0 99.99 99.9)

# ★ 指定どおり
CALIBRATION_TIMES_LIST=(1 3 5)

MAX_MODEL_COUNT=10
TARGET_MODEL_SEED=""

CSV_BASE="/work/robomimic/csv/result/calibration_sweep"
LOG_BASE="/work/robomimic/logs/quantize/calibration/LNN/sweep"

# Ensure base output directories exist before writing files
mkdir -p "${CSV_BASE}" "${LOG_BASE}"

# ★ 平均のみ
SUMMARY_CSV="${CSV_BASE}/summary.csv"
echo "percentile,calibration_times,mean_success_rate" > "${SUMMARY_CSV}"

index_tmp="$(mktemp)"
trap 'rm -f "${index_tmp}"' EXIT

# --- モデル収集 ---
for model_path in "${MODEL_DIR}"/*_model_epoch_*_low_dim_v15_success_*; do
  [[ -f "${model_path}" ]] || continue

  filename="$(basename "${model_path}")"
  prefix_num="${filename%%_*}"

  model_seed=""
  if [[ "${filename}" =~ seed([0-9]+) ]]; then
    model_seed="${BASH_REMATCH[1]}"
  elif [[ "${prefix_num}" =~ ^[0-9]+$ ]]; then
    model_seed="$((10#${prefix_num}))"
  else
    echo "[SKIP] seed could not be parsed: ${filename}"
    continue
  fi

  if [[ -n "${TARGET_MODEL_SEED}" && "${model_seed}" != "${TARGET_MODEL_SEED}" ]]; then
    continue
  fi

  echo "${model_seed}|${model_path}" >> "${index_tmp}"
done

if [[ ! -s "${index_tmp}" ]]; then
  echo "No model matched TARGET_MODEL_SEED=${TARGET_MODEL_SEED}" >&2
  exit 1
fi

mapfile -t sorted_models < <(sort -t'|' -k1,1n "${index_tmp}")

selected_model_paths=()
selected_model_seeds=()

for row in "${sorted_models[@]}"; do
  if [[ -z "${TARGET_MODEL_SEED}" && ${#selected_model_paths[@]} -ge ${MAX_MODEL_COUNT} ]]; then
    break
  fi
  selected_model_seeds+=("${row%%|*}")
  selected_model_paths+=("${row#*|}")
done

if [[ ${#selected_model_paths[@]} -eq 0 ]]; then
  echo "No model selected." >&2
  exit 1
fi

selected_seed_str="$(IFS=';'; echo "${selected_model_seeds[*]}")"
echo "Selected model seeds: ${selected_seed_str}"

# --- sweep ---
for percentile in "${CALIBRATION_PERCENTILES[@]}"; do
  percentile_tag="${percentile//./p}"

  csv_dir="${CSV_BASE}/pct_${percentile_tag}"
  log_dir="${LOG_BASE}/pct_${percentile_tag}"
  mkdir -p "${csv_dir}" "${log_dir}"

  for calib_times in "${CALIBRATION_TIMES_LIST[@]}"; do
    success_tmp="$(mktemp)"

    for idx in "${!selected_model_paths[@]}"; do
      model_path="${selected_model_paths[$idx]}"
      model_seed="${selected_model_seeds[$idx]}"

      csv_path="${csv_dir}/seed${model_seed}.csv"
      run_name="u256_seed${model_seed}_ct${calib_times}_p${percentile}"
      calib_path="${log_dir}/Seed${model_seed}_ct${calib_times}.json"

      echo "Running ${run_name}"

      python /work/robomimic/robomimic/scripts/run_trained_agent.py \
        --agent "${model_path}" \
        --n_rollouts "${N_ROLLOUTS}" \
        --horizon "${HORIZON}" \
        --seed "${SEED}" \
        --dataset_path "${DATASET_PATH}" \
        --name "${run_name}" \
        --csv_path "${csv_path}" \
        --calibration_times "${calib_times}" \
        --calibration_path "${calib_path}" \
        --calibration_percentile "${percentile}" \
        --digital_SRAM_quantization 8 \
        --digital_RRAM_quantization 8 \
        --weight_quantization 6 \
        --LUT_quantization 6 \
        --CAM_quantization 6 \
        --ADC_quantization 8 \
        --DAC_quantization 6 \
        --gaussian 0.0 \
        --cell_bits 3

      # --- success_rate 抽出 ---
      success_rate="$(python - "${csv_path}" "${run_name}" <<'PY'
import csv, sys
csv_path, run_name = sys.argv[1], sys.argv[2]

value = ""
with open(csv_path, newline="") as f:
    for row in csv.DictReader(f):
        if row.get("name") == run_name:
            value = row.get("success_rate", "")
print(value if value else "NaN")
PY
)"
      echo "${success_rate}" >> "${success_tmp}"

      echo "Finished ${run_name}"
      echo "------------------------------------------"
    done

    # --- 平均のみ ---
    mean_success="$(python - "${success_tmp}" <<'PY'
import sys, statistics

vals = []
with open(sys.argv[1]) as f:
    for line in f:
        t = line.strip()
        if t and t.lower() != "nan":
            vals.append(float(t))

print(f"{statistics.mean(vals):.6f}" if vals else "NaN")
PY
)"

    rm -f "${success_tmp}"

    echo "${percentile},${calib_times},${mean_success}" >> "${SUMMARY_CSV}"
  done
done

echo "=========================================="
echo "Sweep completed"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "=========================================="