#!/bin/bash
set -euo pipefail

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15_5.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=0
CSV_BASE="/work/robomimic/csv/result/baseline/LNN_standardization/lift/unit256"
mkdir -p ${CSV_BASE}
MODEL_DIR="/work/robomimic/trained_models/LNN/u256"
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

    name="u256_${filename}"
    units="unit256"
    echo "Running inference for ${name}..."
    python /work/robomimic/robomimic/scripts/run_trained_agent.py \
      --agent "$model_path" \
      --n_rollouts "$N_ROLLOUTS" \
      --horizon "$HORIZON" \
      --seed "$SEED" \
      --dataset_path "$DATASET_PATH" \
      --name "$name" \
      --csv_path "$CSV_BASE/${units}.csv"
    echo "Completed: ${name}"
    echo "----------------------------------------"
  fi
done
echo "=========================================="
echo "All experiments completed!"
echo "Results saved in ${CSV_BASE}"
echo "=========================================="