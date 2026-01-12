#!/bin/bash

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15_3.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=0
CSV_BASE="/work/robomimic/result/rnn"
mkdir -p $(dirname ${CSV_BASE})
MODEL_DIR="/work/robomimic/trained_models/lift/rnn"
for model_path in ${MODEL_DIR}/model_epoch_*_low_dim_v15_success_*; do
  if [[ -f "$model_path" ]]; then
    # ファイル名からseed番号を抽出
    filename=$(basename "$model_path")
    seed=$(echo "$filename" | grep -o 'seed[0-9]\+' | grep -o '[0-9]\+')
    name="u64_${filename}"
    units="unit64"
    echo "Running inference for ${name}..."
    python /work/robomimic/robomimic/scripts/run_trained_agent.py \
      --agent "$model_path" \
      --n_rollouts "$N_ROLLOUTS" \
      --horizon "$HORIZON" \
      --seed "$SEED" \
      --dataset_path "$DATASET_PATH" \
      --name "$name" \
      --csv_path "$CSV_BASE.csv"
    echo "Completed: ${name}"
    echo "----------------------------------------"
  fi
done
echo "=========================================="
echo "All experiments completed!"
echo "Results saved in ${CSV_BASE}"
echo "=========================================="