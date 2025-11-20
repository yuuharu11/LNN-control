#!/bin/bash

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/can/ph/low_dim_v15.hdf5"
N_ROLLOUTS=10
HORIZON=400
SEED=0
CSV_PATH="/work/temp/csv/eval/can-ph.csv"

# name と dataset_path の対応を associative array で定義
declare -A models=(
  # lift task models
    ["lnn"]="/work/robomimic/bc_trained_models/can/ncp/ph/unit256/seed3/models/model_epoch_200.pth"
)

# 各データセットに対して逐次推論を実行
for name in "${!models[@]}"; do
    model_path="${models[$name]}"

    echo "Running inference for ${name}..."
    python /work/robomimic/robomimic/scripts/run_trained_agent.py \
        --agent "${model_path}" \
        --n_rollouts "${N_ROLLOUTS}" \
        --horizon "${HORIZON}" \
        --seed "${SEED}" \
        --dataset_path "${DATASET_PATH}" \
        --name "${name}" \
        --csv_path "${CSV_PATH}" \
        --lnn_record True

    echo "Completed: ${name}"
    echo "----------------------------------------"
done
