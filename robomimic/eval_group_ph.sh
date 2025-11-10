#!/bin/bash

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/lift_tmp/ph/low_dim_v15.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=0
CSV_PATH="/work/robomimic/csv/eval/ph_tmp.csv"

# name と dataset_path の対応を associative array で定義
declare -A models=(
  ["lnn_seed1"]="/work/robomimic/bc_trained_models/lift/lnn/seed1_ph/models/model_epoch_200_low_dim_v15_success_0.98.pth"
  ["lnn_seed2"]="/work/robomimic/bc_trained_models/lift/lnn/seed2_ph/models/model_epoch_100_low_dim_v15_success_0.96.pth"
  ["lnn-lstm_seed1"]="/work/robomimic/bc_trained_models/lift/lnn_lstm/seed1_ph/models/model_epoch_350_low_dim_v15_success_0.98.pth"
  ["lnn-lstm_u128"]="/work/robomimic/bc_trained_models/lift/lnn_lstm/units128/ph/models/model_epoch_50_low_dim_v15_success_1.0.pth"
  ["lnn-cfc_seed1"]="/work/robomimic/bc_trained_models/lift/lnn_cfc/seed1_ph/models/model_epoch_100_low_dim_v15_success_0.96.pth"
  ["lnn-cfc_seed2"]="/work/robomimic/bc_trained_models/lift/lnn_cfc/seed2_ph/models/model_epoch_400_low_dim_v15_success_0.94.pth"
  ["lnn-cfc_seed3"]="/work/robomimic/bc_trained_models/lift/lnn_cfc/seed3_ph/models/model_epoch_100_low_dim_v15_success_0.74.pth"
  ["lnn-cfc_seed4"]="/work/robomimic/bc_trained_models/lift/lnn_cfc/seed4_ph/models/model_epoch_50_low_dim_v15_success_0.84.pth"
  ["bc"]="/work/robomimic/bc_trained_models/lift/core/bc/ph/models/model_epoch_600_low_dim_v15_success_1.0.pth"
  ["bc-rnn"]="/work/robomimic/bc_trained_models/lift/core/bc_rnn/ph/models/model_epoch_250_low_dim_v15_success_1.0.pth"
  ["bcq"]="/work/robomimic/bc_trained_models/lift/core/bcq/ph/models/model_epoch_800_low_dim_v15_success_1.0.pth"
  ["cql"]="/work/robomimic/bc_trained_models/lift/core/cql/ph/models/model_epoch_250_low_dim_v15_success_0.76.pth"
  # ["hbc"]=""
  # ["iris"]=""
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
        --csv_path "${CSV_PATH}"

    echo "Completed: ${name}"
    echo "----------------------------------------"
done
