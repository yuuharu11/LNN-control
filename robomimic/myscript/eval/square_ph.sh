#!/bin/bash

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/square/ph/low_dim_v15.hdf5"
N_ROLLOUTS=100
HORIZON=2000
SEED=0
CSV_PATH="/work/robomimic/csv/eval/square-ph.csv"

# name と dataset_path の対応を associative array で定義
declare -A models=(
  # square task models
  ["bc-ph"]="robomimic/bc_trained_models/square/bc/ph/models/model_epoch_150_low_dim_v15_success_0.28.pth"
  ["bc-rnn-ph"]="/work/robomimic/bc_trained_models/square/bc_rnn/ph/models/model_epoch_150_low_dim_v15_success_0.08.pth"
  ["bcq-ph"]="/work/robomimic/bc_trained_models/square/bcq/ph/models/model_epoch_200_low_dim_v15_success_0.04.pth"
  ["cql-ph"]="/work/robomimic/bc_trained_models/square/cql/ph/models/model_epoch_50_low_dim_v15_success_0.0.pth"
  ["hbc-ph"]="/work/robomimic/bc_trained_models/square/hbc/ph/models/model_epoch_50_low_dim_v15_success_0.3.pth"
  ["iris-ph"]="/work/robomimic/bc_trained_models/square/iris/ph/models/model_epoch_50_low_dim_v15_success_0.24.pth"
  ["ncp-ph_u64_seed1"]="/work/robomimic/bc_trained_models/square/ncp/ph/unit64/seed1/models/model_epoch_150_low_dim_v15_success_0.02.pth"
  ["ncp-ph_u64_seed2"]="/work/robomimic/bc_trained_models/square/ncp/ph/unit64/seed2/models/model_epoch_200_low_dim_v15_success_0.08.pth"
  ["ncp-ph_u64_seed3"]="/work/robomimic/bc_trained_models/square/ncp/ph/unit64/seed3/models/model_epoch_150_low_dim_v15_success_0.02.pth"
  ["ncp-ph_u128_seed1"]="//work/robomimic/bc_trained_models/square/ncp/ph/unit128/seed1/models/model_epoch_150_low_dim_v15_success_0.1.pth"
  ["ncp-ph_u128_seed2"]="/work/robomimic/bc_trained_models/square/ncp/ph/unit128/seed2/models/model_epoch_200_low_dim_v15_success_0.1.pth"
  ["ncp-ph_u128_seed3"]="/work/robomimic/bc_trained_models/square/ncp/ph/unit128/seed3/models/model_epoch_150_low_dim_v15_success_0.16.pth"
  ["ncp-ph_u256_seed1"]="/work/robomimic/bc_trained_models/square/ncp/ph/unit256/seed1/models/model_epoch_150_low_dim_v15_success_0.14.pth"
  ["ncp-ph_u256_seed2"]="/work/robomimic/bc_trained_models/square/ncp/ph/unit256/seed2/models/model_epoch_150_low_dim_v15_success_0.14.pth"
  ["ncp-ph_u256_seed3"]="/work/robomimic/bc_trained_models/square/ncp/ph/unit256/seed3/models/model_epoch_150_low_dim_v15_success_0.16.pth"
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
