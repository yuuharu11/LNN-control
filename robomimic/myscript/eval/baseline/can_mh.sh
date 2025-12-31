#!/bin/bash

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/can/ph/low_dim_v15.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=0
CSV_PATH="/work/robomimic/csv/eval/can-ph.csv"

# name と dataset_path の対応を associative array で定義
declare -A models=(
  # can task models
  ["bc-mh"]="/work/robomimic/bc_trained_models/can/bc/mh/models/model_epoch_150_low_dim_v15_success_0.28.pth"
  ["bc-rnn-mh"]="/work/robomimic/bc_trained_models/can/bc_rnn/mh/models/model_epoch_100_low_dim_v15_success_0.74.pth"
  ["bcq-mh"]="/work/robomimic/bc_trained_models/can/bcq/mh/models/model_epoch_150_low_dim_v15_success_0.14.pth"
  ["cql-mh"]=
  ["hbc-mh"]="/work/robomimic/bc_trained_models/can/hbc/mh/models/model_epoch_100_low_dim_v15_success_0.56.pth"
  ["iris-mh"]="/work/robomimic/bc_trained_models/can/iris/mh/models/model_epoch_50_low_dim_v15_success_0.82.pth"
  ["ncp-mh_u64_seed1"]="/work/robomimic/bc_trained_models/can/ncp/mh/unit64/seed1/models/model_epoch_100_low_dim_v15_success_0.24.pth"
  ["ncp-mh_u64_seed2"]="/work/robomimic/bc_trained_models/can/ncp/mh/unit64/seed2/models/model_epoch_100_low_dim_v15_success_0.26.pth"
  ["ncp-mh_u64_seed3"]="/work/robomimic/bc_trained_models/can/ncp/mh/unit64/seed3/models/model_epoch_200_low_dim_v15_success_0.42.pth"
  ["ncp-mh_u128_seed1"]="/work/robomimic/bc_trained_models/can/ncp/mh/unit128/seed1/models/model_epoch_100_low_dim_v15_success_0.32.pth"
  ["ncp-mh_u128_seed2"]="/work/robomimic/bc_trained_models/can/ncp/mh/unit128/seed2/models/model_epoch_100_low_dim_v15_success_0.58.pth"
  ["ncp-mh_u128_seed3"]="/work/robomimic/bc_trained_models/can/ncp/mh/unit128/seed3/models/model_epoch_200_low_dim_v15_success_0.44.pth"
  ["ncp-mh_u256_seed1"]="/work/robomimic/bc_trained_models/can/ncp/mh/unit256/seed1/models/model_epoch_100_low_dim_v15_success_0.54.pth"
  ["ncp-mh_u256_seed2"]="/work/robomimic/bc_trained_models/can/ncp/mh/unit256/seed2/models/model_epoch_150_low_dim_v15_success_0.56.pth"
  ["ncp-mh_u256_seed3"]="/work/robomimic/bc_trained_models/can/ncp/mh/unit256/seed3/models/model_epoch_100_low_dim_v15_success_0.6.pth"
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
