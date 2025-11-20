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
  #["bc-ph"]="/work/robomimic/bc_trained_models/can/bc/ph/models/model_epoch_100_low_dim_v15_success_0.58.pth"
  #["bc-rnn-ph"]="/work/robomimic/bc_trained_models/can/bc_rnn/ph/models/model_epoch_200_low_dim_v15_success_1.0.pth"
  #["bcq-ph"]="/work/robomimic/bc_trained_models/can/bcq/ph/models/model_epoch_150_low_dim_v15_success_0.28.pth"
  ["cql-ph"]=
  #["hbc-ph"]="/work/robomimic/bc_trained_models/can/hbc/ph/models/model_epoch_50_low_dim_v15_success_0.84.pth"
  #["iris-ph"]="/work/robomimic/bc_trained_models/can/iris/ph/models/model_epoch_50_low_dim_v15_success_0.82.pth"
  #["ncp-ph_u64_seed1"]="/work/robomimic/bc_trained_models/can/ncp/ph/unit64/seed1/models/model_epoch_150_low_dim_v15_success_0.54.pth"
  #["ncp-ph_u64_seed2"]="/work/robomimic/bc_trained_models/can/ncp/ph/unit64/seed2/models/model_epoch_100_low_dim_v15_success_0.66.pth"
  #["ncp-ph_u64_seed3"]="/work/robomimic/bc_trained_models/can/ncp/ph/unit64/seed3/models/model_epoch_150_low_dim_v15_success_0.62.pth"
  #["ncp-ph_u128_seed1"]="/work/robomimic/bc_trained_models/can/ncp/ph/unit128/seed1/models/model_epoch_200_low_dim_v15_success_0.6.pth"
  #["ncp-ph_u128_seed2"]="/work/robomimic/bc_trained_models/can/ncp/ph/unit128/seed2/models/model_epoch_200_low_dim_v15_success_0.72.pth"
  #["ncp-ph_u128_seed3"]="/work/robomimic/bc_trained_models/can/ncp/ph/unit128/seed3/models/model_epoch_150_low_dim_v15_success_0.76.pth"
  #["ncp-ph_u256_seed1"]="/work/robomimic/bc_trained_models/can/ncp/ph/unit256/seed1/models/model_epoch_200_low_dim_v15_success_0.86.pth"
  #["ncp-ph_u256_seed2"]="/work/robomimic/bc_trained_models/can/ncp/ph/unit256/seed2/models/model_epoch_100_low_dim_v15_success_0.96.pth"
  #["ncp-ph_u256_seed3"]="/work/robomimic/bc_trained_models/can/ncp/ph/unit256/seed3/models/model_epoch_150_low_dim_v15_success_0.94.pth"
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
        --record_lnn_states False

    echo "Completed: ${name}"
    echo "----------------------------------------"
done
