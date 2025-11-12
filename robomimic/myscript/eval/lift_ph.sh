#!/bin/bash

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/can/ph/low_dim_v15.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=0
CSV_PATH="/work/robomimic/csv/eval/can-ph.csv"

# name と dataset_path の対応を associative array で定義
declare -A models=(
  # lift task models
  #["lnn_seed1"]="/work/robomimic/bc_trained_models/lift/lnn/seed1_ph/models/model_epoch_200_low_dim_v15_success_0.98.pth"
  #["lnn_seed2"]="/work/robomimic/bc_trained_models/lift/lnn/seed2_ph/models/model_epoch_100_low_dim_v15_success_0.96.pth"
  #["lnn-lstm_seed1"]="/work/robomimic/bc_trained_models/lift/lnn_lstm/seed1_ph/models/model_epoch_350_low_dim_v15_success_0.98.pth"
  #["lnn-lstm_u128"]="/work/robomimic/bc_trained_models/lift/lnn_lstm/units128/ph/models/model_epoch_50_low_dim_v15_success_1.0.pth"
  #["lnn-cfc_seed1"]="/work/robomimic/bc_trained_models/lift/lnn_cfc/seed1_ph/models/model_epoch_100_low_dim_v15_success_0.96.pth"
  #["lnn-cfc_seed2"]="/work/robomimic/bc_trained_models/lift/lnn_cfc/seed2_ph/models/model_epoch_400_low_dim_v15_success_0.94.pth"
  #["lnn-cfc_seed3"]="/work/robomimic/bc_trained_models/lift/lnn_cfc/seed3_ph/models/model_epoch_100_low_dim_v15_success_0.74.pth"
  #["lnn-cfc_seed4"]="/work/robomimic/bc_trained_models/lift/lnn_cfc/seed4_ph/models/model_epoch_50_low_dim_v15_success_0.84.pth"
  #["bc"]="/work/robomimic/bc_trained_models/lift/core/bc/ph/models/model_epoch_600_low_dim_v15_success_1.0.pth"
  #["bc-rnn"]="/work/robomimic/bc_trained_models/lift/core/bc_rnn/ph/models/model_epoch_250_low_dim_v15_success_1.0.pth"
  #["bcq"]="/work/robomimic/bc_trained_models/lift/core/bcq/ph/models/model_epoch_800_low_dim_v15_success_1.0.pth"
  #["cql"]="/work/robomimic/bc_trained_models/lift/core/cql/ph/models/model_epoch_250_low_dim_v15_success_0.76.pth"
  #["hbc"]=""
  #["iris"]=""

  # can task models
  #["bc-ph"]="/work/robomimic/bc_trained_models/can/bc/ph/models/model_epoch_100_low_dim_v15_success_0.58.pth"
  #["bc-mh"]="/work/robomimic/bc_trained_models/can/bc/mh/models/model_epoch_150_low_dim_v15_success_0.28.pth"
  #["bc-rnn-ph"]="/work/robomimic/bc_trained_models/can/bc_rnn/ph/models/model_epoch_200_low_dim_v15_success_1.0.pth"
  #["bc-rnn-mh"]="/work/robomimic/bc_trained_models/can/bc_rnn/mh/models/model_epoch_100_low_dim_v15_success_0.74.pth"
  #["bcq-ph"]="/work/robomimic/bc_trained_models/can/bcq/ph/models/model_epoch_150_low_dim_v15_success_0.28.pth"
  #["bcq-mh"]="//work/robomimic/bc_trained_models/can/bcq/mh/models/model_epoch_150_low_dim_v15_success_0.14.pth"
  #["cql-ph"]=
  #["cql-mh"]=
  #["hbc-ph"]="/work/robomimic/bc_trained_models/can/hbc/ph/models/model_epoch_50_low_dim_v15_success_0.84.pth"
  #["hbc-mh"]="/work/robomimic/bc_trained_models/can/hbc/mh/models/model_epoch_100_low_dim_v15_success_0.56.pth"
  ["iris-ph"]="/work/robomimic/bc_trained_models/can/iris/ph/models/model_epoch_50_low_dim_v15_success_0.82.pth"
  #["iris-mh"]="/work/robomimic/bc_trained_models/can/iris/mh/models/model_epoch_50_low_dim_v15_success_0.82.pth"
  #["ncp-ph_u64_seed1"]="/work/robomimic/bc_trained_models/can/ncp/ph/unit64/seed1/models/model_epoch_150_low_dim_v15_success_0.54.pth"
  #["ncp-ph_u64_seed2"]="/work/robomimic/bc_trained_models/can/ncp/ph/unit64/seed2/models/model_epoch_100_low_dim_v15_success_0.66.pth"
  #["ncp-ph_u64_seed3"]="/work/robomimic/bc_trained_models/can/ncp/ph/unit64/seed3/models/model_epoch_150_low_dim_v15_success_0.62.pth"
  #["ncp-ph_u128_seed1"]="/work/robomimic/bc_trained_models/can/ncp/ph/unit128/seed1/models/model_epoch_200_low_dim_v15_success_0.6.pth"
  #["ncp-ph_u128_seed2"]="/work/robomimic/bc_trained_models/can/ncp/ph/unit128/seed2/models/model_epoch_200_low_dim_v15_success_0.72.pth"
  #["ncp-ph_u128_seed3"]="/work/robomimic/bc_trained_models/can/ncp/ph/unit128/seed3/models/model_epoch_150_low_dim_v15_success_0.76.pth"
  #["ncp-ph_u256_seed1"]="/work/robomimic/bc_trained_models/can/ncp/ph/unit256/seed1/models/model_epoch_200_low_dim_v15_success_0.86.pth"
  #["ncp-ph_u256_seed2"]="/work/robomimic/bc_trained_models/can/ncp/ph/unit256/seed2/models/model_epoch_100_low_dim_v15_success_0.96.pth"
  #["ncp-ph_u256_seed3"]="/work/robomimic/bc_trained_models/can/ncp/ph/unit256/seed3/models/model_epoch_150_low_dim_v15_success_0.94.pth"
  
  #["ncp-mh_u64_seed1"]="/work/robomimic/bc_trained_models/can/ncp/mh/unit64/seed1/models/model_epoch_100_low_dim_v15_success_0.24.pth"
  #["ncp-mh_u64_seed2"]="/work/robomimic/bc_trained_models/can/ncp/mh/unit64/seed2/models/model_epoch_100_low_dim_v15_success_0.26.pth"
  #["ncp-mh_u64_seed3"]="/work/robomimic/bc_trained_models/can/ncp/mh/unit64/seed3/models/model_epoch_200_low_dim_v15_success_0.42.pth"
  #["ncp-mh_u128_seed1"]="/work/robomimic/bc_trained_models/can/ncp/mh/unit128/seed1/models/model_epoch_100_low_dim_v15_success_0.32.pth"
  #["ncp-mh_u128_seed2"]="/work/robomimic/bc_trained_models/can/ncp/mh/unit128/seed2/models/model_epoch_100_low_dim_v15_success_0.58.pth"
  #["ncp-mh_u128_seed3"]="/work/robomimic/bc_trained_models/can/ncp/mh/unit128/seed3/models/model_epoch_200_low_dim_v15_success_0.44.pth"
  #["ncp-mh_u256_seed1"]="/work/robomimic/bc_trained_models/can/ncp/mh/unit256/seed1/models/model_epoch_100_low_dim_v15_success_0.54.pth"
  #["ncp-mh_u256_seed2"]="/work/robomimic/bc_trained_models/can/ncp/mh/unit256/seed2/models/model_epoch_150_low_dim_v15_success_0.56.pth"
  #["ncp-mh_u256_seed3"]="/work/robomimic/bc_trained_models/can/ncp/mh/unit256/seed3/models/model_epoch_100_low_dim_v15_success_0.6.pth"
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
