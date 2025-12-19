#!/bin/bash

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/can/ph/low_dim_v15.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=0
CSV_PATH="/work/robomimic/csv/eval/can/can-best.csv"

# name と dataset_path の対応を associative array で定義
declare -A models=(
  # can task models
  #["bc-ph"]="/work/robomimic/bc_trained_models/core/bc/can/ph/low_dim/trained_models/core_bc_can_ph_low_dim/models/model_epoch_250_low_dim_v15_success_0.88.pth"
  #["bc-pure_seed1"]="/work/robomimic/bc_trained_models/can/bc-pure/ph/seed1/models/model_epoch_250_low_dim_v15_success_0.74.pth"
  #["bc-pure_seed2"]="/work/robomimic/bc_trained_models/can/bc-pure/ph/seed2/models/model_epoch_400_low_dim_v15_success_0.66.pth"
  #["bc-pure_seed3"]="/work/robomimic/bc_trained_models/can/bc-pure/ph/seed3/models/model_epoch_450_low_dim_v15_success_0.8.pth"
  #["bc-rnn-ph"]="/work/robomimic/bc_trained_models/core/bc_rnn/can/ph/low_dim/trained_models/core_bc_rnn_can_ph_low_dim/models/model_epoch_2000.pth"
  #["bc-rnn-pure_seed1"]="/work/robomimic/bc_trained_models/can/bc_rnn-pure/ph/seed1/models/model_epoch_200_low_dim_v15_success_1.0.pth"
  #["bc-rnn-pure_seed2"]="/work/robomimic/bc_trained_models/can/bc_rnn-pure/ph/seed2/models/model_epoch_150_low_dim_v15_success_1.0.pth"
  #["bc-rnn-pure_seed3"]="/work/robomimic/bc_trained_models/can/bc_rnn-pure/ph/seed3/models/model_epoch_250_low_dim_v15_success_1.0.pth"
  #["bcq-ph"]="/work/robomimic/bc_trained_models/can/bcq/ph/models/model_epoch_150_low_dim_v15_success_0.28.pth"
  #["cql-ph"]=/work/robomimic/bc_trained_models/core/cql/can/ph/low_dim/trained_models/core_cql_can_ph_low_dim/models/model_epoch_1750_low_dim_v15_success_0.18.pth
  #["hbc-ph"]="/work/robomimic/bc_trained_models/can/hbc/ph/models/model_epoch_50_low_dim_v15_success_0.84.pth"
  #["iris-ph"]="/work/robomimic/bc_trained_models/can/iris/ph/models/model_epoch_50_low_dim_v15_success_0.82.pth"
  ["ncp_u64_best_seed1"]="/work/robomimic/bc_trained_models/can/ncp-pure-best/ph/unit64/seed1/models/model_epoch_200_low_dim_v15_success_0.88.pth"
  ["ncp_u64_best_seed2"]="/work/robomimic/bc_trained_models/can/ncp-pure-best/ph/unit64/seed2/models/model_epoch_100_low_dim_v15_success_0.8.pth"
  ["ncp_u64_best_seed3"]="//work/robomimic/bc_trained_models/can/ncp-pure-best/ph/unit64/seed3/models/model_epoch_200_low_dim_v15_success_0.8.pth"
  #["ncp_u128_best_seed1"]="/work/robomimic/bc_trained_models/can/ncp-pure-best/ph/unit128/seed1/models/model_epoch_250_low_dim_v15_success_0.74.pth"
  #["ncp_u128_best_seed2"]="/work/robomimic/bc_trained_models/can/ncp-pure-best/ph/unit128/seed2/models/model_epoch_750_low_dim_v15_success_0.78.pth"
  #["ncp_u128_best_seed3"]="/work/robomimic/bc_trained_models/can/ncp-pure-best/ph/unit128/seed3/models/model_epoch_400_low_dim_v15_success_0.82.pth"
  #["ncp_u256_best_seed1"]="/work/robomimic/bc_trained_models/can/ncp-pure-best/ph/unit256/odeu1/seed1/models/model_epoch_100_low_dim_v15_success_0.82.pth"
  #["ncp_u256_best_seed2"]="/work/robomimic/bc_trained_models/can/ncp-pure-best/ph/unit256/odeu1/seed2/models/model_epoch_100_low_dim_v15_success_1.0.pth"
  #["ncp_u256_best_seed3"]="/work/robomimic/bc_trained_models/can/ncp-pure-best/ph/unit256/odeu1/seed3/models/model_epoch_100_low_dim_v15_success_0.78.pth"
  ["ncp_u512_best_seed1"]="/work/robomimic/bc_trained_models/can/ncp-pure-best/ph/unit512/odeu3/seed1/models/model_epoch_400_low_dim_v15_success_0.72.pth"
  ["ncp_u512_best_seed2"]="/work/robomimic/bc_trained_models/can/ncp-pure-best/ph/unit512/odeu3/seed2/models/model_epoch_100_low_dim_v15_success_0.74.pth"
  ["ncp_u512_best_seed3"]="/work/robomimic/bc_trained_models/can/ncp-pure-best/ph/unit512/odeu3/seed3/models/model_epoch_150_low_dim_v15_success_0.78.pth"
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
        --odeu 3

    echo "Completed: ${name}"
    echo "----------------------------------------"
done
