#!/bin/bash

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/square/ph/low_dim_v15.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=0
CSV_PATH="/work/robomimic/csv/eval/square/profile.csv"
# name と dataset_path の対応を associative array で定義
declare -A models=(
  #["bc"]="/work/robomimic/bc_trained_models/core/bc/square/ph/low_dim/trained_models/core_bc_square_ph_low_dim/models/model_epoch_1750_low_dim_v15_success_0.76.pth"
  ["bc-pure_seed1"]="/work/robomimic/bc_trained_models/square/bc-pure/ph/models/model_epoch_700_low_dim_v15_success_0.48.pth"
  ["bc-pure_seed2"]="/work/robomimic/bc_trained_models/square/bc-pure/ph/seed2/models/model_epoch_650_low_dim_v15_success_0.42.pth"
  ["bc-pure_seed3"]="/work/robomimic/bc_trained_models/square/bc-pure/ph/seed3/models/model_epoch_650_low_dim_v15_success_0.5.pth"
  #["bc-rnn"]="/work/robomimic/bc_trained_models/core/bc_rnn/square/ph/low_dim/trained_models/core_bc_rnn_square_ph_low_dim/models/model_epoch_550_low_dim_v15_success_0.68.pth"
  ["bc-rnn-pure_seed1"]="/work/robomimic/bc_trained_models/square/bc_rnn-pure/ph/seed1/models/model_epoch_900_low_dim_v15_success_0.6.pth"
  ["bc-rnn-pure_seed2"]="/work/robomimic/bc_trained_models/square/bc_rnn-pure/ph/seed2/models/model_epoch_600_low_dim_v15_success_0.54.pth"
  ["bc-rnn-pure_seed3"]="/work/robomimic/bc_trained_models/square/bc_rnn-pure/ph/seed3/models/model_epoch_450_low_dim_v15_success_0.54.pth"
  #["bcq"]="/work/robomimic/bc_trained_models/core/bcq/square/ph/low_dim/trained_models/core_bcq_square_ph_low_dim/models/model_epoch_1750_low_dim_v15_success_0.28.pth"
  #["cql"]="/work/robomimic/bc_trained_models/core/cql/square/ph/low_dim/trained_models/core_cql_square_ph_low_dim/models/model_epoch_1550_low_dim_v15_success_0.1.pth"
  #["hbc"]="/work/robomimic/bc_trained_models/core/hbc/square/ph/low_dim/trained_models/core_hbc_square_ph_low_dim/models/model_epoch_1900_low_dim_v15_success_0.58.pth"
  #["iris"]="/work/robomimic/bc_trained_models/core/iris/square/ph/low_dim/trained_models/core_iris_square_ph_low_dim/models/model_epoch_1000_low_dim_v15_success_0.48.pth"
  ["ncp_u64_best_seed1"]="/work/robomimic/bc_trained_models/square/ncp-pure-best/ph/unit64/seed1/models/model_epoch_400_low_dim_v15_success_0.16.pth"
  ["ncp_u64_best_seed2"]="/work/robomimic/bc_trained_models/square/ncp-pure-best/ph/unit64/seed2/models/model_epoch_350_low_dim_v15_success_0.26.pth"
  ["ncp_u64_best_seed3"]="/work/robomimic/bc_trained_models/square/ncp-pure-best/ph/unit64/seed3/models/model_epoch_450_low_dim_v15_success_0.32.pth"
  ["ncp_u128_best_seed1"]="/work/robomimic/bc_trained_models/square/ncp-pure-best/ph/unit128/seed1/models/model_epoch_250_low_dim_v15_success_0.32.pth"
  ["ncp_u128_best_seed3"]="/work/robomimic/bc_trained_models/square/ncp-pure-best/ph/unit128/seed3/models/model_epoch_650_low_dim_v15_success_0.46.pth"
  ["ncp_u256_seed1"]="/work/robomimic/bc_trained_models/square/ncp-pure-best/ph/unit256/odeu3/seed1/models/model_epoch_850_low_dim_v15_success_0.36.pth"
  ["ncp_u256_seed2"]="/work/robomimic/bc_trained_models/square/ncp-pure-best/ph/unit256/odeu3/seed2/models/model_epoch_450_low_dim_v15_success_0.38.pth"
  ["ncp_u256_seed3"]="/work/robomimic/bc_trained_models/square/ncp-pure-best/ph/unit256/odeu3/seed3/models/model_epoch_100_low_dim_v15_success_0.36.pth"
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
