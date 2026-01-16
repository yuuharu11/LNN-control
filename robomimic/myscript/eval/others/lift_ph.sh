#!/bin/bash

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15_2.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=0
CSV_PATH="/work/robomimic/csv/eval/lift/profile.csv"

# name と dataset_path の対応を associative array で定義
declare -A models=(
  #["bc"]="/work/robomimic/bc_trained_models/lift/core/bc/ph/models/model_epoch_600_low_dim_v15_success_1.0.pth"
  #["bc-pure_seed1"]="/work/robomimic/bc_trained_models/lift/bc-pure/ph/seed1/models/model_epoch_150_low_dim_v15_success_1.0.pth"
  #["bc-pure_seed2"]="/work/robomimic/bc_trained_models/lift/bc-pure/ph/seed2/models/model_epoch_150_low_dim_v15_success_0.98.pth"
  #["bc-pure_seed3"]="/work/robomimic/bc_trained_models/lift/bc-pure/ph/seed3/models/model_epoch_400_low_dim_v15_success_1.0.pth"
  #["bc-rnn"]="/work/robomimic/bc_trained_models/lift/core/bc_rnn/ph/models/model_epoch_250_low_dim_v15_success_1.0.pth"
  #["bc-rnn-pure_seed1"]="/work/robomimic/bc_trained_models/lift/bc_rnn-pure/ph/seed1/models/model_epoch_150_low_dim_v15_success_1.0.pth"
  #["bc-rnn-pure_seed2"]="/work/robomimic/bc_trained_models/lift/bc_rnn-pure/ph/seed2/models/model_epoch_100_low_dim_v15_success_1.0.pth"
  #["bc-rnn-pure_seed3"]="/work/robomimic/bc_trained_models/lift/bc_rnn-pure/ph/seed3/models/model_epoch_100_low_dim_v15_success_1.0.pth"
  #["bcq"]="/work/robomimic/bc_trained_models/lift/core/bcq/ph/models/model_epoch_800_low_dim_v15_success_1.0.pth"
  #["cql"]="/work/robomimic/bc_trained_models/lift/core/cql/ph/models/model_epoch_250_low_dim_v15_success_0.76.pth"
  #["hbc"]="/work/robomimic/bc_trained_models/core/hbc/lift/ph/low_dim/trained_models/core_hbc_lift_ph_low_dim/models/model_epoch_50_low_dim_v15_success_1.0.pth"
  #["iris"]="/work/robomimic/bc_trained_models/core/iris/lift/ph/low_dim/trained_models/core_iris_lift_ph_low_dim/models/model_epoch_50_low_dim_v15_success_1.0.pth"
  #["ncp_u64_best_seed1"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit64/seed1/models/model_epoch_250_low_dim_v15_success_1.0.pth"
  #["ncp_u64_best_seed2"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit64/seed2/models/model_epoch_150_low_dim_v15_success_0.96.pth"
  #["ncp_u64_best_seed3"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit64/seed3/models/model_epoch_350_low_dim_v15_success_1.0.pth"
  ["ncp_u64_best_seed4"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit64/seed4/models/model_epoch_350_low_dim_v15_success_0.96.pth"
  ["ncp_u64_best_seed5"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit64/seed5/models/model_epoch_400_low_dim_v15_success_0.96.pth"
  #["ncp_u128_best_seed1"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit128/seed1/models/model_epoch_100_low_dim_v15_success_1.0.pth"
  #["ncp_u128_best_seed2"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit128/seed2/models/model_epoch_450_low_dim_v15_success_1.0.pth"
  #["ncp_u128_best_seed3"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit128/seed3/models/model_epoch_150_low_dim_v15_success_1.0.pth"
  ["ncp_u128_best_seed4"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit128/seed4/models/model_epoch_150_low_dim_v15_success_1.0.pth"
  ["ncp_u128_best_seed5"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit128/seed5/models/model_epoch_300_low_dim_v15_success_1.0.pth"
  #["ncp_u256_best_seed1"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit256/seed1/models/model_epoch_150_low_dim_v15_success_1.0.pth"
  #["ncp_u256_best_seed2"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit256/seed2/models/model_epoch_50_low_dim_v15_success_1.0.pth"
  #["ncp_u256_best_seed3"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit256/seed3/models/model_epoch_150_low_dim_v15_success_1.0.pth"
  ["ncp_u256_best_seed4"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit256/seed4/models/model_epoch_350_low_dim_v15_success_1.0.pth"
  ["ncp_u256_best_seed5"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit256/seed5/models/model_epoch_200_low_dim_v15_success_1.0.pth"
  #["ncp_u512_best_seed1"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit512/seed1/models/model_epoch_50_low_dim_v15_success_1.0.pth"
  #["ncp_u512_best_seed2"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit512/seed2/models/model_epoch_50_low_dim_v15_success_1.0.pth"
  #["ncp_u512_best_seed3"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit512/seed3/models/model_epoch_150_low_dim_v15_success_1.0.pth"
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

    echo "Completed: ${name}"
    echo "----------------------------------------"
done
