#!/bin/bash

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=0
CSV_PATH="/work/robomimic/csv/eval/lift-noise.csv"

# name と dataset_path の対応を associative array で定義
declare -A models=(
  #["bc"]="/work/robomimic/bc_trained_models/lift/core/bc/ph/models/model_epoch_600_low_dim_v15_success_1.0.pth"
  #["bc-rnn"]="/work/robomimic/bc_trained_models/lift/core/bc_rnn/ph/models/model_epoch_250_low_dim_v15_success_1.0.pth"
  #["bcq"]="/work/robomimic/bc_trained_models/lift/core/bcq/ph/models/model_epoch_800_low_dim_v15_success_1.0.pth"
  #["cql"]="/work/robomimic/bc_trained_models/lift/core/cql/ph/models/model_epoch_250_low_dim_v15_success_0.76.pth"
  #["hbc"]="/work/robomimic/bc_trained_models/core/hbc/lift/ph/low_dim/trained_models/core_hbc_lift_ph_low_dim/models/model_epoch_50_low_dim_v15_success_1.0.pth"
  #["iris"]="/work/robomimic/bc_trained_models/core/iris/lift/ph/low_dim/trained_models/core_iris_lift_ph_low_dim/models/model_epoch_50_low_dim_v15_success_1.0.pth"
  ["ncp_u128"]="/work/robomimic/bc_trained_models/lift/ncp/new-ph/unit128/seed1/models/model_epoch_50_low_dim_v15_success_1.0.pth"
  ["ncp_u256"]="/work/robomimic/bc_trained_models/lift/ncp/new-ph/unit256/seed1/models/model_epoch_50_low_dim_v15_success_1.0.pth"
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
        --observation_noise 0.1

    echo "Completed: ${name}"
    echo "----------------------------------------"
done
