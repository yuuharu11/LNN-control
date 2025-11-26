#!/bin/bash

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=0

# name と dataset_path の対応を associative array で定義
declare -A models=(
  ["bc"]="/work/robomimic/bc_trained_models/lift/core/bc/ph/models/model_epoch_600_low_dim_v15_success_1.0.pth"
  ["bc-rnn"]="/work/robomimic/bc_trained_models/lift/core/bc_rnn/ph/models/model_epoch_250_low_dim_v15_success_1.0.pth"
  ["bcq"]="/work/robomimic/bc_trained_models/lift/core/bcq/ph/models/model_epoch_800_low_dim_v15_success_1.0.pth"
  ["cql"]="/work/robomimic/bc_trained_models/lift/core/cql/ph/models/model_epoch_250_low_dim_v15_success_0.76.pth"
  ["hbc"]="/work/robomimic/bc_trained_models/core/hbc/lift/ph/low_dim/trained_models/core_hbc_lift_ph_low_dim/models/model_epoch_50_low_dim_v15_success_1.0.pth"
  ["iris"]="/work/robomimic/bc_trained_models/core/iris/lift/ph/low_dim/trained_models/core_iris_lift_ph_low_dim/models/model_epoch_50_low_dim_v15_success_1.0.pth"
  ["ncp_u128"]="/work/robomimic/bc_trained_models/lift/ncp/new-ph/unit128/seed1/models/model_epoch_50_low_dim_v15_success_1.0.pth"
  ["ncp_u256"]="/work/robomimic/bc_trained_models/lift/ncp/new-ph/unit256/seed1/models/model_epoch_50_low_dim_v15_success_1.0.pth"
)

# ノイズレベルの配列（0.0から1.0まで0.1刻み）
NOISE_LEVELS=(0.03 0.04 0.05)

# 各モデルとノイズレベルに対して推論を実行
for name in "${!models[@]}"; do
    model_path="${models[$name]}"
    
    echo "=========================================="
    echo "Model: ${name}"
    echo "=========================================="
    
    for noise in "${NOISE_LEVELS[@]}"; do
        # CSVパスにノイズレベルを含める
        CSV_PATH="/work/robomimic/csv/eval/lift/noise${noise}.csv"
        
        echo "Running inference for ${name} with noise=${noise}..."
        python /work/robomimic/robomimic/scripts/run_trained_agent.py \
            --agent "${model_path}" \
            --n_rollouts "${N_ROLLOUTS}" \
            --horizon "${HORIZON}" \
            --seed "${SEED}" \
            --dataset_path "${DATASET_PATH}" \
            --name "${name}_noise${noise}" \
            --csv_path "${CSV_PATH}" \
            --observation_noise "${noise}"
        
        echo "Completed: ${name} with noise=${noise}"
        echo "----------------------------------------"
    done
    
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "Results saved in /work/robomimic/csv/eval/"
echo "=========================================="