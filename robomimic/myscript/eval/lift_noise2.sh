#!/bin/bash

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15_2.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=0

# name と dataset_path の対応を associative array で定義
declare -A models=(
  ["bc_seed1"]="/work/robomimic/bc_trained_models/lift/bc-pure/ph/seed1/models/model_epoch_150_low_dim_v15_success_1.0.pth"
  ["bc_seed2"]="/work/robomimic/bc_trained_models/lift/bc-pure/ph/seed2/models/model_epoch_150_low_dim_v15_success_0.98.pth"
  ["bc_seed3"]="/work/robomimic/bc_trained_models/lift/bc-pure/ph/seed3/models/model_epoch_400_low_dim_v15_success_1.0.pth"
  ["bc-rnn_seed1"]="/work/robomimic/bc_trained_models/lift/bc_rnn-pure/ph/seed1/models/model_epoch_150_low_dim_v15_success_1.0.pth"
  ["bc-rnn_seed2"]="/work/robomimic/bc_trained_models/lift/bc_rnn-pure/ph/seed2/models/model_epoch_100_low_dim_v15_success_1.0.pth"
  ["bc-rnn_seed3"]="/work/robomimic/bc_trained_models/lift/bc_rnn-pure/ph/seed3/models/model_epoch_100_low_dim_v15_success_1.0.pth"
  ["ncp-pure_u128_seed1"]="/work/robomimic/bc_trained_models/lift/ncp-pure-new/ph/unit128/seed1/models/model_epoch_200_low_dim_v15_success_1.0.pth"
  ["ncp-pure_u128_seed2"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit128/seed2/models/model_epoch_450_low_dim_v15_success_1.0.pth"
  ["ncp-pure_u128_seed3"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit128/seed3/models/model_epoch_150_low_dim_v15_success_1.0.pth"
  ["ncp-pure_u256_seed1"]="/work/robomimic/bc_trained_models/lift/ncp-pure-new/ph/unit256/seed1/models/model_epoch_100_low_dim_v15_success_0.98.pth"
  ["ncp-pure_u256_seed2"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit256/seed2/models/model_epoch_50_low_dim_v15_success_1.0.pth"
  ["ncp-pure_u256_seed3"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit256/seed3/models/model_epoch_150_low_dim_v15_success_1.0.pth"
)

# ノイズレベルの配列（0.0から0.1まで0.01刻み）
NOISE_LEVELS=(0.0025)
# 各モデルとノイズレベルに対して推論を実行
for name in "${!models[@]}"; do
    model_path="${models[$name]}"
    
    echo "=========================================="
    echo "Model: ${name}"
    echo "=========================================="
    
    for noise in "${NOISE_LEVELS[@]}"; do
        # CSVパスにノイズレベルを含める
        CSV_PATH="/work/robomimic/csv/lift/noise${noise}.csv"
        
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