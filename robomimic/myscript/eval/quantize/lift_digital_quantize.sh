#!/bin/bash

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15_2.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=0
QUANTIZES=(7 5 1)
CSV_BASE="/work/robomimic/csv/eval/lift/quantize/digital/"
mkdir -p ${CSV_BASE}

# name と dataset_path の対応を associative array で定義
declare -A models=(
  ["ncp_u64_best_seed1"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit64/seed1/models/model_epoch_250_low_dim_v15_success_1.0.pth"
  ["ncp_u64_best_seed2"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit64/seed2/models/model_epoch_150_low_dim_v15_success_0.96.pth"
  ["ncp_u64_best_seed3"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit64/seed3/models/model_epoch_350_low_dim_v15_success_1.0.pth"
  ["ncp_u64_best_seed4"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit64/seed4/models/model_epoch_350_low_dim_v15_success_0.96.pth"
  ["ncp_u64_best_seed5"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit64/seed5/models/model_epoch_400_low_dim_v15_success_0.96.pth"
  ["ncp_u128_best_seed1"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit128/seed1/models/model_epoch_100_low_dim_v15_success_1.0.pth"
  ["ncp_u128_best_seed2"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit128/seed2/models/model_epoch_450_low_dim_v15_success_1.0.pth"
  ["ncp_u128_best_seed3"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit128/seed3/models/model_epoch_150_low_dim_v15_success_1.0.pth"
  ["ncp_u128_best_seed4"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit128/seed4/models/model_epoch_150_low_dim_v15_success_1.0.pth"
  ["ncp_u128_best_seed5"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit128/seed5/models/model_epoch_300_low_dim_v15_success_1.0.pth"
  ["ncp_u256_best_seed1"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit256/seed1/models/model_epoch_150_low_dim_v15_success_1.0.pth"
  ["ncp_u256_best_seed2"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit256/seed2/models/model_epoch_50_low_dim_v15_success_1.0.pth"
  ["ncp_u256_best_seed3"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit256/seed3/models/model_epoch_150_low_dim_v15_success_1.0.pth"
  ["ncp_u256_best_seed4"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit256/seed4/models/model_epoch_350_low_dim_v15_success_1.0.pth"
  ["ncp_u256_best_seed5"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit256/seed5/models/model_epoch_200_low_dim_v15_success_1.0.pth"
)

# 各データセットに対して逐次推論を実行
for name in "${!models[@]}"; do
  model_path="${models[$name]}"

  # unitsの抽出
  units=$(echo "${model_path}" | grep -o 'unit[0-9]\+')
  units=${units:-unit_unknown}
    for quantize in "${QUANTIZES[@]}"; do
        echo "Running inference for ${name} with ${quantize}-bit quantization..."
        python /work/robomimic/robomimic/scripts/run_trained_agent.py \
            --agent "${model_path}" \
            --n_rollouts "${N_ROLLOUTS}" \
            --horizon "${HORIZON}" \
            --seed "${SEED}" \
            --dataset_path "${DATASET_PATH}" \
            --digital_RRAM_quantization "${quantize}" \
            --name "${units}_quantized_${quantize}bit" \
            --csv_path "${CSV_BASE}${units}_quantized_${quantize}bit.csv"

        echo "Completed: ${name} with ${quantize}-bit quantization"
        echo "----------------------------------------"
    done

    echo "Completed: ${name}"
    echo "----------------------------------------"
done
echo "=========================================="
echo "All experiments completed!"
echo "Results saved in ${CSV_DIR}"
echo "=========================================="