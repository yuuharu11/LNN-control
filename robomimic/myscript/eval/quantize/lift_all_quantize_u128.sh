#!/bin/bash

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15_4.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=0
high_quantize=(6 5 4)
low_quantize=(5 4 3)
CSV_BASE="/work/robomimic/csv/eval/lift/quantize/all/weight-6bit/ADC-8bit"
LOG_PATH="/work/robomimic/logs/quantize/best/calibration/u128"
mkdir -p ${CSV_BASE}

# name と dataset_path の対応を associative array で定義
declare -A models=(
  ["ncp_u128_best_seed1"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit128/seed1/models/model_epoch_100_low_dim_v15_success_1.0.pth"
  ["ncp_u128_best_seed2"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit128/seed2/models/model_epoch_450_low_dim_v15_success_1.0.pth"
  ["ncp_u128_best_seed3"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit128/seed3/models/model_epoch_150_low_dim_v15_success_1.0.pth"
  ["ncp_u128_best_seed4"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit128/seed4/models/model_epoch_150_low_dim_v15_success_1.0.pth"
  ["ncp_u128_best_seed5"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit128/seed5/models/model_epoch_300_low_dim_v15_success_1.0.pth"
  )
# 各データセットに対して逐次推論を実行
for name in "${!models[@]}"; do
  model_path="${models[$name]}"

  # unitsの抽出
  units=$(echo "${model_path}" | grep -o 'unit[0-9]\+')
  units=${units:-unit_unknown}
  seed=${name##*_seed}
    for high_quantize in "${high_quantize[@]}"; do
        for low_quantize in "${low_quantize[@]}"; do
          echo "Running inference for ${name} with ${high_quantize}-bit ${low_quantize}-bit quantization..."
          python /work/robomimic/robomimic/scripts/run_trained_agent.py \
              --agent "${model_path}" \
              --n_rollouts "${N_ROLLOUTS}" \
              --horizon "${HORIZON}" \
              --seed "${SEED}" \
              --dataset_path "${DATASET_PATH}" \
              --name "${name}_quantized_${high_quantize}-${low_quantize}_w6bit" \
              --calibration_times 3 \
              --calibration_path "${LOG_PATH}/Seed${seed}.json" \
              --calibration_percentile 99.9 \
              --digital_SRAM_quantization 8 \
              --digital_RRAM_quantization 8 \
              --weight_quantization 6 \
              --LUT_quantization ${low_quantize} \
              --CAM_quantization ${high_quantize} \
              --ADC_quantization 8 \
              --DAC_quantization ${low_quantize} \
              --csv_path "${CSV_BASE}/DAC-${low_quantize}bit/LUT-${low_quantize}bit/CAM-${high_quantize}bit/8-8/${units}.csv" 

          echo "Completed: ${name} with ${quantize}-bit quantization"
          echo "----------------------------------------"
      done
    done
    echo "Completed: ${name}"
    echo "----------------------------------------"
done
echo "=========================================="
echo "All experiments completed!"
echo "Results saved in ${CSV_DIR}"
echo "=========================================="