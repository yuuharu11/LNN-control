#!/bin/bash

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15_4.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=0
gaussian=(0.11 0.12 0.13 0.14 0.15)
CSV_BASE="/work/robomimic/csv/eval/lift/error/gaussian/"
LOG_PATH="/work/robomimic/logs/quantize/gaussian/calibration/u64"
mkdir -p ${CSV_BASE}

# name と dataset_path の対応を associative array で定義
declare -A models=(
  ["ncp_u64_best_seed1"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit64/seed1/models/model_epoch_250_low_dim_v15_success_1.0.pth"
  ["ncp_u64_best_seed2"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit64/seed2/models/model_epoch_150_low_dim_v15_success_0.96.pth"
  ["ncp_u64_best_seed3"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit64/seed3/models/model_epoch_350_low_dim_v15_success_1.0.pth"
  ["ncp_u64_best_seed4"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit64/seed4/models/model_epoch_350_low_dim_v15_success_0.96.pth"
  ["ncp_u64_best_seed5"]="/work/robomimic/bc_trained_models/lift/ncp-pure-best/ph/unit64/seed5/models/model_epoch_400_low_dim_v15_success_0.96.pth"
  )

# 各データセットに対して逐次推論を実行
for name in "${!models[@]}"; do
  model_path="${models[$name]}"

  # unitsの抽出
  units=$(echo "${model_path}" | grep -o 'unit[0-9]\+')
  units=${units:-unit_unknown}
  seed=${name##*_seed}
  for g in "${gaussian[@]}"; do
      python /work/robomimic/robomimic/scripts/run_trained_agent.py \
          --agent "${model_path}" \
          --n_rollouts "${N_ROLLOUTS}" \
          --horizon "${HORIZON}" \
          --seed "${SEED}" \
          --dataset_path "${DATASET_PATH}" \
          --name "${name}_gaussian${g}" \
          --calibration_times 3 \
          --calibration_path "${LOG_PATH}/Seed${seed}.json" \
          --calibration_percentile 99.9 \
          --gaussian "${g}" \
          --csv_path "${CSV_BASE}${units}/gaussian${g}.csv" 

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