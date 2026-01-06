#!/bin/bash

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15_2.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=0
shift=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)
CSV_BASE="/work/robomimic/csv/eval/lift/error/shift/"
mkdir -p ${CSV_BASE}

# name と dataset_path の対応を associative array で定義
declare -A models=(
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
  seed=${name##*_seed}
  for g in "${shift[@]}"; do
      echo "Running inference with shift noise stddev=${g}..."
      python /work/robomimic/robomimic/scripts/run_trained_agent.py \
          --agent "${model_path}" \
          --n_rollouts "${N_ROLLOUTS}" \
          --horizon "${HORIZON}" \
          --seed "${SEED}" \
          --dataset_path "${DATASET_PATH}" \
          --name "${name}_shift${g}" \
          --shift "${g}" \
          --csv_path "${CSV_BASE}${units}/shift${g}.csv" 

      echo "----------------------------------------"
  done
    echo "Completed: ${name}"
    echo "----------------------------------------"
done
echo "=========================================="
echo "All experiments completed!"
echo "Results saved in ${CSV_DIR}"
echo "=========================================="