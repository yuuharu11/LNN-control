#!/bin/bash

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15_4.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=0
shift=(0.0 0.005 0.01 0.015 0.02 0.025 0.03)
CSV_BASE="/work/robomimic/csv/result/error/shift/u128"
mkdir -p ${CSV_BASE}
MODEL_DIR="/work/robomimic/trained_models/lift/u128"
seed=1
for model_path in ${MODEL_DIR}/seed*_model_epoch_*_low_dim_v15_success_*; do
  for s in "${shift[@]}"; do
    if [[ -f "$model_path" ]]; then
      name="u128_${seed}"
      echo "Running inference for ${name}..."
      python /work/robomimic/robomimic/scripts/run_trained_agent.py \
          --agent "${model_path}" \
          --n_rollouts "${N_ROLLOUTS}" \
          --horizon "${HORIZON}" \
          --seed "${SEED}" \
          --dataset_path "${DATASET_PATH}" \
          --name "${name}_shift${s}" \
          --shift "${s}" \
          --csv_path "${CSV_BASE}/shift${s}.csv" 

      echo "----------------------------------------"
    fi
  done
  seed=$((seed + 1))
    echo "Completed: ${model_path}"
    echo "----------------------------------------"
done
echo "=========================================="
echo "All experiments completed!"
echo "Results saved in ${CSV_BASE}"
echo "=========================================="