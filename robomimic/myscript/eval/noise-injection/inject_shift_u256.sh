#!/bin/bash

# モデルファイルと共通パラメータ
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15_5.hdf5"
N_ROLLOUTS=100
HORIZON=400
SEED=0
shift=(0.06 0.07 0.08)
CSV_BASE="/work/robomimic/csv/result/error/shift/u256"
mkdir -p ${CSV_BASE}
MODEL_DIR="/work/robomimic/trained_models/lift/u256"
seed=1
for model_path in ${MODEL_DIR}/seed*_model_epoch_*_low_dim_v15_success_*; do
  for s in "${shift[@]}"; do
    if [[ -f "$model_path" ]]; then
      name="u256_${seed}"
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