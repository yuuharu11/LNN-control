#!/bin/bash
# filepath: /work/robomimic/myscript/eval/baseline/lift.sh
set -euo pipefail

DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15_3.hdf5"
N_ROLLOUTS=1
HORIZON=400
SEED=0

CSV_BASE="/work/robomimic/result/u128"
mkdir -p "$(dirname "$CSV_BASE")"

model_path="/work/robomimic/bc_trained_models/lift/ncp-post/ph/unit128/seed1/models/model_epoch_50_low_dim_v15_success_0.0.pth"
name="temp"

python /work/robomimic/robomimic/scripts/run_trained_agent.py \
  --agent "$model_path" \
  --n_rollouts "$N_ROLLOUTS" \
  --horizon "$HORIZON" \
  --seed "$SEED" \
  --dataset_path "$DATASET_PATH" \
  --name "$name" 
echo "Completed: ${name}"
echo "----------------------------------------" 

echo "=========================================="
echo "All experiments completed!"
echo "Results saved in ${CSV_BASE}"
echo "=========================================="