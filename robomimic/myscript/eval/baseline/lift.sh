#!/bin/bash
# filepath: /work/robomimic/myscript/eval/baseline/lift.sh
set -euo pipefail

DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15_3.hdf5"
N_ROLLOUTS=1
HORIZON=400
SEED=0

CSV_BASE="/work/robomimic/result/u256"
mkdir -p "$(dirname "$CSV_BASE")"

model_path="/work/robomimic/trained_models/lift/u256/seed1_model_epoch_150_low_dim_v15_success_1.0.pth"
name="u256_temp"

python /work/robomimic/robomimic/scripts/run_trained_agent.py \
  --agent "$model_path" \
  --n_rollouts "$N_ROLLOUTS" \
  --horizon "$HORIZON" \
  --seed "$SEED" \
  --dataset_path "$DATASET_PATH" \
  --name "$name" \
  --calibration_times 1 \
  --calibration_path "/work/robomimic/logs/baseline/lift/u256/Seed1_neuron_1.json" \
  --calibration_percentile 99.9 
echo "Completed: ${name}"
echo "----------------------------------------" 

echo "=========================================="
echo "All experiments completed!"
echo "Results saved in ${CSV_BASE}"
echo "=========================================="