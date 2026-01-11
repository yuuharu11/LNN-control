#!/bin/bash
set -euo pipefail

# ===== Can Task with Multiple Seeds & W&B Logging (Parallel) =====
WANDB_PROJECT="robomimic_can"
MAX_JOBS="${MAX_JOBS:-10}"   # 同時実行数（例: MAX_JOBS=4 ./core.sh）

echo "Can Task Training with Multiple Seeds (Parallel)"
echo "Project: ${WANDB_PROJECT}"
echo "MAX_JOBS: ${MAX_JOBS}"
echo ""

# ===== Can 設定リスト =====
declare -a CAN_CONFIGS=(
  "bc:can/ph/low_dim"
  "bc_rnn:can/ph/low_dim"
)

# seed をループ
SEEDS=(4 5 6 7 8 9 10)

TOTAL=$(( ${#CAN_CONFIGS[@]} * ${#SEEDS[@]} ))
echo "Total runs: ${TOTAL} (configs=${#CAN_CONFIGS[@]} x seeds=${#SEEDS[@]})"
echo ""

COMPLETED=0
FAILED=0
COUNT=0

# model名から出力フォルダ名へ（必要に応じてここを調整）
model_dir() {
  local model="$1"
  case "$model" in
    bc) echo "bc-pure" ;;
    bc_rnn) echo "bc-rnn-pure" ;;
    *) echo "$model" ;;
  esac
}

run_one() {
  local model="$1"
  local dataset="$2"
  local seed="$3"

  local dataset_type
  dataset_type="$(echo "$dataset" | cut -d'/' -f2)"

  local mdir
  mdir="$(model_dir "$model")"

  local wandb_name="${model}_${dataset_type}_seed${seed}"
  local exp_name="/work/robomimic/bc_trained_models/can/${mdir}/${dataset_type}/seed${seed}"
  local config_path="/work/robomimic/robomimic/exps/my_params/can/${model}.json"

  echo "[RUN] ${wandb_name}"
  echo "      Config: ${config_path}"
  echo "      Exp:    ${exp_name}"

  python /work/robomimic/robomimic/scripts/train.py \
    --config "${config_path}" \
    --name "${exp_name}" \
    --seed "${seed}" \
    --num_epochs 1000
}

wait_one_and_count() {
  # wait -n の終了コードで成功/失敗をカウント（set -e で落ちないようにする）
  set +e
  wait -n
  local status=$?
  set -e

  if (( status == 0 )); then
    COMPLETED=$((COMPLETED + 1))
  else
    FAILED=$((FAILED + 1))
  fi
}

# ===== 並列実行（スロットリング）=====
for config_spec in "${CAN_CONFIGS[@]}"; do
  MODEL="${config_spec%:*}"
  DATASET="${config_spec#*:}"

  for seed in "${SEEDS[@]}"; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] launch: ${MODEL} seed=${seed}"

    run_one "$MODEL" "$DATASET" "$seed" &

    # MAX_JOBS を超えたら1つ完了するまで待つ
    while (( $(jobs -pr | wc -l) >= MAX_JOBS )); do
      wait_one_and_count
    done
  done
done

# 残りジョブを全て回収
while (( $(jobs -pr | wc -l) > 0 )); do
  wait_one_and_count
done

echo ""
echo "=========================================="
echo "All training finished"
echo "=========================================="
echo "Completed: ${COMPLETED}/${TOTAL}"
echo "Failed:    ${FAILED}/${TOTAL}"
echo ""
echo "W&B Dashboard:"
echo "https://wandb.ai/yuuharuharuya1120-japan/${WANDB_PROJECT}"
echo ""