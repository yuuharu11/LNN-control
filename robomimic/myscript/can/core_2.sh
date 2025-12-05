#!/bin/bash
# filepath: /work/robomimic/train_core.sh
set -euo pipefail

# ===== Lift Task with Multiple Seeds & W&B Logging (Sequential) =====
WANDB_PROJECT="robomimic_can"
echo "🚀 Can Training"
echo "   Project: $WANDB_PROJECT"
echo ""

# ===== Lift 設定リスト =====
declare -a CAN_CONFIGS=(
    "bc:can/ph/low_dim"
    "bc_rnn:can/ph/low_dim"
)


TOTAL=${#CAN_CONFIGS[@]}
echo "Total runs: $TOTAL"
echo ""

COUNT=0
COMPLETED=0
FAILED=0

# ===== 逐次処理 =====
for config_spec in "${CAN_CONFIGS[@]}"; do
    MODEL="${config_spec%:*}"
    DATASET="${config_spec#*:}"
    
    COUNT=$((COUNT + 1))

    DATASET_TYPE=$(echo "$DATASET" | cut -d'/' -f2)

    WANDB_NAME="${MODEL}_${DATASET_TYPE}"
    EXP_NAME="/work/robomimic/bc_trained_models/can/${MODEL}-pure/${DATASET_TYPE}/seed2"

    CONFIG_PATH="/work/robomimic/robomimic/exps/my_params/can/${MODEL}.json"
    
    echo "[$COUNT/$TOTAL] 🌱 Starting: $WANDB_NAME"
    echo "              Config: $CONFIG_PATH"
    echo ""
    
    python /work/robomimic/robomimic/scripts/train.py \
        --name "$EXP_NAME" \
        --config "$CONFIG_PATH" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_name "$WANDB_NAME" \
        --wandb \
        --seed 2 \
        --num_epochs 1000 \

done

echo ""
echo "=========================================="
echo "🏁 全トレーニング完了"
echo "=========================================="
echo "完了: $COMPLETED/$TOTAL"
echo "失敗: $FAILED/$TOTAL"
echo ""
echo "📊 W&B ダッシュボード:"
echo "   https://wandb.ai/yuuharuharuya1120-japan/$WANDB_PROJECT"
echo ""