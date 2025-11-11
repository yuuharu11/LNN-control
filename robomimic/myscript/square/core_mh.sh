#!/bin/bash
# filepath: /work/robomimic/train_core.sh
set -euo pipefail

# ===== Lift Task with Multiple Seeds & W&B Logging (Sequential) =====
WANDB_PROJECT="robomimic_square"
echo "🚀 square Training"
echo "   Project: $WANDB_PROJECT"
echo ""

# ===== Lift 設定リスト =====
declare -a SQUARE_CONFIGS=(
    "bc:square/mh/low_dim"
    "bc_rnn:square/mh/low_dim"
    "bcq:square/mh/low_dim"
    "cql:square/mh/low_dim"
    "hbc:square/mh/low_dim"
    "iris:square/mh/low_dim"
)

TOTAL=${#SQUARE_CONFIGS[@]}
echo "Total runs: $TOTAL"
echo ""

COUNT=0
COMPLETED=0
FAILED=0

# ===== 逐次処理 =====
for config_spec in "${SQUARE_CONFIGS[@]}"; do
    MODEL="${config_spec%:*}"
    DATASET="${config_spec#*:}"
    
    COUNT=$((COUNT + 1))

    DATASET_TYPE=$(echo "$DATASET" | cut -d'/' -f2)

    WANDB_NAME="${MODEL}_${DATASET_TYPE}"
    EXP_NAME="/work/robomimic/bc_trained_models/square/${MODEL}/${DATASET_TYPE}"

    CONFIG_PATH="/work/robomimic/robomimic/exps/paper/core/${DATASET}/${MODEL}.json"
    
    echo "[$COUNT/$TOTAL] 🌱 Starting: $WANDB_NAME"
    echo "              Config: $CONFIG_PATH"
    echo ""
    
    python /work/robomimic/robomimic/scripts/train.py \
        --config "$CONFIG_PATH" \
        --name "$EXP_NAME" \
        --num_epochs 200 \
        --seed 1 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_name "$WANDB_NAME" \
        --wandb \

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