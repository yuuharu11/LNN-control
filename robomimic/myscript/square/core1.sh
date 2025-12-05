#!/bin/bash
# filepath: /work/robomimic/train_core.sh
set -euo pipefail

# ===== Lift Task with Multiple Seeds & W&B Logging (Sequential) =====
WANDB_PROJECT="robomimic_square"
echo "🚀 Square Training"
echo "   Project: $WANDB_PROJECT"
echo ""

# ===== Lift 設定リスト =====
declare -a SQUARE_CONFIGS=(
    "bc:square/ph/low_dim"
    "bc_rnn:square/ph/low_dim"
    #"bcq:square/ph/low_dim"
    #"cql:square/ph/low_dim"
    #"hbc:square/ph/low_dim"
    #"iris:square/ph/low_dim"
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

    WANDB_NAME="${MODEL}_seed1"
    EXP_NAME="/work/robomimic/bc_trained_models/square/${MODEL}-pure/${DATASET_TYPE}/seed1"

    CONFIG_PATH="/work/robomimic/robomimic/exps/my_params/square/${MODEL}.json"
    
    echo "[$COUNT/$TOTAL] 🌱 Starting: $WANDB_NAME"
    echo "              Config: $CONFIG_PATH"
    echo ""
    
    python /work/robomimic/robomimic/scripts/train.py \
        --config "$CONFIG_PATH" \
        --name "$EXP_NAME" \
        --seed 1 \
        --num_epochs 1000 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_name "$WANDB_NAME" \
        --wandb \

done

echo ""
echo "=========================================="
echo "🏁 全トレーニング完了"
echo "=========================================="
echo ""
echo "📊 W&B ダッシュボード:"
echo "   https://wandb.ai/yuuharuharuya1120-japan/$WANDB_PROJECT"
echo ""