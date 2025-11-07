#!/bin/bash
# filepath: /work/robomimic/train_core.sh
set -euo pipefail

# ===== Lift Task with Multiple Seeds & W&B Logging (Sequential) =====
WANDB_PROJECT="robomimic_lift"
SEEDS=(1 2 3 4 5)
echo "🚀 Lift Task Training with Multiple Seeds (Sequential)"
echo "   Project: $WANDB_PROJECT"
echo "   Seeds: ${SEEDS[@]}"
echo ""

# ===== Lift 設定リスト =====
declare -a LIFT_CONFIGS=(
    "bc:lift/ph/low_dim"
    "bc_rnn:lift/ph/low_dim"
    "bcq:lift/ph/low_dim"
    "cql:lift/ph/low_dim"
    "hbc:lift/ph/low_dim"
    "iris:lift/ph/low_dim"
    "bc:lift/mh/low_dim"
    "bc_rnn:lift/mh/low_dim"
    "bcq:lift/mh/low_dim"
    "cql:lift/mh/low_dim"
    "hbc:lift/mh/low_dim"
    "iris:lift/mh/low_dim"
    "bc_rnn:lift/mg/low_dim_sparse"
    "bcq:lift/mg/low_dim_sparse"
    "cql:lift/mg/low_dim_sparse"
    "hbc:lift/mg/low_dim_sparse"
    "iris:lift/mg/low_dim_sparse"
)

TOTAL=$((${#LIFT_CONFIGS[@]} * ${#SEEDS[@]}))
echo "Total runs: $TOTAL"
echo ""

COUNT=0
COMPLETED=0
FAILED=0

# ===== 逐次処理 =====
for config_spec in "${LIFT_CONFIGS[@]}"; do
    MODEL="${config_spec%:*}"
    DATASET="${config_spec#*:}"
    
    for SEED in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))

        DATASET_TYPE=$(echo "$DATASET" | cut -d'/' -f2)

        WANDB_NAME="${MODEL}_seed${SEED}_${DATASET_TYPE}"
        EXP_NAME="lift/${MODEL}/seed${SEED}_${DATASET_TYPE}"

        CONFIG_PATH="/work/robomimic/robomimic/exps/paper/core/${DATASET}/${MODEL}.json"
        
        echo "[$COUNT/$TOTAL] 🌱 Starting: $WANDB_NAME"
        echo "              Config: $CONFIG_PATH"
        echo "              Seed: $SEED"
        echo ""
        
        python /work/robomimic/robomimic/scripts/train.py \
          --config "$CONFIG_PATH" \
          --name "$EXP_NAME" \
          --seed "$SEED" \
          --wandb_project "$WANDB_PROJECT" \
          --wandb_name "$WANDB_NAME" \
          --wandb \
        
        echo ""
    done
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