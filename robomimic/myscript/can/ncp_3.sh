#!/bin/bash

# ===== Multiple Seeds & Datasets & Units Training Script =====
# NCP モデルを複数の seed、dataset、units で学習します

SEEDS=(3)
WANDB_PROJECT="robomimic_can"
DATASETS=(
    "/work/robomimic/datasets/can/ph/low_dim_v15.hdf5"
)
# ✅ UNITS パラメータを追加
UNITS=(128)

echo "🚀 NCP mixed_memory Training with Multiple Seeds, Datasets & Units"
echo "   Project: $WANDB_PROJECT"
echo "   Seeds: ${SEEDS[@]}"
echo "   Datasets: ${#DATASETS[@]}"
echo "   Units: ${UNITS[@]}"
echo ""

# ✅ TOTAL を計算（Seeds × Datasets × Units）
TOTAL=$((${#SEEDS[@]} * ${#DATASETS[@]} * ${#UNITS[@]}))
COUNT=0
COMPLETED=0
FAILED=0

echo "Total runs: $TOTAL"
echo ""

# ===== ループで学習実行 =====
for SEED in "${SEEDS[@]}"; do
  for DATA_PATH in "${DATASETS[@]}"; do
    # ✅ UNITS のループを追加
    for UNIT in "${UNITS[@]}"; do
      COUNT=$((COUNT + 1))
      
      # データセット名(ph, mg, mh)をファイルパスではなく親ディレクトリから取得
      DATA_TYPE_DIR=$(dirname "$DATA_PATH")
      DATASET_NAME=$(basename "$DATA_TYPE_DIR")
      
      # ✅ wandb_name と exp_name に UNIT を含める
      WANDB_NAME="ncp_u${UNIT}_seed${SEED}_${DATASET_NAME}"
      EXP_NAME="can/ncp-pure-best/${DATASET_NAME}/unit${UNIT}/seed${SEED}"
      
      echo "[$COUNT/$TOTAL] 🌱 Starting: seed=$SEED, dataset=$DATASET_NAME, unit=$UNIT"
      echo "   wandb_name: $WANDB_NAME"
      echo "   exp_name: $EXP_NAME"
      echo ""
      
      # ✅ エラーハンドリングを追加
      if python /work/robomimic/robomimic/scripts/train.py \
        --name "$EXP_NAME" \
        --dataset "$DATA_PATH" \
        --config /work/robomimic/robomimic/exps/my_params/can/ncp.json \
        --num_epochs 1000 \
        --seed "$SEED" \
        --units "$UNIT" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_name "$WANDB_NAME" \
        --wandb; then
        COMPLETED=$((COMPLETED + 1))
        echo "✅ $WANDB_NAME completed successfully"
      else
        FAILED=$((FAILED + 1))
        echo "❌ $WANDB_NAME failed"
      fi
      
      echo ""
    done
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