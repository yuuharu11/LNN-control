#!/bin/bash
# filepath: /work/robomimic/train_lnn.sh

# ===== Multiple Seeds & Datasets Training Script =====
# LNN モデルを複数の seed と dataset で学習します

SEEDS=(1 2 3 4 5)
WANDB_PROJECT="robomimic_lift"
DATASETS=(
    "/work/robomimic/datasets/lift/ph/low_dim_v15.hdf5"
    "/work/robomimic/datasets/lift/mg/low_dim_sparse_v15.hdf5"
    "/work/robomimic/datasets/lift/mh/low_dim_v15.hdf5"
)


echo "🚀 LNN Training with Multiple Seeds"
echo "   Project: $WANDB_PROJECT"
echo "   Seeds: ${SEEDS[@]}"
echo "   Datasets: ${#DATASETS[@]}"
echo ""

TOTAL=$((${#SEEDS[@]} * ${#DATASETS[@]}))
COUNT=0

# ===== ループで学習実行 =====
for SEED in "${SEEDS[@]}"; do
  for DATA_PATH in "${DATASETS[@]}"; do
    COUNT=$((COUNT + 1))
    
    # データセット名(ph, mg, mh)をファイルパスではなく親ディレクトリから取得
    DATA_TYPE_DIR=$(dirname "$DATA_PATH")
    DATASET_NAME=$(basename "$DATA_TYPE_DIR")
    
    # wandb_name と exp_name を設定
    WANDB_NAME="lnn_seed${SEED}_${DATASET_NAME}"
    EXP_NAME="lift/lnn/seed${SEED}_${DATASET_NAME}"
    
    echo "[$COUNT/$TOTAL] 🌱 Starting: seed=$SEED, dataset=$DATASET_NAME"
    echo "   wandb_name: $WANDB_NAME"
    
    python /work/robomimic/robomimic/scripts/train.py \
      --name "$EXP_NAME" \
      --dataset "$DATA_PATH" \
      --config /work/robomimic/robomimic/exps/my_params/lnn.json \
      --seed "$SEED" \
      --wandb_project "$WANDB_PROJECT" \
      --wandb_name "$WANDB_NAME" 
  done
done

echo ""
echo "✅ $TOTAL/$TOTAL トレーニング完了"
echo ""