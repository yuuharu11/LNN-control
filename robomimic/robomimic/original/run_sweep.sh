#!/bin/bash

# 使用方法を表示
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --sweep-id ID       Sweep ID (optional, will create new sweep if not provided)"
    echo "  -n, --num-agents NUM    Number of parallel agents (default: 4)"
    echo "  -d, --dataset PATH      Dataset path (default: /work/robomimic/datasets/lift/ph/low_dim_v141.hdf5)"
    echo "  -c, --config PATH       Base config path (default: /work/robomimic/robomimic/exps/templates/bc.json)"
    echo "  -y, --yaml PATH         Sweep YAML config (default: /work/robomimic/robomimic/original/wandb_default.yaml)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Create new sweep and start 4 agents"
    echo "  $0"
    echo ""
    echo "  # Use existing sweep"
    echo "  $0 --sweep-id yuuharuharuya1120-japan/project/sweep-id"
    echo ""
    echo "  # Create new sweep with 8 agents"
    echo "  $0 --num-agents 8"
    exit 1
}

# デフォルト値
NUM_AGENTS=1
DATASET_PATH="/work/robomimic/datasets/lift/ph/low_dim_v15.hdf5"
CONFIG_PATH="/work/robomimic/robomimic/exps/templates/bc.json"
YAML_PATH="/work/robomimic/robomimic/original/wandb_default.yaml"
SWEEP_ID="yuuharuharuya1120-japan/work-robomimic_robomimic_original/xfdzjx5h"

# 引数をパース
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--sweep-id)
            SWEEP_ID="$2"
            shift 2
            ;;
        -n|--num-agents)
            NUM_AGENTS="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -y|--yaml)
            YAML_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# 環境変数を設定
export SWEEP_DATASET="$DATASET_PATH"
export SWEEP_BASE_CONFIG="$CONFIG_PATH"

echo "=========================================="
echo "WandB Sweep Setup"
echo "=========================================="
echo "Dataset:      $DATASET_PATH"
echo "Config:       $CONFIG_PATH"
echo "Num Agents:   $NUM_AGENTS"
echo "=========================================="
echo ""

# Sweep IDが指定されていない場合は新規作成
if [ -z "$SWEEP_ID" ]; then
    echo "Creating new WandB sweep from: $YAML_PATH"
    echo ""
    
    SWEEP_OUTPUT=$(wandb sweep "$YAML_PATH" 2>&1)
    echo "$SWEEP_OUTPUT"
    echo ""
    
    SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'wandb agent \K[^ ]+' | tail -1)
    
    if [ -z "$SWEEP_ID" ]; then
        echo "❌ Failed to create sweep"
        echo "Please check the YAML file and try again"
        exit 1
    fi
    
    echo "✅ Sweep created: $SWEEP_ID"
else
    echo "Using existing sweep: $SWEEP_ID"
fi

echo ""
echo "=========================================="
echo "Starting $NUM_AGENTS WandB Agents"
echo "=========================================="
echo ""

# エージェントを並列起動
for i in $(seq 1 $NUM_AGENTS); do
    echo "🚀 Starting agent $i/$NUM_AGENTS..."
    wandb agent "$SWEEP_ID" &
    
    # 少し待機（同時起動による競合を避ける）
    sleep 2
done

echo ""
echo "=========================================="
echo "✅ All $NUM_AGENTS agents started"
echo "=========================================="
echo ""
echo "📊 View sweep progress:"
echo "   https://wandb.ai/$(echo $SWEEP_ID | cut -d'/' -f1,2)/sweeps/$(echo $SWEEP_ID | cut -d'/' -f3)"
echo ""
echo "🛑 Stop all agents:"
echo "   pkill -f 'wandb agent $SWEEP_ID'"
echo "   # or press Ctrl+C to stop this script (agents will continue in background)"
echo ""
echo "📝 View agent output:"
echo "   Use 'wandb agent $SWEEP_ID' in separate terminal for foreground execution"
echo "=========================================="

# Ctrl+Cで全エージェントを停止できるようにする
trap "echo ''; echo 'Stopping all agents...'; pkill -f 'wandb agent $SWEEP_ID'; exit 0" INT

# バックグラウンドプロセスを待機
echo ""
echo "💡 Agents running in background. Press Ctrl+C to stop all agents."
echo ""
wait