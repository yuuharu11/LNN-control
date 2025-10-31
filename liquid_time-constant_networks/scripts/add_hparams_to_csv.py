#!/usr/bin/env python
"""
CSVファイルにチェックポイントパスから抽出したハイパーパラメータを追記する
"""
import pandas as pd
import re
from pathlib import Path

def extract_hyperparams_from_path(checkpoint_path):
    """
    チェックポイントパスからハイパーパラメータを抽出
    
    例: outputs/robomimic/lift/sweep/rnn/hid32_layer1/seed_1/checkpoints/last.ckpt
    -> hidden_size=32, n_layers=1
    """
    # 正規表現でhiddenサイズとレイヤー数を抽出
    match = re.search(r'hid(\d+)_layer(\d+)', checkpoint_path)
    
    if match:
        hidden_size = int(match.group(1))
        n_layers = int(match.group(2))
        return hidden_size, n_layers
    else:
        return None, None

def main():
    # CSVファイルのパス
    csv_path = Path("/work/liquid_time-constant_networks/csv/robomimic/lift/lstm/lstm.csv")
    
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    # CSVを読み込み
    df = pd.read_csv(csv_path)
    
    print(f"📊 Processing {len(df)} rows...")
    
    # ハイパーパラメータを抽出
    hidden_sizes = []
    n_layers_list = []
    
    for idx, row in df.iterrows():
        checkpoint_path = row['checkpoint_path']
        hidden_size, n_layers = extract_hyperparams_from_path(checkpoint_path)
        
        hidden_sizes.append(hidden_size)
        n_layers_list.append(n_layers)
        
        if idx < 5:  # 最初の5行を表示
            print(f"  Row {idx}: {Path(checkpoint_path).parent.parent.name}")
            print(f"    -> hidden_size={hidden_size}, n_layers={n_layers}")
    
    # 新しい列を追加（checkpoint_pathの直後に挿入）
    checkpoint_col_idx = df.columns.get_loc('checkpoint_path')
    
    df.insert(checkpoint_col_idx + 1, 'hidden_size', hidden_sizes)
    df.insert(checkpoint_col_idx + 2, 'n_layers', n_layers_list)
    
    # 元のファイルに上書き
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Updated: {csv_path}")
    print(f"\n📋 New columns added: hidden_size, n_layers")
    print(f"Total columns: {len(df.columns)}")
    
    # サンプルデータを表示
    print(f"\n📊 Sample data (first 5 rows):")
    print(df[['seed', 'hidden_size', 'n_layers', 'success_rate', 'avg_reward']].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
