#!/usr/bin/env python
"""
CFC CSVファイルにチェックポイントパスから抽出したハイパーパラメータを追記する
"""
import pandas as pd
import re
from pathlib import Path

def extract_cfc_hyperparams_from_path(checkpoint_path):
    """
    CFCのチェックポイントパスからハイパーパラメータを抽出
    
    例: outputs/robomimic/lift/sweep/cfc/units64_out4_in4/seed_1/checkpoints/last.ckpt
    -> units=64, output_units=4, input_size=4
    """
    # 正規表現でunits, output_units, input_sizeを抽出
    match = re.search(r'units(\d+)_out(\d+)_in(\d+)', checkpoint_path)
    
    if match:
        units = int(match.group(1))
        output_units = int(match.group(2))
        input_size = int(match.group(3))
        return units, output_units, input_size
    else:
        return None, None, None

def main():
    # CSVファイルのパス
    csv_path = Path("/work/liquid_time-constant_networks/csv/robomimic/lift/ncp/ltc.csv")
    
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    # CSVを読み込み
    df = pd.read_csv(csv_path)
    
    print(f"📊 Processing {len(df)} rows...")
    
    # ハイパーパラメータを抽出
    units_list = []
    output_units_list = []
    input_size_list = []
    
    for idx, row in df.iterrows():
        checkpoint_path = row['checkpoint_path']
        units, output_units, input_size = extract_cfc_hyperparams_from_path(checkpoint_path)
        
        units_list.append(units)
        output_units_list.append(output_units)
        input_size_list.append(input_size)
        
        if idx < 5:  # 最初の5行を表示
            print(f"  Row {idx}: {Path(checkpoint_path).parent.parent.name}")
            print(f"    -> units={units}, output_units={output_units}, input_size={input_size}")
    
    # 新しい列を追加（checkpoint_pathの直後に挿入）
    checkpoint_col_idx = df.columns.get_loc('checkpoint_path')
    
    df.insert(checkpoint_col_idx + 1, 'units', units_list)
    df.insert(checkpoint_col_idx + 2, 'output_units', output_units_list)
    df.insert(checkpoint_col_idx + 3, 'input_size', input_size_list)
    
    # 元のファイルに上書き
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Updated: {csv_path}")
    print(f"\n📋 New columns added: units, output_units, input_size")
    print(f"Total columns: {len(df.columns)}")
    
    # サンプルデータを表示
    print(f"\n📊 Sample data (first 5 rows):")
    sample_cols = ['seed', 'units', 'output_units', 'input_size', 'success_rate', 'avg_reward']
    print(df[sample_cols].head(5).to_string(index=False))
    
    # 統計情報
    print(f"\n📈 Hyperparameter statistics:")
    print(f"  Units: {sorted(df['units'].unique())}")
    print(f"  Output units: {sorted(df['output_units'].unique())}")
    print(f"  Input size: {sorted(df['input_size'].unique())}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")

if __name__ == "__main__":
    main()