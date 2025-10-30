
import h5py
import numpy as np

# データセットのパス
dataset_path = "/work/robomimic/datasets/lift/mg/low_dim_sparse_v15.hdf5"

# HDF5 ファイルを開く
with h5py.File(dataset_path, "r") as f:
    # トップレベルのキー確認
    print("=" * 80)
    print("Top-level keys:", list(f.keys()))
    print("=" * 80)
    
    # 'data' キーの内容を確認
    if 'data' in f:
        print("\n'data' group keys:", list(f['data'].keys()))
        
        # 最初のデモンストレーションを取得
        demo_keys = list(f['data'].keys())
        if demo_keys:
            first_demo = demo_keys[0]
            print(f"\nFirst demo: {first_demo}")
            print(f"Keys in '{first_demo}':", list(f['data'][first_demo].keys()))
            
            # 観測データの確認
            if 'obs' in f['data'][first_demo]:
                obs_keys = list(f['data'][first_demo]['obs'].keys())
                print(f"\n📋 Observation keys in '{first_demo}':")
                for key in obs_keys:
                    obs_data = f['data'][first_demo]['obs'][key]
                    print(f"  - {key}: shape={obs_data.shape}, dtype={obs_data.dtype}")
            
            # アクションデータの確認
            if 'actions' in f['data'][first_demo]:
                actions = f['data'][first_demo]['actions']
                print(f"\n🎮 Actions in '{first_demo}':")
                print(f"  shape: {actions.shape}, dtype: {actions.dtype}")
                print(f"  min: {np.min(actions, axis=0)}")
                print(f"  max: {np.max(actions, axis=0)}")
            
            # 報酬データの確認
            if 'rewards' in f['data'][first_demo]:
                rewards = f['data'][first_demo]['rewards']
                print(f"\n💰 Rewards in '{first_demo}':")
                print(f"  shape: {rewards.shape}")
                print(f"  total: {np.sum(rewards)}")
            
            # dones の確認
            if 'dones' in f['data'][first_demo]:
                dones = f['data'][first_demo]['dones']
                print(f"\n✅ Dones in '{first_demo}':")
                print(f"  shape: {dones.shape}")
                print(f"  episode length: {len(dones)}")
    
    # 'mask' の確認
    if 'mask' in f:
        mask_data = f['mask']
        print(f"\n🎭 Mask data:")
        if isinstance(mask_data, h5py.Dataset):
            print(f"  shape: {mask_data.shape}, dtype: {mask_data.dtype}")
        else:
            print(f"  keys: {list(mask_data.keys())}")
    
    # データセット全体の統計
    print("\n" + "=" * 80)
    print("Dataset Statistics:")
    print("=" * 80)
    if 'data' in f:
        num_demos = len(f['data'].keys())
        print(f"Number of demonstrations: {num_demos}")
        
        # 全デモの長さを確認
        demo_lengths = []
        for demo_key in list(f['data'].keys())[:5]:  # 最初の5個のみ
            if 'actions' in f['data'][demo_key]:
                demo_lengths.append(len(f['data'][demo_key]['actions']))
        
        if demo_lengths:
            print(f"Sample demo lengths: {demo_lengths}")
            print(f"Average length: {np.mean(demo_lengths):.1f}")
