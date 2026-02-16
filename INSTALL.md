# 環境構築ガイド — LTC 学習 (Robomimic / Robosuite)

## クイックスタート

### 1. Miniconda のインストール

```bash
apt install wget
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source $HOME/.bashrc && conda --version
```

### 2. Conda 環境の作成

```bash
conda create -n robomimic_venv python=3.10
conda activate robomimic_venv
```

### 3. 依存ライブラリのインストール

```bash
# システムライブラリ
apt update
apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# EGL（GPU 描画 — /dev/dri が存在する場合に推奨）
apt install -y libegl1-mesa libegl1-mesa-dev libgles2-mesa-dev

# OSMesa（CPU 描画 — フォールバック用）
apt install -y libosmesa6-dev
```

### 3.1 OSMesa 描画用の libstdc++ 互換性修正

```bash
# 方法 1: conda の libstdc++ を更新（推奨）
conda install -c conda-forge libstdcxx-ng

# 方法 2: システムライブラリを優先する（代替）
# ~/.bashrc に追記:
echo 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 4. Robomimic と Robosuite のソースインストール

```bash
# robomimic のインストール
cd /work/robosuite/robomimic
pip install -e .

# robosuite のインストール
cd /work/robosuite
pip install -e .
```

### 5. インストールの確認

```bash
# PyTorch の確認
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# robosuite の確認
python -c "import robosuite; print('Robosuite version:', robosuite.__version__)"

# ncps の確認
python -c "import ncps; print('NCP/LTC imported successfully')"
```

## 代替: フルインストール

オプションパッケージを含むすべての依存をインストールする場合:

```bash
pip install -r requirements_full.txt
```

## トラブルシューティング

### 問題: robosuite の ModuleNotFoundError

**対処:**
```bash
export PYTHONPATH="/work/robosuite:$PYTHONPATH"
```

または `.bashrc` に追記して恒久化:
```bash
echo 'export PYTHONPATH="/work/robosuite:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

### 問題: CUDA バージョンの不一致

**対処:**
CUDA バージョンを確認:
```bash
nvidia-smi
```

対応する PyTorch バージョンを https://pytorch.org/get-started/locally/ からインストールしてください。

### 問題: EGL / OSMesa 描画エラー

**対処 1: EGL + GPU を使用（推奨 — 高速）**
```bash
# DRI デバイスが利用可能か確認
ls -la /dev/dri/  # renderD128, card0 等が表示されるはず

# EGL 用の環境変数を設定
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# ~/.bashrc に追記して恒久化
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
echo 'export PYOPENGL_PLATFORM=egl' >> ~/.bashrc
source ~/.bashrc

# 動作確認
python -c "import os; os.environ['MUJOCO_GL']='egl'; os.environ['PYOPENGL_PLATFORM']='egl'; import mujoco; print('✅ EGL OK - GPU rendering enabled')"
```

**対処 2: OSMesa を使用（ソフトウェア描画 — 低速だがどこでも動作）**
```bash
# OSMesa のインストール
apt install -y libosmesa6-dev

# 互換性のため libstdc++ を更新
conda install -c conda-forge libstdcxx-ng

# 環境変数を設定
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# 動作確認
python -c "import os; os.environ['MUJOCO_GL']='osmesa'; os.environ['PYOPENGL_PLATFORM']='osmesa'; import mujoco; print('✅ OSMesa OK - CPU rendering')"
```

**対処 3: 描画なしモード（カメラ観測を使わない学習向け）**
```python
env = suite.make(
    "Lift",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)
```
