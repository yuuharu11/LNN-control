# Robosuite & Robomimic 統合リポジトリ

このリポジトリは [robosuite](https://github.com/ARISE-Initiative/robosuite) と [robomimic](https://github.com/ARISE-Initiative/robomimic) を統合し、反射モデルの実験を行う環境です。  

## 概要

- **Robosuite**: MuJoCoベースの物理シミュレーションによるロボット環境を提供
- **Robomimic**: 模倣学習アルゴリズム（BC, BCQ, CQL など）や模倣学習用のデータセットを提供
- **目的**: 低次元データや画像ベースのロボットデータセットを用いたポリシー学習と推論を行う

## インストール
参考：
https://robomimic.github.io/docs/introduction/overview.html


```bash
# リポジトリのクローン
git clone https://github.com/yourusername/your-repo.git
cd your-repo

# 仮想環境構築用のMinicondaのインストール
apt install wget
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source $HOME/.bashrc && conda --version

# Python 仮想環境の作成
conda create -n robomimic_venv python=3.10
conda activate robomimic_venv

# robomimicのインストール
cd robomimic
pip install -e .

# robosuiteのインストール
cd robosuite
pip install -r requirements.txt

# 必要なパッケージのインストール
# Install system libraries
apt update
apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# For OSMesa (for CPU rendering)
apt install -y libosmesa6-dev
conda install -c conda-forge libstdcxx-ng

pip install ncps
pip install -r /work/requirements.txt

```

## セットアップ
### データセットのダウンロード
全てのタスクのダウンロード
```bash
python /work/robomimic/robomimic/scripts/download_datasets.py --tasks all
```
データセットのhdf5ファイルは/work/robomimic/datasets下におく

### configファイルの作成
論文の再現実験用のconfigファイルは以下のコマンドで取得できる
```bash
python /work/robomimic/robomimic/scripts/generate_paper_configs.py
```
configファイルは/work/robomimic/robomimic/exps下におく
ここで学習や推論の細かい設定を行う

### 学習のテンプレート
```bash
python train.py --config yourconfig.json --dataset yourdataset.hdf5
```



