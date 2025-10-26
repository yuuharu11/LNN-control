# Robosuite & Robomimic 統合リポジトリ

このリポジトリは [robosuite](https://github.com/ARISE-Initiative/robosuite) と [robomimic](https://github.com/ARISE-Initiative/robomimic) を統合し、末梢神経モデルの実験を行う環境です。  
ロボット操作タスクの学習や推論を簡単に試すことができます(予定)。

## 概要

- **Robosuite**: 物理シミュレーションによるロボット環境を提供。
- **Robomimic**: 模倣学習アルゴリズム（BC, BCQ, CQL など）やデータセットを提供。
- **目的**: 低次元データや画像ベースのロボットデータセットを用いた政策学習と評価を手軽に行う。

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/your-repo.git
cd your-repo

# Python 仮想環境の作成
python3 -m venv robomimic_venv
source robomimic_venv/bin/activate

# 必要なパッケージのインストール
pip install -r requirements.txt

```
## 報酬系
/robosuite/enviroments/manipulation/下のpythonファイルにてreward関数で定義
例）Liftタスク（1ステップ）
・Reaching：グリッパーと物体の距離に応じて定義（0-1.0）
・Grasping：物体の把持（0.25）
・Lifting：物体がテーブルから4cm以上浮いている（0.5）
・High Lift：物体がテーブルから15cm以上浮いている（1.0）
・Success：タスクが成功条件を満たしている（2.0）

