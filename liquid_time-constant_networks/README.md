# Liquid Time-Constant Networks — 学習・評価フレームワーク

LTC (Liquid Time-Constant) / CfC (Closed-form Continuous-depth) などの時系列モデルを、  
**PyTorch Lightning + Hydra** で統一的に学習・評価するためのコードです。

---

## 目次

1. [できること](#できること)
2. [ディレクトリ構成](#ディレクトリ構成)
3. [前提条件](#前提条件)
4. [学習の実行方法](#学習の実行方法)
5. [設定システム（Hydra）の仕組み](#設定システムhydraの仕組み)
6. [対応モデル・データセット](#対応モデルデータセット)
7. [評価（ロールアウト）](#評価ロールアウト)
8. [出力の見方](#出力の見方)
9. [カスタマイズ方法](#カスタマイズ方法)

---

## できること

- **時系列分類**: UCI-HAR, MNIST, CIFAR-10, PAMAP2 などのベンチマークデータで LTC / CfC を評価
- **ロボット模倣学習**: robomimic の Lift タスク（低次元）で NCP / CfC / LSTM / MLP 等を比較
- **robosuite ロールアウト評価**: 学習済みチェックポイントを使ってシミュレーション上の成功率を計測
- **ハイパーパラメータスイープ**: シェルスクリプトでユニット数・層数・ODE unfold 数等の組み合わせを一括実行
- **継続学習**: EWC, リプレイ, PNN, PackNet などの手法に対応
- **ノイズ耐性評価**: センサーノイズレベルを変えた条件でのロバスト性テスト

---

## ディレクトリ構成

```text
liquid_time-constant_networks/
├── train.py                         ★ 学習エントリポイント
│
├── configs/                         ← Hydra 設定ファイル群（後述）
│   ├── config.yaml                     ルート設定
│   ├── experiment/                     実験プリセット（モデル×データを一括指定）
│   │   ├── ncp/                          NCP 系（uci_har, mnist, cifar10, pamap2）
│   │   ├── ncp_cfc/                      CfC 系
│   │   ├── lstm/                         LSTM ベースライン
│   │   ├── cnn/                          CNN ベースライン
│   │   ├── rnn/                          RNN ベースライン
│   │   └── robosuite/lift/               ロボット操作タスク（ncp, cfc, lstm, mlp, cnn, rnn）
│   ├── model/                          モデル構造の定義
│   ├── dataset/                        データセットの定義
│   ├── task/                            タスク種別（分類 / 回帰）
│   ├── optimizer/                      オプティマイザ（AdamW 等）
│   ├── scheduler/                      LR スケジューラ（cosine_warmup, plateau）
│   ├── trainer/                        PyTorch Lightning Trainer 設定
│   └── callbacks/                      コールバック設定
│
├── src/                             ← Python ソースコード
│   ├── models/
│   │   ├── ncps/                       LTC / CfC の実装
│   │   │   ├── ltc.py                     LTC モデル
│   │   │   ├── cfc.py                     CfC モデル
│   │   │   ├── lstm.py                    LSTM（ncps 版）
│   │   │   └── cells/                     セルレベルの実装
│   │   │       ├── ltc_cell.py              LTC セル
│   │   │       ├── cfc_cell.py              CfC セル（逐次処理）
│   │   │       ├── cfc_parallel.py          CfC セル（並列処理）
│   │   │       └── wired_cfc_cell.py        ワイヤリング付き CfC セル
│   │   ├── sequence/                   汎用シーケンスモデル（MLP, CNN, RNN）
│   │   ├── baseline/                   ベースライン LSTM
│   │   ├── continual_learning/         継続学習（PNN, PackNet）
│   │   └── wirings/                    NCP ワイヤリング定義
│   ├── dataloaders/                    データローダ
│   │   ├── uci_har.py                    UCI-HAR
│   │   ├── uci_har_cil.py               UCI-HAR（継続学習用）
│   │   ├── uci_har_noise.py             UCI-HAR（ノイズ付き）
│   │   └── robosuite/                   robomimic 用ローダ
│   ├── tasks/                          エンコーダ・デコーダ・損失・メトリクス
│   ├── callbacks/                      カスタムコールバック
│   │   ├── experiment_logger.py          実験ログ（CSV 出力）
│   │   ├── flops_counter.py              FLOPs 計測
│   │   ├── memory_profiler.py            メモリプロファイラ
│   │   ├── weight_visualizer.py          重み可視化
│   │   └── ...
│   └── utils/                          ユーティリティ（最適化, ノイズ生成 等）
│
├── train_scripts/                   ← 実行用シェルスクリプト
│   ├── robomimic/
│   │   └── ncp_lift.sh                  Lift タスクで NCP を学習
│   ├── mnist/
│   │   └── run_ltc.sh                   MNIST で HP スイープ
│   └── uci_har/
│       └── baseline.sh                  UCI-HAR ノイズ耐性評価
│
├── eval_robosuite/                  ← robosuite ロールアウト評価
│   ├── evaluate_robosuite.py           基本ロールアウト評価
│   ├── evaluate_robosuite_with_camera.py   カメラ付き評価（動画保存可）
│   ├── run_trained_agent.py            学習済みエージェント実行
│   └── train_robomimic_*.sh            robomimic 経由でのモデル学習
│
├── outputs/                         ← 学習出力（自動生成）
├── results/                         ← 集計済みメトリクス
├── csv/                             ← 実験結果 CSV
└── ipynb/                           ← 分析用ノートブック・スクリプト
```

---

## 前提条件

親ディレクトリ（`/work`）の `README.md` および `INSTALL.md` に従って環境構築が完了していること。

最低限必要なパッケージ:

- PyTorch >= 2.0
- PyTorch Lightning >= 2.0
- Hydra >= 1.3
- ncps（`pip install ncps`）
- robomimic / robosuite（ロボットタスクを使う場合）

---

## 学習の実行方法

### 方法 1: 用意済みのスクリプトを使う（推奨）

```bash
cd /work/liquid_time-constant_networks

# Robomimic Lift タスクで NCP を学習
./train_scripts/robomimic/ncp_lift.sh

# MNIST で LTC のハイパーパラメータスイープ
./train_scripts/mnist/run_ltc.sh

# UCI-HAR でノイズ耐性ベースライン評価
./train_scripts/uci_har/baseline.sh
```

### 方法 2: train.py を直接実行

```bash
cd /work/liquid_time-constant_networks

# experiment= で実験プリセットを指定（model + dataset + task を一括設定）
python train.py experiment=ncp/uci_har train.seed=1

# robomimic Lift + NCP（エポック数を上書き）
python train.py \
  experiment=robosuite/lift/ncp \
  dataset=robomimic_lift_lowdim \
  train.seed=42 \
  trainer.max_epochs=30

# モデルのハイパーパラメータをコマンドラインで上書き
python train.py \
  experiment=ncp/uci_har \
  model.layer.units.1.units=64 \
  model.n_layers=2 \
  optimizer.lr=0.001
```

### テストのみ実行（学習済みモデルの評価）

```bash
python train.py \
  experiment=ncp/uci_har \
  train.test_only=true \
  train.pretrained_model_path=/path/to/checkpoint.ckpt
```

---

## 設定システム（Hydra）の仕組み

Hydra は複数の YAML ファイルを **合成** して1つの設定を作ります。

### 設定の階層

```text
config.yaml（ルート）
  └── experiment=robosuite/lift/ncp  ← 実験プリセット
        ├── model: ncps_ltc          ← どのモデルを使うか
        ├── dataset: robomimic_lift_lowdim  ← どのデータを使うか
        ├── task: regression         ← 分類 or 回帰
        ├── optimizer: adamw         ← オプティマイザ
        └── scheduler: cosine_warmup ← LR スケジューラ
```

### 仕組みのポイント

1. **experiment YAML を指定するだけで、必要な設定がすべて揃う**
   ```bash
   python train.py experiment=ncp/uci_har
   # → model, dataset, task, optimizer, scheduler が自動で設定される
   ```

2. **コマンドラインでどの値でも上書きできる**
   ```bash
   python train.py experiment=ncp/uci_har model.n_layers=3 optimizer.lr=1e-4
   ```

3. **実行ごとの設定は自動保存される**
   ```text
   outputs/<日時>/.hydra/config.yaml   ← 実行時の最終設定
   outputs/<日時>/.hydra/overrides.yaml ← コマンドラインで上書きした値
   ```

### 新しい実験プリセットを作るには

`configs/experiment/` に YAML を追加するだけです:

```yaml
# configs/experiment/my_experiment.yaml
# @package _global_
defaults:
  - /trainer: default
  - /model: ncps_ltc        # ← 使いたいモデル
  - /loader: default
  - /dataset: uci_har_dil   # ← 使いたいデータセット
  - /task: multiclass_classification
  - /optimizer: adamw
  - /scheduler: cosine_warmup

train:
  monitor: val/accuracy
  mode: max

model:
  d_model: 9
  layer:
    units:
    - name: AutoNCP
    - units: 64
    - output_units: 6

loader:
  batch_size: 128

optimizer:
  lr: 5e-3

trainer:
  max_epochs: 30
```

```bash
python train.py experiment=my_experiment
```

---

## 対応モデル・データセット

### モデル

| モデル | config キー | 説明 |
|---|---|---|
| LTC (NCP) | `ncps_ltc` | Liquid Time-Constant ネットワーク（ODE ベース） |
| CfC | `ncps_cfc` | Closed-form Continuous-depth（LTC の閉形式近似） |
| LSTM | `lstm` | ベースライン LSTM |
| RNN | `rnn` | 標準 RNN |
| CNN | `cnn` / `cnn_har` | 畳み込みベースライン |
| MLP | `mlp` | 全結合ベースライン |

### データセット

| データセット | config キー | タスク | 入出力 |
|---|---|---|---|
| UCI-HAR | `uci_har_dil` | 6クラス分類 | 9軸センサー → 行動ラベル |
| Robomimic Lift | `robomimic_lift_lowdim` | 回帰 | 19次元状態 → 7次元アクション |
| MNIST | (experiment で指定) | 分類 | 28×28 画像 → 数字ラベル |
| CIFAR-10 | `cifar10` | 分類 | 32×32 画像 → クラスラベル |
| PAMAP2 | (experiment で指定) | 分類 | 加速度センサー → 行動ラベル |

---

## 評価（ロールアウト）

学習済みチェックポイントを使って、robosuite シミュレーション上でロボットを動かし、成功率や推論速度を計測します。

```bash
cd /work/liquid_time-constant_networks/eval_robosuite

# 基本評価（成功率・報酬・FPS を計測）
python evaluate_robosuite.py \
  --checkpoint /path/to/last.ckpt \
  --num_rollouts 10 \
  --max_steps 400 \
  --device cuda

# 動画を保存しながら評価
python evaluate_robosuite_with_camera.py \
  --checkpoint /path/to/last.ckpt \
  --save_video --video_file rollout.mp4

# 結果を CSV に記録
python evaluate_robosuite.py \
  --checkpoint /path/to/last.ckpt \
  --csv_out results.csv --seed 1

# シェルスクリプトで一括評価
./eval_robosuite.sh
```

---

## 出力の見方

学習を実行すると、`outputs/` に以下が自動生成されます:

```text
outputs/
└── 2026-02-16/12-34-56-789012/    ← 実行日時
    ├── checkpoints/
    │   └── last-*.ckpt              ← モデルチェックポイント
    ├── .hydra/
    │   ├── config.yaml              ← 最終設定（全パラメータ）
    │   └── overrides.yaml           ← コマンドラインでの上書き分
    └── wandb/                       ← W&B ログ（有効な場合）
```

CSV 結果は `callbacks.experiment_logger.output_file` で指定したパスに出力されます。

---

## カスタマイズ方法

| やりたいこと | やり方 |
|---|---|
| **モデルを追加** | `src/models/` に実装 → `configs/model/` に YAML 追加 |
| **データセットを追加** | `src/dataloaders/` にローダ実装 → `configs/dataset/` に YAML 追加 |
| **コールバックを追加** | `src/callbacks/` に実装 → `configs/callbacks/` に YAML 追加 |
| **実験条件を追加** | `configs/experiment/` に YAML を作成（上記の例を参照） |
| **ロガーを変更** | `config.yaml` の `wandb` セクションを編集（`mode: disabled` で無効化） |
| **継続学習を使う** | `train.replay`, `train.regularization`, `train.architecture` を設定 |