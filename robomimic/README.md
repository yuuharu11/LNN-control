# robomimic

[robosuite](../robosuite/) 環境上でのロボット模倣学習フレームワーク。
[ARISE Initiative](https://github.com/ARISE-Initiative) が開発・公開しており、本プロジェクトではこれをベースに **Liquid Neural Network (LNN)** を用いたポリシー学習を行っている。

> **公式リソース**: [Homepage](https://robomimic.github.io/) / [論文](https://arxiv.org/abs/2108.03298) / [公式ドキュメント](https://robomimic.github.io/docs/introduction/overview.html) / [GitHub](https://github.com/ARISE-Initiative/robomimic)

> **バージョン**: v0.5.0（本リポジトリに含まれるもの）

---

## 概要

robomimic は、ロボット操作タスクにおけるオフラインの模倣学習・強化学習を行うためのフレームワークである。

- **標準化されたデータセット**: robosuite 環境で収集された人間デモンストレーションデータ
- **多数の学習アルゴリズム**: BC, BC-RNN, BC-Transformer, BCQ, CQL, IQL, TD3+BC, Diffusion Policy 等
- **モジュール設計**: 観測モダリティ・エンコーダ・ポリシーネットワークの組み合わせを柔軟に変更可能
- **ハイパーパラメータ管理**: JSON ベースの設定ファイルとスイープ機能

### 本プロジェクト独自の拡張

公式 robomimic に加え、以下のカスタム実装が追加されている。

- **BC_LNN / BC_LNN_GMM**: Liquid Neural Network (CfC / LTC) を用いた Behavioral Cloning アルゴリズム
- **`models/lnn/`**: CfC セル、LTC セル、量子化対応 LTC セル、ワイヤリング構成などの LNN モジュール群
- **各タスク向け学習・評価スクリプト**: `myscript/` 配下にタスクごとの実験スクリプトを整備
- **量子化・ノイズ注入実験**: 量子化やセンサノイズ注入に関する評価スクリプト・ノートブック

---

## ディレクトリ構成

```
robomimic/
├── setup.py                  # パッケージインストール設定
├── Dockerfile                # Docker 環境構築
├── robomimic/                # メインパッケージ（後述）
├── examples/                 # 使用例・チュートリアル
├── myscript/                 # 本プロジェクト独自の実験スクリプト
├── datasets/                 # デモンストレーションデータセット
├── tests/                    # テストファイル
└── docs/                     # 公式ドキュメント
```

### `robomimic/`（メインパッケージ）の構成

| ディレクトリ | 内容 |
|---|---|
| `algo/` | 学習アルゴリズムの実装（BC, BCQ, CQL, IQL, TD3+BC, Diffusion Policy, **BC_LNN** 等） |
| `config/` | アルゴリズムごとの設定クラス。JSON テンプレートも同梱 |
| `models/` | ネットワークアーキテクチャ（ポリシー、値関数、観測エンコーダ、VAE、Transformer、**LNN**） |
| `models/lnn/` | **Liquid Neural Network モジュール**（CfC, LTC, 量子化 LTC, ワイヤリング） |
| `envs/` | 環境インターフェース（robosuite, Gym, iGibson） |
| `scripts/` | 学習・評価・データ処理のスクリプト群 |
| `utils/` | ユーティリティ（データセット I/O, ログ, 損失関数, テンソル操作, 可視化等） |
| `exps/` | 実験設定テンプレート（JSON）とパラメータスイープ設定 |
| `wandb_sweep/` | W&B によるハイパーパラメータスイープ設定 |

---

## 利用可能なアルゴリズム

### 模倣学習

| アルゴリズム | クラス名 | 説明 |
|---|---|---|
| BC | `BC`, `BC_Gaussian`, `BC_GMM`, `BC_VAE` | Behavioral Cloning（各種出力分布） |
| BC-RNN | `BC_RNN`, `BC_RNN_GMM` | RNN ベースの BC |
| **BC-LNN** | `BC_LNN`, `BC_LNN_GMM` | **Liquid Neural Network ベースの BC（本プロジェクト独自）** |
| BC-Transformer | `BC_Transformer` | Transformer ベースの BC |
| HBC | `HBC` | Hierarchical BC |
| Diffusion Policy | `DiffusionPolicyUNet` | 拡散モデルベースのポリシー |

### オフライン強化学習

| アルゴリズム | クラス名 | 説明 |
|---|---|---|
| BCQ | `BCQ` | Batch-Constrained Q-Learning |
| CQL | `CQL` | Conservative Q-Learning |
| IQL | `IQL` | Implicit Q-Learning |
| TD3+BC | `TD3_BC` | TD3 with BC regularization |
| IRIS | `IRIS` | 階層型オフライン RL |

---

## LNN モジュール（`models/lnn/`）

本プロジェクトの中核である Liquid Neural Network の実装。

| ファイル | 内容 |
|---|---|
| `cfc.py` | Closed-Form Continuous-time (CfC) ネットワーク |
| `ltc.py` | Liquid Time-Constant (LTC) ネットワーク |
| `lstm.py` | LSTM（mixed_memory用） |
| `sequence_module.py` | シーケンスモジュール共通基盤 |
| `wirings.py` | ニューロン間のワイヤリング構成 |
| `cells/cfc_cell.py` | CfC セルの実装 |
| `cells/ltc_cell.py` | LTC セルの実装 |
| `cells/wired_cfc_cell.py` | ワイヤリング付き CfC セル |

---

## 対応タスク・データセット

`datasets/` 配下に以下のタスクのデモンストレーションデータが格納されている。

| タスク | ディレクトリ | 説明 |
|---|---|---|
| Lift | `datasets/lift/` | 立方体を持ち上げる |
| Can | `datasets/can/` | 缶を指定位置に置く |
| Square | `datasets/square/` | ナットをペグに嵌める |
| Tool Hang | `datasets/tool_hang/` | 工具をフックに掛ける |
| Transport | `datasets/transport/` | 双腕でオブジェクトを運搬する |

---

## 主なスクリプト

### `robomimic/scripts/`（公式スクリプト）

| スクリプト | 用途 |
|---|---|
| `train.py` | モデルの学習（メインエントリポイント） |
| `run_trained_agent.py` | 学習済みエージェントの評価・ロールアウト |
| `playback_dataset.py` | デモデータの再生・可視化 |
| `download_datasets.py` | HuggingFace からのデータセットダウンロード |
| `dataset_states_to_obs.py` | 環境状態から観測量を生成 |
| `split_train_val.py` | Train/Validation 分割 |
| `hyperparam_helper.py` | ハイパーパラメータスイープ用設定生成 |
| `get_dataset_info.py` | データセット情報の表示 |

### `myscript/`（本プロジェクト独自）

タスクごとに学習・評価用のシェルスクリプトを整備。

| ディレクトリ | 内容 |
|---|---|
| `core/` | 全タスク共通の学習スクリプト（`can.sh`, `lift.sh`, `square.sh`, `tool_hang.sh`, `transport.sh`） |
| `can/`, `lift/`, `square/`, `tool_hang/`, `transport/` | タスク別のベースライン・NCP 実験スクリプト |
| `eval/baseline/` | ベースライン評価スクリプト |
| `eval/noise-injection/` | ガウスノイズ・シフトノイズ注入実験 |
| `eval/quantize/` | 量子化実験 |
| `ipynb/` | 分析・可視化用 Jupyter ノートブック |

---

## 基本的な使い方

### 学習の実行

```bash
python robomimic/scripts/train.py --config <config.json>
```

設定テンプレートは `robomimic/exps/` にアルゴリズムごとの JSON が用意されている（`bc.json`, `bc_lnn.json`, `bc_rnn.json`, `bcq.json` 等）。
JSONファイルを書き換えることで学習時のモデル構造や設定（エポック数やバッチ数）などを変更することが可能。また、wandbやtensorboardのloggerにより学習を記録することも可能。
基本的に設定を変えたいときはこのテンプレートを新たに作成して変更するか、train関数の引数を変更することで対応。
デフォルトでは50epochごとにvalidationが行われ、チェックポイントパスが`/robomimic/bc_trained_models/`に保存される。なお、タスクの成功率が最もよいときに明示的にファイル名に記録される仕組みになっている。
この頻度やタスクの推論回数は変更することが可能。

### 学習済みモデルの評価

```bash
python robomimic/scripts/run_trained_agent.py \
    --agent <model.pth> \
    --n_rollouts 50 \
    --horizon 400
```
horizonは1回の推論の上限timestep数。これが長すぎるとうまく学習できていないときに推論時間が長くなってしまう。n_rolloutsは推論回数。今回の実験では100回としたが, 推論時間を短くしたい場合には小さくする。

### 量子化
量子化はNCPにのみ対応している。各演算ブロックの入力や重みに対して個別に量子化を行うことができ、それぞれ以下のようになっている。
| 引数 | ブロック |
|---|---|
| `--CAM_quantization` | CAMブロックの入力に対する量子化 |
| `--LUT_quantization` | LUTに保存する値に対する量子化 |
| `--weight_quantization` | CiMの重みに対する量子化 |
| `--digital_SRAM_quantization` | SRAM Bufferに保存する値に対する量子化 |
| `--digital_RRAM_quantization` | ReRAMに保存する値に対する量子化 |
| `--ADC_quantization` | ADCを模したAD変換時に生じる量子化 |
| `--DAC_quantization` | DACを模したDA変換時に生じる量子化 |

なお、量子化にあたって範囲を事前に決定する必要がある部分に関してはキャリブレーションを行うことで決定した量子化範囲を用いる。これには以下の引数を設定する必要がある。
| 引数 | ブロック |
|---|---|
| `--calibration_times` | 推論前に行うキャリブレーションの回数 |
| `--calibration_path` | キャリブレーション結果を保存した（する）パス |
| `--calibration_percentile` | キャリブレーションの結果、クリッピングする範囲 |

量子化を行うことで得られるモデルサイズの削減結果や精度低下などは`/robomimic/myscript/ipynb/quantization.ipynb`などで可視化・評価する。

### エラー注入
エラー注入はNCPのMVMブロック（アナログCiM演算）にのみ対応している。CiMに保存することが想定される重みに対してデバイスエラーを模したガウシアンノイズ、シフトノイズを加えるというものである。具体的なコードは`/robomimic/myscript/eval/noise_injection/`を参照されたい。
| 引数 | ブロック |
|---|---|
| `--gaussian` | ガウシアンノイズの大きさ（std [n.s.]） |
| `--shift` | シフトノイズの大きさ（std [n.s.]） |
| `--cell_bits` | 1セルに保存するビット数（MLC想定）|


## 引用

```bibtex
@inproceedings{robomimic2021,
  title={What Matters in Learning from Offline Human Demonstrations for Robot Manipulation},
  author={Ajay Mandlekar and Danfei Xu and Josiah Wong and Soroush Nasiriany and Chen Wang and Rohun Kulkarni and Li Fei-Fei and Silvio Savarese and Yuke Zhu and Roberto Mart\'{i}n-Mart\'{i}n},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2021}
}
```