# Robosuite + Robomimic + LTC 実験リポジトリ

MuJoCo ロボットシミュレーション上で、LTC (Liquid Time-Constant) / CfC などのニューラルネットワークを用いた **模倣学習** を行う実験環境です。

---

## 目次

1. [全体像](#全体像)
2. [ディレクトリ構成](#ディレクトリ構成)
3. [セットアップ](#セットアップ)
4. [参考リンク・引用](#参考リンク引用)

---

## 全体像

本リポジトリは 3 つのサブプロジェクトで構成されています。
各リポジトリは基本的にOSSを自分で実験用に改変したものになります。robosuit/については改変していません。

| サブプロジェクト | 役割 | 元リポジトリ |
|---|---|---|
| `robosuite/` | MuJoCo ロボット環境 | [ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite) |
| `robomimic/` | 学習アルゴリズム + デモデータセット管理 | [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic) |
| `liquid_time-constant_networks/` | LTC/NCPモデル | [mlech26I/ncps](https://github.com/mlech26l/ncps?tab=readme-ov-file) |

---

## ディレクトリ構成

```text
/work
├── README.md                        ← 本ファイル
├── INSTALL.md                       ← インストールガイド
├── requirements.txt                 ← robomimic_venv仮想環境における依存環境（robotタスク向け）
├── requirements_base.txt            ← 通常の依存環境（通常タスク向け）
|
├── liquid_time-constant_networks/      
│   
├── robomimic/                           
│
└── robosuite/       
```  

## セットアップ

> 詳細な手順やトラブルシューティングは `INSTALL.md` を参照してください。
うまくいかない場合には下に記載した公式ページを参照してください。

### 前提

- OS: Ubuntu 系 Linux
- Python: 3.10 推奨
- Conda（Miniconda）

---

## 参考リンク・引用

- Robomimic: https://robomimic.github.io/docs/introduction/overview.html
- Robosuite: https://robosuite.ai/docs/overview.html
- NCP: https://www.nature.com/articles/s42256-020-00237-3

```bibtex
@article{hasani2020liquid,
  title={Liquid time-constant networks},
  author={Hasani, Ramin and Lechner, Mathias and Amini, Alexander and Rus, Daniela and Grosu, Radu},
  journal={arXiv preprint arXiv:2006.04439},
  year={2020}
}
```
```bibtex
@inproceedings{robosuite2020,
  title={robosuite: A Modular Simulation Framework and Benchmark for Robot Learning},
  author={Yuke Zhu and Josiah Wong and Ajay Mandlekar and Roberto Mart\'{i}n-Mart\'{i}n and Abhishek Joshi and Kevin Lin and Abhiram Maddukuri and Soroush Nasiriany and Yifeng Zhu},
  booktitle={arXiv preprint arXiv:2009.12293},
  year={2020}
}
```
```bibtex
@inproceedings{robosuite2020,
  title={robosuite: A modular simulation framework and benchmark for robot learning},
  author={Zhu, Yuke and Wong, Josiah and Mandlekar, Ajay and others},
  booktitle={arXiv preprint arXiv:2009.12293},
  year={2020}
}
```




