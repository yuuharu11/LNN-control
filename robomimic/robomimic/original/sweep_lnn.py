import os
import json
import tempfile
import subprocess
import wandb

def main():
    run = wandb.init(
        project="robomimic_lnn_sweeps",
        settings=wandb.Settings(init_timeout=120),
        resume="never"
    )
    cfg = run.config  # agent から受け取る設定にアクセス

    # ベース設定を読み込み（LNN 用）
    base_cfg_path = "/work/robomimic/robomimic/exps/templates/bc_lnn.json"
    with open(base_cfg_path, 'r') as f:
        base_config = json.load(f)

    # --- LNN パラメータの上書き ---
    units = int(getattr(cfg, "units", 128))
    seq_length = int(getattr(cfg, "seq_length", 20))
    horizon = int(getattr(cfg, "horizon", 10))
    ode_unfolds = int(getattr(cfg, "ode_unfolds", 5))
    epsilon = float(getattr(cfg, "epsilon", 0.1))

    # ✅ seq_length ≥ horizon の制約チェック
    if seq_length < horizon:
        raise ValueError(
            f"seq_length ({seq_length}) must be >= horizon ({horizon})"
        )

    # LNN アルゴリズム設定
    base_config.setdefault("algo", {})
    base_config["algo"].setdefault("lnn", {})
    base_config["algo"]["lnn"]["units"][1]["units"] = units
    base_config["algo"]["lnn"]["horizon"] = horizon
    base_config["algo"]["lnn"]["ode_unfolds"] = ode_unfolds
    base_config["algo"]["lnn"]["epsilon"] = epsilon

    # 学習設定
    base_config["train"]["seq_length"] = seq_length

    lr = float(getattr(cfg, "learning_rate", 1e-4))
    l2 = float(getattr(cfg, "l2_weight", 0.0))

    base_config["algo"].setdefault("optim_params", {}).setdefault("policy", {})
    base_config["algo"]["optim_params"]["policy"]["learning_rate"] = {
        "initial": lr
    }
    base_config["algo"]["optim_params"]["policy"].setdefault("regularization", {})
    base_config["algo"]["optim_params"]["policy"]["regularization"]["L2"] = l2

    # run 名の作成
    run_name = (
        f"lnn_u{units}_seq{seq_length}_h{horizon}_"
        f"ode{ode_unfolds}_eps{epsilon:.2f}_"
        f"lr{lr:.0e}_l2{l2:.0e}"
    )

    base_config["train"]["num_epochs"] = 50
    base_config["experiment"]["save"]["every_n_epochs"] = 25
    base_config["experiment"]["rollout"]["rate"] = 25

    # ✅ wandb 設定
    base_config["experiment"].setdefault("logging", {})
    base_config["experiment"]["logging"]["log_wandb"] = True
    base_config["experiment"]["logging"]["wandb_proj_name"] = "robomimic_lnn_sweeps"
    base_config["experiment"]["logging"]["wandb_run_name"] = run_name
    base_config["experiment"]["name"] = f"lnn_sweep/{run_name}"

    # ✅ dataset の決定
    dataset_path = os.environ.get("SWEEP_DATASET", None)
    if dataset_path is None:
        dataset_path = base_config.get("train", {}).get("data", None)
    if not dataset_path:
        raise RuntimeError(
            "Dataset path not specified. Set SWEEP_DATASET env or train.data in bc_lnn.json."
        )

    # 一時ファイルに書き出し
    fd, tmp_path = tempfile.mkstemp(prefix="sweep_config_lnn_", suffix=".json")
    os.close(fd)
    with open(tmp_path, 'w') as f:
        json.dump(base_config, f, indent=4)

    try:
        cmd = [
            "python3",
            "/work/robomimic/robomimic/scripts/train.py",
            "--config", tmp_path,
            "--dataset", dataset_path
        ]
        env = os.environ.copy()
        env.pop("WANDB_RUN_ID", None)
        env.pop("WANDB_RESUME", None)
        subprocess.run(cmd, check=True, env=env)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        wandb.finish()

if __name__ == "__main__":
    main()
