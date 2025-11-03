import os
import json
import tempfile
import subprocess
import wandb

def main():
    run = wandb.init(
        project="robomimic_sweeps",
        settings=wandb.Settings(init_timeout=120),
        resume="never"
        )
    cfg = run.config  # agentから受け取る設定にアクセス

    # ベース設定を読み込み
    base_cfg_path = "/work/robomimic/robomimic/exps/templates/bc.json"
    with open(base_cfg_path, 'r') as f:
        base_config = json.load(f)

    # --- 上書きパラメータ ---
    mlp_num_layers = int(getattr(cfg, "mlp_num_layers", 2))
    mlp_width = int(getattr(cfg, "mlp_width", 256))
    base_config.setdefault("algo", {})
    base_config["algo"]["actor_layer_dims"] = [mlp_width] * mlp_num_layers

    seq_length = int(getattr(cfg, "seq_length", 10))
    base_config["train"]["seq_length"] = seq_length

    lr = getattr(cfg, "learning_rate", 1e-4)
    l2 = getattr(cfg, "l2_weight", 0.0)

    base_config["algo"].setdefault("optim_params", {}).setdefault("policy", {})
    base_config["algo"]["optim_params"]["policy"]["learning_rate"] = {
        "initial": lr
    }
    base_config["algo"]["optim_params"]["policy"].setdefault("regularization", {})
    base_config["algo"]["optim_params"]["policy"]["regularization"]["L2"] = l2

    run_name = f"mlp_l{mlp_num_layers}w{mlp_width}_seq{seq_length}_lr{lr:.0e}_l2{l2:.0e}"

    base_config["train"]["num_epochs"] = 50
    base_config["experiment"]["save"]["every_n_epochs"] = 25
    base_config["experiment"]["rollout"]["rate"] = 25


    # ✅ run IDなどは wandb から取得せず手動設定
    base_config["experiment"].setdefault("logging", {})
    base_config["experiment"]["logging"]["log_wandb"] = True
    base_config["experiment"]["name"] = f"bc_sweep/{run_name}"

    # ✅ dataset の決定
    dataset_path = os.environ.get("SWEEP_DATASET", None)
    if dataset_path is None:
        dataset_path = base_config.get("train", {}).get("data", None)
    if not dataset_path:
        raise RuntimeError("Dataset path not specified. Set SWEEP_DATASET env or train.data in bc.json.")

    # 一時ファイルに書き出し
    fd, tmp_path = tempfile.mkstemp(prefix="sweep_config_", suffix=".json")
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
        os.remove(tmp_path)
        wandb.finish()

if __name__ == "__main__":
    main()
