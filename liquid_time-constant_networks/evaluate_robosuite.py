#!/usr/bin/env python
"""
Robosuite rollout evaluator using trained NCP/LTC model weights.
Measures average reward, steps, inference time, FPS, success rate, GPU memory,
records reward std, and optionally saves rollout video and writes CSV summary.

Usage example:
python evaluate_robosuite.py --checkpoint /path/to/last.ckpt --num_rollouts 5 --max_steps 400 \
    --device cuda --save_video --video_file rollout.mp4 --csv_out results.csv --seed 1
"""
import argparse
import time
import os
import csv
from pathlib import Path
import numpy as np
import torch

try:
    import robosuite as suite
except Exception as e:
    raise ImportError("robosuite required: pip install robosuite. Error: " + str(e))

# Import training module's Lightning class. Must be on PYTHONPATH.
from train import SequenceLightningModule

# --- SUCCESS CRITERION ---
# "aで成功判定" の指定に基づき、ここでは「キューブ高さ（z） > THRESHOLD」を成功と判定します。
SUCCESS_Z_THRESHOLD = 0.15  # 必要に応じて調整

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt)")
    p.add_argument("--num_rollouts", type=int, default=10, help="Number of rollouts")
    p.add_argument("--max_steps", type=int, default=400, help="Max steps per rollout")
    p.add_argument("--device", type=str, default="cuda", help="Device for model (cuda or cpu)")
    p.add_argument("--save_video", action="store_true", help="Save video of first rollout")
    p.add_argument("--video_file", type=str, default="rollout.mp4", help="Output video file path")
    p.add_argument("--render", action="store_true", help="Render to screen (requires display)")
    p.add_argument("--csv_out", type=str, default=None, help="CSV file to append results (optional)")
    p.add_argument("--seed", type=int, default=0, help="seed for logging/recording")
    return p.parse_args()

def get_object_height_from_obs(obs):
    """
    Try to infer object's z coordinate from observation dict.
    Priorities:
      - 'cube_pos' (robosuite default low-dim)
      - 'object-state' (robomimic style): try to extract pos (first 3 elements)
      - 'gripper_to_cube_pos' vector + gripper pos (if present) — fallback not implemented
    Returns z (float) or None if not found.
    """
    if "cube_pos" in obs:
        z = float(obs["cube_pos"][2])
        return z
    if "object-state" in obs:
        arr = np.asarray(obs["object-state"])
        if arr.size >= 3:
            return float(arr[2])
    # some datasets use 'object' or 'object_pos'
    if "object" in obs:
        arr = np.asarray(obs["object"])
        if arr.size >= 3:
            return float(arr[2])
    return None

def write_csv_row(csv_path: str, row: dict):
    """Append a row to csv. Write header if file not exists."""
    csv_path = Path(csv_path)
    header = list(row.keys())
    exists = csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model (Lightning) from checkpoint
    print(f"[INFO] Loading model checkpoint: {args.checkpoint}")
    model = SequenceLightningModule.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(device)

    # create robosuite env.
    # If saving video, enable offscreen rendering + camera obs.
    use_camera_obs = bool(args.save_video)
    try:
        env = suite.make(
            env_name="Lift",
            robots="Panda",
            has_renderer=(args.render and not args.save_video),
            has_offscreen_renderer=args.save_video,
            use_camera_obs=use_camera_obs,
            camera_names="frontview" if use_camera_obs else None,
            camera_heights=480 if use_camera_obs else None,
            camera_widths=640 if use_camera_obs else None,
            reward_shaping=True,
            use_object_obs=True,
            horizon=args.max_steps,
            control_freq=20,
        )
    except Exception as e:
        # If offscreen initialisation fails (EGL issues), fall back to no-video mode and warn.
        print("[WARN] Failed to create env with offscreen renderer (video disabled). Error:", e)
        use_camera_obs = False
        env = suite.make(
            env_name="Lift",
            robots="Panda",
            has_renderer=args.render,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            reward_shaping=True,
            use_object_obs=True,
            horizon=args.max_steps,
            control_freq=20,
        )

    # metrics accumulators
    rollout_rewards = []
    rollout_steps = []
    inference_times = []
    gpu_mem_list = []
    successes = []

    seq_len = 10  # fixed sequence length used at training
    print(f"✅ Starting {args.num_rollouts} rollouts (seq_len={seq_len})")

    # for video
    saved_video_path = None

    for r in range(args.num_rollouts):
        obs_history = []
        obs = env.reset()
        total_reward = 0.0
        state = None
        frames = []  # collect frames if needed

        for step in range(args.max_steps):
            # build observation vector consistent with training
            # try common keys
            try:
                obs_vec = np.concatenate([
                    obs.get("robot0_eef_pos", np.zeros(3)),
                    obs.get("robot0_eef_quat", np.zeros(4)),
                    obs.get("robot0_gripper_qpos", np.zeros(1)),
                    obs.get("object-state", obs.get("object", np.zeros(3)))
                ]).astype(np.float32)
            except Exception:
                # fallback: flatten and concatenate all numeric arrays (best-effort)
                parts = []
                for k, v in obs.items():
                    try:
                        a = np.asarray(v).ravel()
                        parts.append(a)
                    except Exception:
                        continue
                if len(parts) == 0:
                    raise RuntimeError("Unable to construct observation vector from env.obs")
                obs_vec = np.concatenate(parts).astype(np.float32)

            obs_history.append(obs_vec)
            if len(obs_history) < seq_len:
                seq = [obs_history[0]] * (seq_len - len(obs_history)) + obs_history
            else:
                seq = obs_history[-seq_len:]

            obs_tensor = torch.from_numpy(np.stack(seq)).unsqueeze(0).to(device)  # (1, L, D)

            # measure GPU memory & inference time
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.time()
            with torch.no_grad():
                # prefer rollout_forward if available
                if hasattr(model, "rollout_forward"):
                    action_seq, state = model.rollout_forward(obs_tensor, state)
                else:
                    # fallback: try calling model directly expecting it to accept (obs_tensor,)
                    try:
                        out = model(obs_tensor)
                        # model may return (actions, ...) or only actions
                        if isinstance(out, tuple) or isinstance(out, list):
                            action_seq = out[0]
                        else:
                            action_seq = out
                    except Exception as e:
                        raise RuntimeError("Model cannot be called in this evaluator. Implement `rollout_forward` or make model callable. Err: " + str(e))
            elapsed = time.time() - t0
            inference_times.append(elapsed)
            if device.type == "cuda":
                gpu_mem_list.append(torch.cuda.max_memory_allocated(device) / (1024 ** 2))  # MB
            else:
                gpu_mem_list.append(0.0)

            # take last step action
            action = action_seq[0, -1, :].cpu().numpy()
            action = np.clip(action, -1.0, 1.0)

            obs, reward, done, info = env.step(action)
            total_reward += float(reward)

            # capture frame for video if requested (camera obs key depends on robosuite)
            if args.save_video and use_camera_obs:
                # robosuite returns camera obs keyed by camera name
                # e.g., obs['frontview'] is image array HxWx3
                cam_key = "frontview"
                if cam_key in obs:
                    frames.append(obs[cam_key])
                else:
                    # fallback: check keys for any ndarray with shape (H,W,3)
                    for k, v in obs.items():
                        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[2] in (3, 4):
                            frames.append(v)
                            break

            if args.render:
                env.render()

            if done:
                break

        # end of one rollout
        rollout_rewards.append(total_reward)
        rollout_steps.append(step + 1)

        # success detection using object z coordinate
        z = get_object_height_from_obs(obs)
        is_success = False
        if z is not None:
            is_success = float(z) > SUCCESS_Z_THRESHOLD
        successes.append(1 if is_success else 0)

        print(f"Rollout {r+1}/{args.num_rollouts}: Reward={total_reward:.3f}, Steps={step+1}, Success={is_success}")

        # save first rollout video if requested
        if args.save_video and use_camera_obs and r == 0 and len(frames) > 0:
            try:
                import imageio
                imageio.mimsave(args.video_file, frames, fps=20)
                saved_video_path = args.video_file
                print(f"✅ Video saved: {saved_video_path}")
            except Exception as e:
                print("[WARN] Failed to save video:", e)

    env.close()

    # compute metrics
    rollout_rewards = np.array(rollout_rewards, dtype=np.float32)
    rollout_steps = np.array(rollout_steps, dtype=np.int32)
    inference_times = np.array(inference_times, dtype=np.float32) if len(inference_times) > 0 else np.array([0.0], dtype=np.float32)
    gpu_mem_list = np.array(gpu_mem_list, dtype=np.float32) if len(gpu_mem_list) > 0 else np.array([0.0], dtype=np.float32)

    avg_reward = float(np.mean(rollout_rewards))
    reward_std = float(np.std(rollout_rewards))
    avg_steps = float(np.mean(rollout_steps))
    avg_infer = float(np.mean(inference_times)) if inference_times.size > 0 else 0.0
    infer_std = float(np.std(inference_times)) if inference_times.size > 0 else 0.0
    fps = (1.0 / avg_infer) if avg_infer > 0 else 0.0
    success_rate = float(np.mean(successes) * 100.0)
    avg_gpu_mem = float(np.mean(gpu_mem_list))
    peak_gpu_mem = float(np.max(gpu_mem_list) if gpu_mem_list.size > 0 else 0.0)

    # print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print(f"Checkpt: {args.checkpoint}")
    print(f"Avg Reward: {avg_reward:.3f} ± {reward_std:.3f}")
    print(f"Reward Range: [{float(np.min(rollout_rewards)):.3f}, {float(np.max(rollout_rewards)):.3f}]")
    print(f"Avg Steps: {avg_steps:.1f}")
    print(f"Avg Inference Time: {avg_infer*1000:.2f} ms ± {infer_std*1000:.2f} ms")
    print(f"Approx FPS: {fps:.2f}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Avg GPU Mem (MB): {avg_gpu_mem:.2f}")
    print(f"Peak GPU Mem (MB): {peak_gpu_mem:.2f}")
    if saved_video_path:
        print(f"Saved Video: {saved_video_path}")
    print("="*60)

    # optionally write CSV row
    if args.csv_out:
        row = {
            "seed": args.seed,
            "success_rate": f"{success_rate:.3f}",
            "avg_reward": f"{avg_reward:.6f}",
            "reward_std": f"{reward_std:.6f}",
            "avg_steps": f"{avg_steps:.3f}",
            "avg_infer_ms": f"{avg_infer*1000:.6f}",
            "fps": f"{fps:.3f}",
            "avg_gpu_mem_mb": f"{avg_gpu_mem:.3f}",
            "peak_gpu_mem_mb": f"{peak_gpu_mem:.3f}",
            "checkpoint_path": str(args.checkpoint),
            "video_path": str(saved_video_path) if saved_video_path else "",
        }
        try:
            write_csv_row(args.csv_out, row)
            print(f"[INFO] Appended results to CSV: {args.csv_out}")
        except Exception as e:
            print("[WARN] Failed to write CSV:", e)

if __name__ == "__main__":
    main()
