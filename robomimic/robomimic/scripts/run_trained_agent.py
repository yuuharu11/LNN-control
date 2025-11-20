"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 \
        --camera_names agentview robot0_eye_in_hand 

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs 

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""
import argparse
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy
import psutil
import time
import csv
import os

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy

class PerformanceMonitor:
    """ monitor gpu/CPU memory usage and time taken for rollouts """
    def __init__(self, device):
        self.device = device
        self.use_gpu = torch.cuda.is_available() and ("cuda" in str(device))
    
    def gpu_memory_mb(self):
        """ gpu memory usage in MB """
        if self.use_gpu:
            return torch.cuda.memory_allocated(self.device) / 1024.0 / 1024.0
        else:
            return 0.0

    def cpu_memory_mb(self):
        """ cpu memory usage in MB """
        process=psutil.Process()

        return process.memory_info().rss / 1024.0 / 1024.0
    
class LNNStateRecorder:
    """Recorder for LNN states during rollout, with optional hook-based recording."""
    def __init__(self, enable=False, device='cuda'):
        self.enable = enable
        self.device = device
        self.states = []
        self.hparams = None
        self.initialized = False
        self.hooks = []

    def _hook_fn(self, module, input, output):
        if not self.enable:
            return

        # forward returns: (readout, new_state)
        if not isinstance(output, tuple) or len(output) < 2:
            print("❌ Output is not a valid (readout, state) tuple.")
            return

        hidden = output[1]

        # Case A: normal (tensor)
        if isinstance(hidden, torch.Tensor):
            hs = hidden.detach().cpu().numpy()
            self.states.append(hs)
            print(f"✅ Saved hidden state tensor, shape={hs.shape}, total={len(self.states)}")
            return

        # Case B: mixed memory (tuple of tensors)
        if isinstance(hidden, tuple):
            # Expecting (h_state, c_state)
            hs_list = []
            for i, h in enumerate(hidden):
                if not isinstance(h, torch.Tensor):
                    print(f"❌ hidden[{i}] is not Tensor: {type(h)}")
                    return
                hs_list.append(h.detach().cpu().numpy())
            # 保存はまとめて1つのエントリとして
            self.states.append(hs_list)
            print(f"✅ Saved mixed hidden state (h_state, c_state). total={len(self.states)}")
            return

        print("❌ Hidden state is neither Tensor nor tuple of Tensors.")

    def attach_hooks(self, lnn_model):
        """Attach forward hooks to all LNN layers."""
        if not self.enable or self.initialized:
            return
        
        if lnn_model is not None:
            h = lnn_model.register_forward_hook(self._hook_fn)
            self.hooks.append(h)
            print(f"  [LNN Recorder] Registered hook on LNN model")
        else:
            print(f"  [LNN Recorder] Warning: No LNN model found")
        
        self.initialized = True

    def detach_hooks(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def capture_hparams(self, lnn_model):
        """Capture hyperparameters from an LTC LNN model."""
        if not self.enable or self.hparams is not None:
            return

        hparams = {}

        cell = lnn_model.rnn_cell

        # 基本情報
        hparams['state_size'] = getattr(cell, 'state_size', None)
        hparams['sensory_size'] = getattr(cell, 'sensory_size', None)
        hparams['motor_size'] = getattr(cell, 'motor_size', None)
        hparams['output_size'] = getattr(cell, 'output_size', None)

        params_of_interest = [
            "gleak", "vleak", "cm", "w", "sigma", "mu", "erev",
            "sensory_w", "sensory_sigma", "sensory_mu", "sensory_erev",
            "input_w", "input_b", "output_w", "output_b", "sparsity_mask", "sensory_sparsity_mask"
        ]
        params = {}
        for k in params_of_interest:
            if k in cell._params:
                v = cell._params[k]
                if isinstance(v, torch.nn.Parameter):
                    params[k] = v.detach().cpu().numpy()
                else:
                    params[k] = deepcopy(v)
        hparams['params'] = params

        # ワイヤリング情報
        if hasattr(cell, '_wiring'):
            wiring = cell._wiring
            hparams['wiring_units'] = getattr(wiring, 'units', None)
            hparams['input_dim'] = getattr(wiring, 'input_dim', None)
            hparams['output_dim'] = getattr(wiring, 'output_dim', None)
            hparams['adjacency_matrix'] = getattr(wiring, 'adjacency_matrix', None)
            hparams['sensory_adjacency_matrix'] = getattr(wiring, 'sensory_adjacency_matrix', None)

        self.hparams = hparams

    def _to_serializable(self, obj):
        """Recursively convert numpy arrays and other types to JSON-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_serializable(v) for v in obj]
        return obj

    def save_to_hdf5(self, path):
        """Save recorded states and hparams to HDF5."""
        if not self.enable:
            return
        print(f"  [LNN Recorder] Saving {len(self.states)} states to: {path}")

        # ディレクトリが存在しない場合は作成
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with h5py.File(path, "w") as f:
            # 状態の保存
            if self.states:
                states_grp = f.create_group("states")
                for i, state in enumerate(self.states):
                    states_grp.create_dataset(
                            f"step_{i}", 
                            data=state, 
                            compression='gzip'
                        )
                    
            # hparamsの保存
            if self.hparams:
                hparams_grp = f.create_group("hparams")
                
                # パラメータを個別に保存
                if 'params' in self.hparams:
                    params_grp = hparams_grp.create_group("params")
                    for k, v in self.hparams['params'].items():
                        if isinstance(v, np.ndarray):
                            params_grp.create_dataset(k, data=v, compression='gzip')
                    print(f"    Saved {len(self.hparams['params'])} parameters")
                
                # その他のメタデータ
                for k, v in self.hparams.items():
                    if k != 'params' and v is not None:
                        if isinstance(v, np.ndarray):
                            hparams_grp.create_dataset(k, data=v, compression='gzip')
                        else:
                            try:
                                hparams_grp.attrs[k] = self._to_serializable(v)
                            except:
                                pass
            else:
                print("    ⚠️  Warning: No hparams to save!")

    def reset(self):
        """Clear recorded states."""
        self.states = []

def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None, performance_monitor=None, obs_keys=None, lnn_record=False):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu. 
            They are excluded by default because the low-dimensional simulation states should be a minimal 
            representation of the environment. 
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    if lnn_record:
        lnn_recorder = LNNStateRecorder(enable=True, device='cuda')
        
        core = policy.policy.nets['policy'].core
        print("LNN recording core:", core)
        lnn_recorder.capture_hparams(core)
        lnn_recorder.attach_hooks(core)

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)

    # filter observation(for low dim policy)
    obs = ObsUtils.filter_obs(obs, obs_keys)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))

    # for monitoring performance
    step_latencies, policy_latencies, env_step_times = [], [], []
    gpu_memories, cpu_memories = [], []

    try:
        for step_i in range(horizon):
            gpu_memory=0.0
            cpu_memory=0.0

            t1 = time.time()
            # get action from policy
            act = policy(ob=obs)

            t2 = time.time()
            policy_latencies.append(t2 - t1)

            # play action
            next_obs, r, done, _ = env.step(act)
            t3 = time.time()
            env_step_times.append(t3 - t2)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            if performance_monitor is not None:
                gpu_memories.append(performance_monitor.gpu_memory_mb()-gpu_memory)
                cpu_memories.append(performance_monitor.cpu_memory_mb()-cpu_memory)

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                traj["obs"].append(obs)
                traj["next_obs"].append(next_obs)

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

            step_latencies.append(time.time() - t1)

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    finally:
        if lnn_record:
            lnn_recorder.save_to_hdf5("/work/tmp/lnn_hparams.hdf5")
            lnn_recorder.detach_hooks()

    stats = dict(Return=total_reward, 
                Horizon=(step_i + 1), 
                Success_Rate=float(success),
                Avg_Policy_Latency=np.mean(policy_latencies),
                Std_Policy_Latency=np.std(policy_latencies),
                Avg_Env_Step_Time=np.mean(env_step_times),
                Std_Env_Step_Time=np.std(env_step_times),
                Avg_Step_Latency=np.mean(step_latencies),
                Std_Step_Latency=np.std(step_latencies),
                Avg_GPU_Memory_MB=np.mean(gpu_memories),
                Std_GPU_Memory_MB=np.std(gpu_memories),
                Avg_CPU_Memory_MB=np.mean(cpu_memories),
                Std_CPU_Memory_MB=np.std(cpu_memories),
                )

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


def run_trained_agent(args):
    # some arg checking
    write_video = (args.video_path is not None)
    write_csv = (args.csv_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # relative path to agent
    ckpt_path = args.agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    if rollout_horizon is None:
        # read horizon from config
        rollout_horizon = config.experiment.rollout.horizon
    # read obs_keys from config
    with config.unlocked():
        obs_keys = list(config.observation.modalities.obs.low_dim)
        if obs_keys is None or len(obs_keys) == 0:
            obs_keys = list(config.observation.planner.modalities.obs.low_dim)
            if obs_keys is None or len(obs_keys) == 0:
                obs_keys = list(config.observation.value_planner.planner.modalities.obs.low_dim)
    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=args.env, 
        render=args.render, 
        render_offscreen=(args.video_path is not None), 
        verbose=True,
    )

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    csv_file = None
    if write_csv:

        fieldnames = [
            'name',
            'return',
            'horizon',
            'success_rate',
            'avg_policy_latency_ms',
            'std_policy_latency_ms',
            'avg_env_step_time_ms',
            'std_env_step_time_ms',
            'avg_step_latency_ms',
            'std_step_latency_ms',
            'avg_gpu_memory_increase_mb',
            'std_gpu_memory_increase_mb',
            'avg_cpu_memory_increase_mb',
            'std_cpu_memory_increase_mb'
        ]

        csv_dir = os.path.dirname(args.csv_path)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)

        if os.path.exists(args.csv_path):
            print("CSV file {} already exists and will be appended.".format(args.csv_path))
            csv_file = open(args.csv_path, mode='a', newline='')
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        else:
            print("Creating new CSV file at {}".format(args.csv_path))
            csv_file = open(args.csv_path, mode='w', newline='')
            # write csv header
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []
    for i in range(rollout_num_episodes):
        stats, traj = rollout(
            policy=policy, 
            env=env, 
            horizon=rollout_horizon, 
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip, 
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
            performance_monitor=PerformanceMonitor(device=device),
            obs_keys=obs_keys,
            lnn_record=args.lnn_record
        )
        args.lnn_record = False  # only record LNN states for the first rollout
        rollout_stats.append(stats)
        print(f"Rollout {i+1}/{rollout_num_episodes}", end='\r', flush=True)

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))

    if write_csv:
        # write average stats to csv
        csv_writer.writerow({
            'name': args.name,
            'return': avg_rollout_stats["Return"],
            'horizon': avg_rollout_stats["Horizon"],
            'success_rate': avg_rollout_stats["Num_Success"] / rollout_num_episodes,
            'avg_policy_latency_ms': avg_rollout_stats["Avg_Policy_Latency"] * 1000.0,
            'std_policy_latency_ms': avg_rollout_stats["Std_Policy_Latency"] * 1000.0,
            'avg_env_step_time_ms': avg_rollout_stats["Avg_Env_Step_Time"] * 1000.0,
            'std_env_step_time_ms': avg_rollout_stats["Std_Env_Step_Time"] * 1000.0,
            'avg_step_latency_ms': avg_rollout_stats["Avg_Step_Latency"] * 1000.0,
            'std_step_latency_ms': avg_rollout_stats["Std_Step_Latency"] * 1000.0,
            'avg_gpu_memory_increase_mb': avg_rollout_stats["Avg_GPU_Memory_MB"],
            'std_gpu_memory_increase_mb': avg_rollout_stats["Std_GPU_Memory_MB"],
            'avg_cpu_memory_increase_mb': avg_rollout_stats["Avg_CPU_Memory_MB"],
            'std_cpu_memory_increase_mb': avg_rollout_stats["Std_CPU_Memory_MB"],
        })
        csv_file.close()
        print("Wrote rollout stats to {}".format(args.csv_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # register name
    parser.add_argument(
        "--name",
        type=str,
        default="default",
    )

    # csv path to write rollout stats
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
    )

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=27,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    parser.add_argument(
        "--lnn_record",
        type=bool,
        default=False,
        help="If true, record LNN states during rollout.",
    )

    args = parser.parse_args()
    run_trained_agent(args)

