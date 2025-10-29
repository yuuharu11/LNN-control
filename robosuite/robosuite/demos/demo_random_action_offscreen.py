import time
import os

# ⚠️ 重要: robosuiteをインポートする前に環境変数を設定
# EGL (GPU) または OSMesa (CPU) を選択
os.environ["MUJOCO_GL"] = "egl"  # GPU高速レンダリング (DRI必要) または "osmesa" (CPU)
os.environ["PYOPENGL_PLATFORM"] = "egl"  # または "osmesa"

import numpy as np
from robosuite.robots import MobileRobot
from robosuite.utils.input_utils import *

MAX_FR = 25  # max frame rate for running simulation

if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    options["env_name"] = choose_environment()

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # Choose env config and add it to options
        options["env_configuration"] = choose_multi_arm_config()

        # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        if options["env_configuration"] == "single-robot":
            options["robots"] = choose_robots(exclude_bimanual=False, use_humanoids=True, exclude_single_arm=True)
        else:
            options["robots"] = []

            # Have user choose two robots
            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=False, use_humanoids=True))
    # If a humanoid environment has been chosen, choose humanoid robots
    elif "Humanoid" in options["env_name"]:
        options["robots"] = choose_robots(use_humanoids=True)
    else:
        options["robots"] = choose_robots(exclude_bimanual=False, use_humanoids=True)

    # 動画保存するかどうか
    save_video = True
    frames = [] if save_video else None

    # initialize the task (オフスクリーン設定)
    env = suite.make(
        **options,
        has_renderer=False,              # ウィンドウなし
        has_offscreen_renderer=save_video,  # 動画保存時のみTrue
        ignore_done=True,
        use_camera_obs=save_video,       # 動画保存時のみカメラ観測を有効化
        camera_names=["frontview"] if save_video else None,
        camera_heights=480,
        camera_widths=640,
        control_freq=20,
    )
    obs = env.reset()
    
    # env.viewerは存在しないので、カメラ設定は削除
    # env.viewer.set_camera(camera_id=0)  # ← 削除
    
    for robot in env.robots:
        if isinstance(robot, MobileRobot):
            robot.enable_parts(legs=False, base=False)

    # do visualization
    print(f"\nRunning {10000} steps with random actions...")
    for i in range(10000):
        start = time.time()
        action = np.random.randn(*env.action_spec[0].shape)
        obs, reward, done, _ = env.step(action)
        
        # オフスクリーンではenv.render()は効果なし
        # env.render()  # ← 削除
        
        # 画像を保存（10ステップごと）
        if save_video and i % 10 == 0:
            if "frontview_image" in obs:
                frames.append(obs["frontview_image"])

        # 進捗表示
        if (i + 1) % 1000 == 0:
            print(f"  Step {i + 1}/10000")

        # limit frame rate if necessary
        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)

    # 動画保存
    if save_video and frames:
        try:
            import imageio
            video_path = "/work/random_rollout.mp4"
            print(f"\nSaving video to {video_path}...")
            imageio.mimsave(video_path, frames, fps=20)
            print(f"✅ Video saved successfully ({len(frames)} frames)")
        except ImportError:
            print("⚠️ imageio not installed. Install with: pip install imageio imageio-ffmpeg")
        except Exception as e:
            print(f"⚠️ Failed to save video: {e}")
    
    env.close()
    print("\n✅ Done!")
