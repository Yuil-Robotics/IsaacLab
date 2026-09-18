# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the Go2 locomotion policy with RSL-RL PPO."""

import argparse
import importlib.metadata as metadata
import os
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Go2 locomotion with RSL-RL PPO.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Video length in environment steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Go2-MoECTS-v0", help="Task name.")
parser.add_argument("--seed", type=int, default=None, help="Environment and agent random seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override the training iterations.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume from a checkpoint.")
parser.add_argument("--load_run", type=str, default=None, help="Run directory name or regular expression.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint filename or regular expression.")
parser.add_argument("--run_name", type=str, default=None, help="Suffix for the training run.")
parser.add_argument("--smoke_test", action="store_true", help="Small terrain and 2 iterations for pipeline validation.")
parser.add_argument("--gui", action="store_true", help="Compatibility alias for '--viz kit'.")
parser.add_argument(
    "--command_debug_vis",
    action="store_true",
    default=None,
    help="Show target and actual vx/vy/wz arrows; enabled by default with Kit GUI.",
)
parser.add_argument(
    "--no_command_debug_vis",
    dest="command_debug_vis",
    action="store_false",
    help="Hide target and actual velocity arrows.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.gui:
    args_cli.visualizer = ["kit"]
    args_cli.visualizer_explicit = True
del args_cli.gui

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import go2_moe_cts  # noqa: F401
import gymnasium as gym
import torch
from go2_moe_cts.learning.runner import OnPolicyRunnerCTS
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main() -> None:
    """Train the Go2 locomotion policy."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    show_commands = args_cli.command_debug_vis
    if show_commands is None:
        show_commands = "kit" in (args_cli.visualizer or [])
    env_cfg.commands.base_velocity.debug_vis = show_commands
    if show_commands:
        print("[Velocity arrows] Green=target (+0.65 m), blue=actual (+0.48 m); vx/vy/wz combined.")
        print("[Velocity arrows] Same style as sim2sim: yaw tilts the arrow; pure yaw points left/right.")

    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.smoke_test:
        env_cfg.scene.num_envs = args_cli.num_envs or 8
        env_cfg.scene.terrain.terrain_generator.num_rows = 2
        env_cfg.scene.terrain.terrain_generator.num_cols = 20
        env_cfg.scene.terrain.max_init_terrain_level = 0
        agent_cfg.max_iterations = args_cli.max_iterations or 2
        agent_cfg.save_interval = 1
    agent_cfg.resume = args_cli.resume
    agent_cfg.device = args_cli.device
    env_cfg.seed = agent_cfg.seed

    is_cts = agent_cfg.class_name == "OnPolicyRunnerCTS"
    if not is_cts:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    runner_class = OnPolicyRunnerCTS if is_cts else OnPolicyRunner

    project_dir = Path(__file__).resolve().parents[2]
    log_root = str(project_dir / "logs" / "rsl_rl" / agent_cfg.experiment_name)
    log_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_name += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, log_name)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    resume_path = None
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    try:
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        runner.add_git_repo_to_log(__file__)

        if resume_path is not None:
            print(f"[INFO] Resuming from: {resume_path}")
            runner.load(resume_path)

        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        start_time = time.time()
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        if is_cts:
            runner.export_policy_to_jit(path=os.path.join(log_dir, "exported"))
            runner.export_policy_to_onnx(path=os.path.join(log_dir, "exported"))
        print(f"[INFO] Training time: {time.time() - start_time:.2f} seconds")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
