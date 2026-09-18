# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a Go2 locomotion policy and export it as JIT / ONNX.

Usage
-----
    ./isaaclab.sh -p scripts/play.py \\
        --task Isaac-Velocity-Rough-Go2-MoECTS-Play-v0 \\
        --checkpoint logs/rsl_rl/go2_moe_cts_45x10/<run>/model_<iter>.pt

The exported ``policy.pt`` (TorchScript) is saved alongside the checkpoint
and can be used directly with the MuJoCo deploy script.
"""

import argparse
import importlib.metadata as metadata
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate Go2 locomotion policy.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Go2-MoECTS-Play-v0")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path; defaults to the latest run.")
parser.add_argument(
    "--num_steps",
    type=int,
    default=0,
    help="Number of policy steps to simulate. Use 0 to run until Ctrl+C.",
)
parser.add_argument("--real_time", action="store_true", help="Throttle simulation to the environment control rate.")
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

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import go2_moe_cts  # noqa: F401
import gymnasium as gym
import torch
from go2_moe_cts.learning.runner import OnPolicyRunnerCTS
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401


def main():
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.seed = args_cli.seed

    show_commands = args_cli.command_debug_vis
    if show_commands is None:
        show_commands = "kit" in (args_cli.visualizer or [])
    env_cfg.commands.base_velocity.debug_vis = show_commands
    if show_commands:
        print("[Velocity arrows] Green=target (+0.65 m), blue=actual (+0.48 m); vx/vy/wz combined.")
        print("[Velocity arrows] Same style as sim2sim: yaw tilts the arrow; pure yaw points left/right.")

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device
    is_cts = agent_cfg.class_name == "OnPolicyRunnerCTS"
    if not is_cts:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    runner_class = OnPolicyRunnerCTS if is_cts else OnPolicyRunner

    project_dir = Path(__file__).resolve().parents[2]
    log_root = str(project_dir / "logs" / "rsl_rl" / agent_cfg.experiment_name)
    if args_cli.checkpoint:
        checkpoint = Path(args_cli.checkpoint).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = project_dir / checkpoint
        resume_path = retrieve_file_path(str(checkpoint))
    else:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        runner_kwargs = {"inference_only": True} if is_cts else {}
        runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device, **runner_kwargs)
        runner.load(resume_path, load_optimizer=False)

        export_dir = os.path.join(os.path.dirname(resume_path), "exported")
        runner.export_policy_to_jit(path=export_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_dir, filename="policy.onnx")
        print(f"[INFO] Policies exported to: {export_dir}")

        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations()
        step = 0
        try:
            while args_cli.num_steps <= 0 or step < args_cli.num_steps:
                start_time = time.time()
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    policy.reset(dones)
                step += 1

                if args_cli.real_time:
                    sleep_time = env.unwrapped.step_dt - (time.time() - start_time)
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\n[INFO] Play stopped by user.")
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
