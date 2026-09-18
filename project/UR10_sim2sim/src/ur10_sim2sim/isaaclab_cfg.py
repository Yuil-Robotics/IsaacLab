# Copyright (c) 2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Project-owned Isaac Lab task and training configurations.

The classes in this module inherit the upstream UR10e deployment task. Project
experiments can override values here without changing Isaac Lab source files.
"""

from __future__ import annotations

import sys

import gymnasium as gym

from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.deploy.reach.config.ur_10e.agents.rsl_rl_ppo_cfg import (
    URReachPPORunnerCfg,
)
from isaaclab_tasks.manager_based.manipulation.deploy.reach.config.ur_10e.joint_pos_env_cfg import (
    UR10eReachEnvCfg,
)

TRAIN_TASK_ID = "Isaac-Reach-UR10e-Sim2Sim-v0"
PLAY_TASK_ID = "Isaac-Reach-UR10e-Sim2Sim-Play-v0"
POSE_COMMAND_ENTRY_POINT = "ur10_sim2sim.isaaclab_visualization:UR10BasePoseCommand"


@configclass
class UR10Sim2SimReachEnvCfg(UR10eReachEnvCfg):
    """UR10e Reach training environment owned by this project."""

    def __post_init__(self) -> None:
        """Apply project-specific environment settings."""
        super().__post_init__()
        self.scene.num_envs = 4096
        self.commands.ee_pose.class_type = POSE_COMMAND_ENTRY_POINT
        self.commands.ee_pose.debug_vis = True

        goal_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/UR10Sim2Sim/TargetFrame")
        goal_marker_cfg.markers["frame"].scale = (0.3, 0.3, 0.3)
        self.commands.ee_pose.goal_pose_visualizer_cfg = goal_marker_cfg

        current_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/UR10Sim2Sim/EndEffectorFrame")
        current_marker_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        self.commands.ee_pose.current_pose_visualizer_cfg = current_marker_cfg


@configclass
class UR10Sim2SimReachEnvCfgPlay(UR10Sim2SimReachEnvCfg):
    """Evaluation variant with fewer environments and deterministic observations."""

    def __post_init__(self) -> None:
        """Apply evaluation settings."""
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class UR10Sim2SimReachPPORunnerCfg(URReachPPORunnerCfg):
    """Project PPO configuration with a shorter rollout."""

    # Collect 64 policy steps per environment before each PPO update.
    num_steps_per_env = 64
    max_iterations = 1500
    save_interval = 50
    experiment_name = "reach_ur10_sim2sim"


def register_tasks() -> list[str]:
    """Register project Gym tasks and preserve Hydra arguments.

    Returns:
        Original command-line arguments for Isaac Lab's external-callback
        intersection.
    """
    if TRAIN_TASK_ID not in gym.registry:
        gym.register(
            id=TRAIN_TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}:UR10Sim2SimReachEnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}:UR10Sim2SimReachPPORunnerCfg",
            },
        )
    if PLAY_TASK_ID not in gym.registry:
        gym.register(
            id=PLAY_TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}:UR10Sim2SimReachEnvCfgPlay",
                "rsl_rl_cfg_entry_point": f"{__name__}:UR10Sim2SimReachPPORunnerCfg",
            },
        )
    return sys.argv[1:]
