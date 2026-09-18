# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone Yuil Dog flat-ground locomotion task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from .yuil_dog_cfg import (
    _apply_yuil_dog_randomization,
    _apply_yuil_dog_rewards,
    _apply_yuil_dog_robot,
    _apply_yuil_dog_terminations,
    _apply_yuil_dog_velocity_command,
)

YUIL_DOG_FLAT_TRAIN_TASK_ID = "Isaac-Velocity-Flat-Yuil-Dog-v0"
YUIL_DOG_FLAT_PLAY_TASK_ID = "Isaac-Velocity-Flat-Yuil-Dog-Play-v0"
YUIL_DOG_FLAT_EVAL_TASK_ID = "Isaac-Velocity-Flat-Yuil-Dog-Eval-v0"
YUIL_DOG_FLAT_EXPERIMENT_NAME = "yuil_dog_flat_leg_major"


def _apply_yuil_dog_flat_terrain(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Configure an infinite local ground plane without terrain curriculum."""
    env_cfg.scene.terrain.class_type = "go2_sim2sim.terrain:LocalPlaneTerrainImporter"
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.20, 0.20, 0.20),
        roughness=0.8,
    )
    env_cfg.scene.height_scanner = None
    env_cfg.observations.policy.height_scan = None
    env_cfg.curriculum.terrain_levels = None


def _apply_yuil_dog_flat_rewards(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Retain the Yuil Dog objective and measure base height from the flat plane."""
    _apply_yuil_dog_rewards(env_cfg)
    env_cfg.rewards.base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-40.0,
        params={
            "target_height": 0.33,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class YuilDogFlatTrainEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Standalone Yuil Dog training task for flat-ground locomotion."""

    def __post_init__(self) -> None:
        """Initialize the flat scene, robot, commands, and training objective."""
        super().__post_init__()
        self.scene.num_envs = 4096
        _apply_yuil_dog_robot(self)
        _apply_yuil_dog_flat_terrain(self)
        _apply_yuil_dog_randomization(self)
        _apply_yuil_dog_velocity_command(self, debug_vis=False)
        _apply_yuil_dog_flat_rewards(self)
        _apply_yuil_dog_terminations(self)
        self.terminations.terrain_out_of_bounds = None


@configclass
class YuilDogFlatPlayEnvCfg(YuilDogFlatTrainEnvCfg):
    """Standalone Yuil Dog flat-ground policy playback task."""

    def __post_init__(self) -> None:
        """Configure deterministic playback on a small flat scene."""
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.commands.base_velocity.debug_vis = True
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class YuilDogFlatEvalEnvCfg(YuilDogFlatTrainEnvCfg):
    """Standalone deterministic Yuil Dog flat-ground evaluation task."""

    def __post_init__(self) -> None:
        """Remove stochastic perturbations for benchmark evaluation."""
        super().__post_init__()
        self.scene.num_envs = 256
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_straight_envs = 0.0
        self.commands.base_velocity.rel_turning_envs = 0.0
        self.commands.base_velocity.rel_lateral_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.ranges.heading = None

        self.observations.policy.enable_corruption = False
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


@configclass
class YuilDogFlatPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    """Yuil Dog flat-ground PPO runner settings."""

    experiment_name = YUIL_DOG_FLAT_EXPERIMENT_NAME
    max_iterations = 3000
    save_interval = 50
    clip_actions = 3.5

    def __post_init__(self) -> None:
        """Apply the isolated experiment name and training budget."""
        super().__post_init__()
        self.experiment_name = YUIL_DOG_FLAT_EXPERIMENT_NAME
        self.max_iterations = 3000
        self.save_interval = 50
        self.clip_actions = 3.5
