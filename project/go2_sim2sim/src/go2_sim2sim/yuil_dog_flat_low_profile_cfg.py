# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Yuil Dog flat-ground task using the go2_rl_robotlab reward suite."""

from __future__ import annotations

from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from . import low_profile_rewards, robotlab_rewards, rough_mdp
from .asset_cfg import YUIL_DOG_FOOT_NAMES, YUIL_DOG_JOINT_NAMES
from .yuil_dog_flat_robotlab_cfg import (
    YuilDogFlatRobotLabEvalEnvCfg,
    YuilDogFlatRobotLabPlayEnvCfg,
    YuilDogFlatRobotLabPPORunnerCfg,
    YuilDogFlatRobotLabTrainEnvCfg,
)

YUIL_DOG_FLAT_LOW_PROFILE_TRAIN_TASK_ID = "Isaac-Velocity-Flat-Yuil-Dog-Low-Profile-v0"
YUIL_DOG_FLAT_LOW_PROFILE_PLAY_TASK_ID = "Isaac-Velocity-Flat-Yuil-Dog-Low-Profile-Play-v0"
YUIL_DOG_FLAT_LOW_PROFILE_EVAL_TASK_ID = "Isaac-Velocity-Flat-Yuil-Dog-Low-Profile-Eval-v0"
YUIL_DOG_FLAT_LOW_PROFILE_EXPERIMENT_NAME = "yuil_dog_flat_low_profile"

BASE_HEIGHT_PREFERRED_MIN: float = 0.33
"""Legacy preferred base-bottom clearance [m], retained for compatibility."""

BASE_HEIGHT_HARD_MAX: float = 0.35
"""Legacy base-bottom clearance ceiling [m], retained for compatibility."""

BASE_HEIGHT_TARGET: float = 0.32
"""Target Yuil Dog base-link height above the ground [m]."""

MAXIMUM_TROT_FREQUENCY: float = 2.5
"""Legacy cadence ceiling [Hz], retained for compatibility and no longer used by this task."""

FORWARD_MINIMUM_CONTACT_TIME: float = 0.30
"""Minimum completed foot-contact time for sagittal locomotion [s]."""

LATERAL_MINIMUM_CONTACT_TIME: float = 0.24
"""Minimum completed foot-contact time for lateral locomotion [s]."""

TURNING_MINIMUM_CONTACT_TIME: float = 0.22
"""Minimum completed foot-contact time for turning locomotion [s]."""

FOOT_CLEARANCE_TARGET: float = 0.05
"""Target swing-foot clearance above the most recent support height [m]."""

MAXIMUM_STRIDE_LENGTH: float = 0.52
"""Legacy maximum stride target [m], retained for compatibility."""


def _clear_config_terms(group) -> None:
    """Disable all manager terms currently stored on a configuration group."""
    for attr in list(vars(group).keys()):
        if not attr.startswith("_"):
            setattr(group, attr, None)


def _apply_yuil_dog_flat_low_profile_rewards(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Replace inherited rewards with the go2_rl_robotlab reward suite."""
    _clear_config_terms(env_cfg.rewards)

    joint_names = list(YUIL_DOG_JOINT_NAMES)
    foot_names = list(YUIL_DOG_FOOT_NAMES)
    robot_asset_cfg = SceneEntityCfg("robot")
    all_joints_asset_cfg = SceneEntityCfg("robot", joint_names=joint_names, preserve_order=True)
    hip_joints_asset_cfg = SceneEntityCfg("robot", joint_names=[".*_hip_roll"])
    leg_joints_asset_cfg = SceneEntityCfg("robot", joint_names=[".*_hip_pitch", ".*_knee_pitch"])
    base_asset_cfg = SceneEntityCfg("robot", body_names=["base_link"])
    foot_asset_cfg = SceneEntityCfg("robot", body_names=foot_names)
    foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=foot_names)
    height_sensor_cfg = SceneEntityCfg("height_scanner")
    undesired_contact_sensor_cfg = SceneEntityCfg("contact_forces", body_names=[".*_hip_pitch", ".*_knee_pitch"])

    env_cfg.rewards.track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5, "asset_cfg": robot_asset_cfg},
    )
    env_cfg.rewards.track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5, "asset_cfg": robot_asset_cfg},
    )
    env_cfg.rewards.lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
        params={"asset_cfg": robot_asset_cfg},
    )
    env_cfg.rewards.ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": robot_asset_cfg},
    )
    env_cfg.rewards.joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": all_joints_asset_cfg},
    )
    env_cfg.rewards.joint_power = RewTerm(
        func=robotlab_rewards.joint_power,
        weight=-2.0e-5,
        params={"asset_cfg": all_joints_asset_cfg},
    )
    env_cfg.rewards.joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-4,
        params={"asset_cfg": all_joints_asset_cfg},
    )
    env_cfg.rewards.base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "asset_cfg": base_asset_cfg,
            "target_height": BASE_HEIGHT_TARGET,
            "sensor_cfg": height_sensor_cfg,
        },
    )
    env_cfg.rewards.action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    env_cfg.rewards.action_smoothness_l2 = RewTerm(func=robotlab_rewards.action_smoothness_l2, weight=-0.01)
    env_cfg.rewards.undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": undesired_contact_sensor_cfg, "threshold": 5.0},
    )
    env_cfg.rewards.joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": all_joints_asset_cfg},
    )
    env_cfg.rewards.feet_regulation = RewTerm(
        func=robotlab_rewards.feet_regulation,
        weight=-0.05,
        params={
            "base_height_target": BASE_HEIGHT_TARGET,
            "asset_cfg": foot_asset_cfg,
            "sensor_cfg": height_sensor_cfg,
        },
    )
    env_cfg.rewards.hip_pos_penalty_l1 = RewTerm(
        func=robotlab_rewards.hip_pos_penalty_l1,
        weight=-0.05,
        params={
            "command_name": "base_velocity",
            "asset_cfg": hip_joints_asset_cfg,
            "stand_still_scale": 1.0,
            "command_threshold": 0.1,
        },
    )
    env_cfg.rewards.joint_pos_penalty_l1 = RewTerm(
        func=robotlab_rewards.joint_pos_penalty_l1,
        weight=-0.01,
        params={
            "command_name": "base_velocity",
            "asset_cfg": leg_joints_asset_cfg,
            "stand_still_scale": 1.0,
            "velocity_threshold": 0.1,
            "command_threshold": 0.1,
        },
    )
    env_cfg.rewards.contact_duration = RewTerm(
        func=low_profile_rewards.FootContactDurationReward,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "foot_names": foot_names,
            "sensor_cfg": foot_sensor_cfg,
            "motion_start": 0.05,
            "motion_full": 0.20,
            "yaw_scale": 0.25,
            "forward_minimum_contact_time": FORWARD_MINIMUM_CONTACT_TIME,
            "lateral_minimum_contact_time": LATERAL_MINIMUM_CONTACT_TIME,
            "turning_minimum_contact_time": TURNING_MINIMUM_CONTACT_TIME,
            "contact_time_margin": 0.08,
            "command_change_threshold": 0.05,
        },
    )
    env_cfg.rewards.foot_clearance = RewTerm(
        func=rough_mdp.RoughTerrainFootClearanceReward,
        weight=0.3,
        params={
            "asset_cfg": foot_asset_cfg,
            "sensor_cfg": foot_sensor_cfg,
            "target_height": FOOT_CLEARANCE_TARGET,
            "std": 0.03,
            "motion_start": 0.05,
            "motion_full": 0.20,
            "yaw_scale": 0.25,
            "recovery_velocity_threshold": 0.6,
            "command_name": "base_velocity",
            "flight_grace_time": 0.04,
            "flight_full_time": 0.10,
        },
    )


def _apply_yuil_dog_flat_low_profile_task(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Apply only the reward and curriculum overrides."""
    _apply_yuil_dog_flat_low_profile_rewards(env_cfg)
    _clear_config_terms(env_cfg.curriculum)


@configclass
class YuilDogFlatLowProfileTrainEnvCfg(YuilDogFlatRobotLabTrainEnvCfg):
    """Low-profile training task using the RobotLab robot and environment."""

    def __post_init__(self) -> None:
        """Keep the inherited environment and replace its locomotion objective."""
        super().__post_init__()
        _apply_yuil_dog_flat_low_profile_task(self)


@configclass
class YuilDogFlatLowProfilePlayEnvCfg(YuilDogFlatRobotLabPlayEnvCfg):
    """Deterministic playback configuration for the low-profile task."""

    def __post_init__(self) -> None:
        """Keep RobotLab playback settings and apply low-profile rewards."""
        super().__post_init__()
        _apply_yuil_dog_flat_low_profile_task(self)


@configclass
class YuilDogFlatLowProfileEvalEnvCfg(YuilDogFlatRobotLabEvalEnvCfg):
    """Deterministic evaluation configuration for the low-profile task."""

    def __post_init__(self) -> None:
        """Keep RobotLab evaluation settings and apply low-profile rewards."""
        super().__post_init__()
        _apply_yuil_dog_flat_low_profile_task(self)


@configclass
class YuilDogFlatLowProfilePPORunnerCfg(YuilDogFlatRobotLabPPORunnerCfg):
    """PPO runner with an isolated low-profile experiment directory."""

    experiment_name = YUIL_DOG_FLAT_LOW_PROFILE_EXPERIMENT_NAME

    def __post_init__(self) -> None:
        """Restore the low-profile experiment name after parent initialization."""
        super().__post_init__()
        self.experiment_name = YUIL_DOG_FLAT_LOW_PROFILE_EXPERIMENT_NAME
