# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Active reward terms, weights and parameters for Yuil Dog CTS."""

from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ...mdp import rewards_cts as cts
from ...mdp.rewards import joint_pos_penalty_l1, joint_power, track_ang_vel_z_exp, track_lin_vel_xy_exp
from ...robots.yuil_dog.joints_cfg import YUIL_DOG_FOOT_NAMES, YUIL_DOG_JOINT_NAMES
from ...robots.yuil_dog.joints_cfg import YUIL_DOG_ROOT_HEIGHT as _BASE_HEIGHT

_JOINTS = SceneEntityCfg("robot", joint_names=list(YUIL_DOG_JOINT_NAMES), preserve_order=True)
_FEET = SceneEntityCfg("contact_forces", body_names=list(YUIL_DOG_FOOT_NAMES), preserve_order=True)


@configclass
class RewardsCTSCfg:
    """Reference reward weights with Yuil joint/body names and nominal height."""

    track_lin_vel_xy = RewTerm(
        func=track_lin_vel_xy_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z = RewTerm(
        func=track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1e-7, params={"asset_cfg": _JOINTS})
    joint_power = RewTerm(func=joint_power, weight=-2e-5, params={"asset_cfg": _JOINTS})
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4, params={"asset_cfg": _JOINTS})
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={"target_height": _BASE_HEIGHT, "sensor_cfg": SceneEntityCfg("height_scanner_small")},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness_l2 = RewTerm(func=cts.action_smoothness_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_(hip_pitch|knee_pitch)"),
            "threshold": 5.0,
        },
    )
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0, params={"asset_cfg": _JOINTS})
    feet_regulation = RewTerm(
        func=cts.feet_regulation,
        weight=-0.05,
        params={
            "base_height_target": _BASE_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot", body_names=list(YUIL_DOG_FOOT_NAMES)),
            "sensor_cfg": SceneEntityCfg("height_scanner_small"),
        },
    )
    hip_pos_penalty_l1 = RewTerm(
        func=cts.hip_pos_penalty_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_roll")},
    )
    joint_pos_penalty_l1 = RewTerm(
        func=joint_pos_penalty_l1,
        weight=-0.01,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_(hip|knee)_pitch"),
            "stand_still_scale": 1.0,
            "velocity_threshold": 0.1,
            "command_threshold": 0.1,
        },
    )
