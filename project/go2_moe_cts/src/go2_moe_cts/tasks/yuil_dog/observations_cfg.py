# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""History and privileged observation groups for Yuil Dog CTS."""

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ...env_cfg import ObservationsCfg
from ...mdp import observations_cts as cts
from ...robots.yuil_dog.joints_cfg import YUIL_DOG_FOOT_NAMES, YUIL_DOG_JOINT_NAMES

_JOINTS = SceneEntityCfg("robot", joint_names=list(YUIL_DOG_JOINT_NAMES), preserve_order=True)
_FEET = SceneEntityCfg("contact_forces", body_names=list(YUIL_DOG_FOOT_NAMES), preserve_order=True)


@configclass
class ObservationsCTSCfg:
    """450D noisy history, 263D clean privileged state, and the latest 45D frame."""

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}, clip=(-100.0, 100.0)
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _JOINTS}, clip=(-100.0, 100.0))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _JOINTS}, clip=(-100.0, 100.0), scale=0.05)
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))
        joint_acc = ObsTerm(func=cts.joint_acc, params={"asset_cfg": _JOINTS}, clip=(-100.0, 100.0), scale=1e-4)
        joint_torque = ObsTerm(func=cts.joint_effort, params={"asset_cfg": _JOINTS}, clip=(-100.0, 100.0), scale=0.01)
        contact_force = ObsTerm(
            func=cts.foot_contact_force_norm, params={"sensor_cfg": _FEET}, clip=(-100.0, 100.0), scale=1e-3
        )
        height_scan = ObsTerm(
            func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")}, clip=(-1.0, 1.0), scale=2.5
        )
        enable_corruption = False
        concatenate_terms = True

    @configclass
    class SingleObsCfg(ObsGroup):
        latest = ObsTerm(func=cts.single_policy_obs)
        enable_corruption = False
        concatenate_terms = True

    policy = ObservationsCfg.PolicyCfg()
    critic = CriticCfg()
    single_obs = SingleObsCfg()
