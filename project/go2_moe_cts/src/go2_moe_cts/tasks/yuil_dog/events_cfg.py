# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mass, material, actuator, reset and push randomization settings."""

from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from ...mdp.events import randomize_motor_zero_offset


def configure_events(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Apply the reference domain randomization using Yuil Dog body names."""
    base_cfg = SceneEntityCfg("robot", body_names="base_link")
    env_cfg.events.add_base_mass.params.update(
        asset_cfg=base_cfg, mass_distribution_params=(-1.0, 1.0), recompute_inertia=True
    )
    env_cfg.events.base_com.params.update(asset_cfg=base_cfg, com_range={axis: (-0.03, 0.03) for axis in "xyz"})
    env_cfg.events.physics_material.params.update(
        static_friction_range=(0.0, 2.0),
        dynamic_friction_range=(0.0, 2.0),
        restitution_range=(0.0, 0.5),
        make_consistent=True,
    )
    env_cfg.events.reset_robot_joints.params.update(position_range=(0.5, 1.5), velocity_range=(0.0, 0.0))
    env_cfg.events.reset_base.params["pose_range"]["z"] = (0.0, 0.2)
    env_cfg.events.push_robot.interval_range_s = (4.0, 4.0)
    env_cfg.events.push_robot.params["velocity_range"] = {
        "x": (-0.4, 0.4),
        "y": (-0.4, 0.4),
        "roll": (-0.6, 0.6),
        "pitch": (-0.6, 0.6),
        "yaw": (-0.6, 0.6),
    }
    env_cfg.events.scale_limb_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!base_link$).*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    env_cfg.events.actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    env_cfg.events.motor_zero_offset = EventTerm(func=randomize_motor_zero_offset, mode="reset")
