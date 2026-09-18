# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton configuration matching the Robotis HX5 V7 policy contract."""

from __future__ import annotations

from pathlib import Path

from isaaclab_newton.sim.schemas import (
    NewtonArticulationRootPropertiesCfg,
    NewtonJointDrivePropertiesCfg,
    NewtonMaterialPropertiesCfg,
    NewtonRigidBodyPropertiesCfg,
)

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass

from .hx5_cube_env_cfg import (
    DEFAULT_JOINT_POSITIONS,
    PROJECT_DIR,
    ROBOT_ROOT_ORIENTATION_XYZW,
    ROBOT_ROOT_POSITION,
)
from .hx5_cube_newton_env_cfg import Hx5CubeNewtonEnvCfg, default_robot_usd_path

V7_CUBE_USD = PROJECT_DIR / "assets" / "dex_cube_v7_newton" / "dex_cube.usda"
V7_OBJECT_INITIAL_POSITION = (0.0, -0.44, 0.60)
V7_TARGET_POSITION = (0.0, -0.44, 0.56)
V7_CUBE_SIDE_LENGTH = 0.048
V7_CUBE_MASS = 0.0365
V7_CUBE_DIAGONAL_INERTIA = V7_CUBE_MASS * V7_CUBE_SIDE_LENGTH**2 / 6.0


def make_v7_robot_cfg(usd_path: str | Path) -> ArticulationCfg:
    """Create the V7 nominal Robotis hand configuration for Newton.

    Args:
        usd_path: Path to the HX5 hand USD asset.

    Returns:
        Newton-compatible articulation with the V7 actuator parameters.
    """
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(usd_path).expanduser()),
            activate_contact_sensors=False,
            physics_material=NewtonMaterialPropertiesCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            rigid_props=NewtonRigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=NewtonArticulationRootPropertiesCfg(self_collision_enabled=True),
            joint_drive_props=NewtonJointDrivePropertiesCfg(drive_type="force"),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_ROOT_POSITION,
            rot=ROBOT_ROOT_ORIENTATION_XYZW,
            joint_pos=DEFAULT_JOINT_POSITIONS,
        ),
        actuators={
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["finger_l_joint[0-9]+"],
                effort_limit_sim=1.03,
                stiffness=3.0,
                damping=0.05,
                friction=0.01,
                armature=0.0,
            )
        },
        soft_joint_pos_limit_factor=1.0,
    )


@configclass
class Hx5CubeV7NewtonEnvCfg(Hx5CubeNewtonEnvCfg):
    """Continuous Newton playback task for the Robotis V7 policy."""

    robot_cfg: ArticulationCfg = make_v7_robot_cfg(default_robot_usd_path())

    object_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(V7_CUBE_USD),
            semantic_tags=[("class", "cube")],
            physics_material=NewtonMaterialPropertiesCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=V7_OBJECT_INITIAL_POSITION,
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
        articulation_root_prim_path="",
    )

    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={"goal": sim_utils.UsdFileCfg(usd_path=str(V7_CUBE_USD))},
    )

    reset_dof_pos_noise = 0.05
    action_penalty_scale = -0.0006
    action_bound_threshold = 0.90
    action_bound_penalty_scale = -0.01
    fall_penalty = 0.0
    success_tolerance = 0.2
    pos_success_tolerance = 0.045
    success_hold_steps = 1
    terminate_on_success = False
    reset_target_on_success = True

    action_clip_value = 1.0
    action_joint_limit_margin = 0.03
    act_moving_average = 0.8
    max_target_delta = 0.08
    action_pipeline_delay_physics_steps_range = (0, 2)
    motor_target_time_constant_s_range = (0.005, 0.015)

    cube_pose_obs_delay_steps_range = (1, 2)
    cube_position_obs_noise_std = 0.002
    cube_orientation_obs_noise_std = 0.0349066
    cube_pose_obs_noise_clip_sigma = 3.0
    cube_velocity_obs_low_pass_alpha = 0.35
    cube_linear_velocity_obs_max = 3.0
    cube_angular_velocity_obs_max = 30.0


__all__ = [
    "Hx5CubeV7NewtonEnvCfg",
    "V7_CUBE_DIAGONAL_INERTIA",
    "V7_CUBE_MASS",
    "V7_CUBE_SIDE_LENGTH",
    "V7_CUBE_USD",
    "V7_OBJECT_INITIAL_POSITION",
    "V7_TARGET_POSITION",
    "make_v7_robot_cfg",
]
