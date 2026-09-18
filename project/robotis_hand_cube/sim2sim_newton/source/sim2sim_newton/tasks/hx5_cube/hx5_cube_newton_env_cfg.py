# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton/MJWarp configuration for playing the trained Robotis HX5 cube policy."""

from __future__ import annotations

import os
from pathlib import Path

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import (
    NewtonArticulationRootPropertiesCfg,
    NewtonJointDrivePropertiesCfg,
    NewtonMaterialPropertiesCfg,
    NewtonRigidBodyPropertiesCfg,
)

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from .hx5_cube_env_cfg import (
    ACTUATED_JOINT_NAMES,
    DEFAULT_JOINT_POSITIONS,
    FINGERTIP_BODY_NAMES,
    NEWTON_CUBE_USD,
    NEWTON_ROBOT_USD,
    OBJECT_INITIAL_POSITION,
    POLICY_ACTION_DIM,
    POLICY_OBSERVATION_DIM,
    ROBOT_BODY_COUNT,
    ROBOT_ROOT_ORIENTATION_XYZW,
    ROBOT_ROOT_POSITION,
    SIM2REAL_ACTUATOR_ARMATURE,
)

__all__ = [
    "Hx5CubeNewtonEnvCfg",
    "Hx5CubeNewtonSingleGoalEnvCfg",
    "POLICY_OBSERVATION_DIM",
    "POLICY_ACTION_DIM",
    "ROBOT_BODY_COUNT",
    "SINGLE_GOAL_OBJECT_INITIAL_POSITION",
]

# Backwards-compatible backend-specific name.  PhysX robust training and
# Newton validation intentionally share the same Sim2Real nominal.
NEWTON_ACTUATOR_ARMATURE = SIM2REAL_ACTUATOR_ARMATURE
GOAL_MARKER_POSITION = (-0.2, -0.45, 0.68)
SINGLE_GOAL_OBJECT_INITIAL_POSITION = (0.0, -0.44, 0.60)


def default_robot_usd_path() -> Path:
    """Return the configured Newton-compatible Robotis hand USD path."""
    configured_path = os.environ.get("ROBOTIS_HAND_USD")
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return NEWTON_ROBOT_USD.resolve()


def make_robot_cfg(usd_path: str | Path) -> ArticulationCfg:
    """Create the Newton-compatible Robotis hand articulation configuration.

    Args:
        usd_path: Path to the HX5 hand USD asset.

    Returns:
        Newton-compatible Robotis hand articulation configuration.
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
                stiffness=1.0,
                damping=0.11,
                friction=0.0,
                armature=NEWTON_ACTUATOR_ARMATURE,
            )
        },
        soft_joint_pos_limit_factor=1.0,
    )


@configclass
class Hx5CubeNewtonEnvCfg(DirectRLEnvCfg):
    """Play-only Robotis HX5 cube task using the Newton MJWarp backend."""

    seed = 42
    decimation = 4
    episode_length_s = 10.0
    action_space = POLICY_ACTION_DIM
    observation_space = POLICY_OBSERVATION_DIM
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics_material=NewtonMaterialPropertiesCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=400,
                nconmax=200,
                impratio=1.0,
                cone="pyramidal",
                update_data_interval=2,
                iterations=100,
            ),
            num_substeps=8,
            debug_mode=False,
        ),
    )

    robot_cfg: ArticulationCfg = make_robot_cfg(default_robot_usd_path())
    actuated_joint_names = ACTUATED_JOINT_NAMES
    fingertip_body_names = FINGERTIP_BODY_NAMES

    object_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(NEWTON_CUBE_USD),
            semantic_tags=[("class", "cube")],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=OBJECT_INITIAL_POSITION,
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
        articulation_root_prim_path="",
    )

    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=str(NEWTON_CUBE_USD),
            )
        },
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=0.75,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    reset_position_noise = 0.01
    reset_rotation_noise = 1.0
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.0

    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0003
    reach_goal_bonus = 250.0
    fall_penalty = -100.0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.4
    success_hold_steps = 15
    terminate_on_success = False
    reset_target_on_success = True
    success_requires_not_fallen = False
    reward_mode = "proximity"
    max_consecutive_success = 0
    success_count_threshold = 1
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0


@configclass
class Hx5CubeNewtonSingleGoalEnvCfg(Hx5CubeNewtonEnvCfg):
    """Newton evaluation task that terminates after one stable goal."""

    fall_penalty = -50.0
    terminate_on_success = True
    reset_target_on_success = False
    success_requires_not_fallen = True
    reward_mode = "single_goal_progress"
    action_clip_value = 1.0
    act_moving_average = 0.8
    joint_limit_safety_margin_fraction = 0.03
    joint_target_velocity_limit = 2.4
    rotation_progress_scale = 20.0
    orientation_error_penalty_scale = -0.5
    distance_safe_radius = 0.06
    distance_penalty_scale = -100.0
    action_rate_penalty_scale = -0.01
    action_saturation_threshold = 0.9
    action_saturation_penalty_scale = -0.01
    joint_velocity_penalty_scale = -0.01
    # Match the PhysX single-goal task: 0.5 s at the 30 Hz policy rate.
    success_hold_steps = 15
    # Compatibility-only setting; angular speed does not gate success.
    success_max_object_angvel = 2.0
    hold_reward_scale = 1.0
    time_penalty = -0.01
    timeout_penalty = -50.0

    # Match nominal PhysX observations: exact pose at 30 Hz with velocity
    # reconstructed from consecutive pose measurements through the same filter.
    enable_cube_pose_obs_noise = False
    enable_cube_pose_sample_hold = False
    enable_cube_velocity_from_pose = True
    cube_position_obs_noise_std = (0.0, 0.0, 0.0)
    cube_position_obs_bias_range = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    cube_orientation_obs_noise_std = 0.0
    cube_orientation_obs_bias_max = 0.0
    cube_pose_obs_noise_clip_sigma = 0.0
    cube_velocity_obs_low_pass_alpha = 0.35
    cube_linear_velocity_obs_max = 3.0
    cube_angular_velocity_obs_max = 30.0

    def __post_init__(self) -> None:
        """Match the PhysX single-goal cube and in-hand reference positions."""
        self.object_cfg.init_state.pos = SINGLE_GOAL_OBJECT_INITIAL_POSITION
