# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX training configuration for the Robotis HX5 cube reorientation task."""

from __future__ import annotations

from pathlib import Path

from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

##
# Constants
##

POLICY_ACTION_DIM = 20
ROBOT_BODY_COUNT = 21

# observation_space: 20 dof pos + 20 dof vel + 3 obj pos + 4 obj rot +
#                    3 obj linvel + 3 obj angvel + 3 in_hand_pos + 4 goal_rot +
#                    4 rel_rot + (5 * 3) fingertip pos + (5 * 4) fingertip rot +
#                    (5 * 6) fingertip vel + 20 actions
#                  = 20+20+3+4+3+3+3+4+4+15+20+30+20 = 149
POLICY_OBSERVATION_DIM = 149

PROJECT_DIR = Path(__file__).resolve().parents[4]

# Local copies of the benchmark's Isaac 5.1 assets.
NEWTON_ROBOT_USD = PROJECT_DIR / "assets" / "hx5_d20_left_original_newton" / "hx5_d20_left.usda"
NEWTON_CUBE_USD = PROJECT_DIR / "assets" / "dex_cube_original_newton" / "dex_cube.usda"
BENCHMARK_CUBE_USD = PROJECT_DIR / "assets" / "dex_cube_original_newton" / "source" / "dex_cube_instanceable.usd"

# Original robot USD (used by validate_setup)
ORIGINAL_ROBOT_USD = (
    PROJECT_DIR
    / "robotis_hand"
    / "robotis_hand_description"
    / "urdf"
    / "hx5_d20_left_2"
    / "hx5_d20_left_2"
    / "hx5_d20_left_2.usd"
)

# Goal marker position for diagnostics / play script
GOAL_MARKER_POSITION = (-0.2, -0.45, 0.68)

# DexCube physical properties (for validate_setup)
DEX_CUBE_SIDE_LENGTH = 0.048
DEX_CUBE_MASS = 0.077
DEX_CUBE_DENSITY = DEX_CUBE_MASS / DEX_CUBE_SIDE_LENGTH**3
DEX_CUBE_DIAGONAL_INERTIA = DEX_CUBE_MASS * DEX_CUBE_SIDE_LENGTH**2 / 6.0

# Provisional joint-space reflected inertia for Sim2Real training.  The
# deterministic benchmark keeps the serialized zero value; only the robust
# task uses this nominal and randomizes around it.
SIM2REAL_ACTUATOR_ARMATURE = 0.0009

ACTUATED_JOINT_NAMES = [
    "finger_l_joint5",
    "finger_l_joint6",
    "finger_l_joint7",
    "finger_l_joint8",
    "finger_l_joint9",
    "finger_l_joint10",
    "finger_l_joint11",
    "finger_l_joint12",
    "finger_l_joint13",
    "finger_l_joint14",
    "finger_l_joint15",
    "finger_l_joint16",
    "finger_l_joint17",
    "finger_l_joint18",
    "finger_l_joint19",
    "finger_l_joint20",
    "finger_l_joint1",
    "finger_l_joint2",
    "finger_l_joint3",
    "finger_l_joint4",
]

FINGERTIP_BODY_NAMES = [
    "finger_l_link8",
    "finger_l_link12",
    "finger_l_link16",
    "finger_l_link20",
    "finger_l_link4",
]

DEFAULT_JOINT_POSITIONS = {
    "finger_l_joint5": -0.6,
    "finger_l_joint6": 0.0,
    "finger_l_joint7": 0.0,
    "finger_l_joint8": 0.0,
    "finger_l_joint9": -0.6,
    "finger_l_joint10": 0.0,
    "finger_l_joint11": 0.0,
    "finger_l_joint12": 0.0,
    "finger_l_joint13": -0.6,
    "finger_l_joint14": 0.0,
    "finger_l_joint15": 0.0,
    "finger_l_joint16": 0.0,
    "finger_l_joint17": -0.6,
    "finger_l_joint18": 0.0,
    "finger_l_joint19": 0.0,
    "finger_l_joint20": 0.0,
    "finger_l_joint1": 0.2,
    "finger_l_joint2": 0.15,
    "finger_l_joint3": 0.0,
    "finger_l_joint4": 0.0,
}

# Robot root pose — wxyz quaternion (0.5, 0.5, -0.5, 0.5) → xyzw (0.5, -0.5, 0.5, 0.5)
ROBOT_ROOT_POSITION = (0.0, -0.3, 0.5)
ROBOT_ROOT_ORIENTATION_XYZW = (0.5, -0.5, 0.5, 0.5)
OBJECT_INITIAL_POSITION = (0.0, -0.42, 0.6)

##
# PhysX robot USD path
##

_PHYSX_ROBOT_USD = (
    PROJECT_DIR
    / "robotis_hand"
    / "robotis_hand_description"
    / "urdf"
    / "hx5_d20_left_2"
    / "hx5_d20_left_2"
    / "hx5_d20_left_2.usd"
)


def _physx_robot_usd() -> str:
    """Return the PhysX-compatible Robotis hand USD path."""
    return str(_PHYSX_ROBOT_USD)


##
# Event randomization presets
##


@configclass
class RobotisHandEventCfg:
    """Domain randomization events for the Robotis hand cube task (PhysX)."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
            "make_consistent": True,
        },
    )
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
            "make_consistent": True,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    robot_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


##
# Robot configuration preset
##


@configclass
class RobotisHandRobotCfg(PresetCfg):
    """Robotis HX5 hand articulation configuration.

    ``physx`` uses the original USD converted from URDF for PhysX training.
    ``newton_mjwarp`` uses the Newton-prepared USDA for sim-to-sim playback.
    """

    physx: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_PHYSX_ROBOT_USD),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=True,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
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
                damping=0.1,
            )
        },
        soft_joint_pos_limit_factor=1.0,
    )
    default: ArticulationCfg = physx


##
# Object configuration preset
##


@configclass
class RobotisHandObjectCfg(PresetCfg):
    """DexCube object configuration presets for PhysX and Newton backends."""

    physx: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(BENCHMARK_CUBE_USD),
            scale=(0.8, 0.8, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=DEX_CUBE_MASS, density=DEX_CUBE_DENSITY),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_INITIAL_POSITION, rot=(0.0, 0.0, 0.0, 1.0)),
    )
    default: RigidObjectCfg = physx


##
# Scene configuration preset
##


@configclass
class RobotisHandSceneCfg(PresetCfg):
    """Scene config — PhysX uses Fabric cloning, Newton does not."""

    physx: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=True
    )
    default: InteractiveSceneCfg = physx


##
# Physics configuration preset
##


@configclass
class RobotisHandPhysicsCfg(PresetCfg):
    """Physics backend configuration presets."""

    physx: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.2,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=163840,
    )
    default: PhysxCfg = physx


##
# Main environment configuration
##


@configclass
class RobotisHandEnvCfg(DirectRLEnvCfg):
    """PhysX training environment for the Robotis HX5 cube reorientation task.

    This environment uses the benchmark-compatible Robotis runtime and is
    designed to be trained with RSL-RL via::

        ./isaaclab.sh train --rl_library rsl_rl \\
            --task Isaac-Repose-Cube-Robotis-Direct-v0

    Domain randomization can be enabled by using the robust-train variant:

        ./isaaclab.sh train --rl_library rsl_rl \\
            --task Isaac-Repose-Cube-Robotis-Robust-Train-v0
    """

    # env
    seed = 42
    decimation = 4
    episode_length_s = 10.0
    action_space = POLICY_ACTION_DIM
    observation_space = POLICY_OBSERVATION_DIM
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            compliant_contact_stiffness=0.0,
            compliant_contact_damping=0.0,
        ),
        physics=RobotisHandPhysicsCfg(),
    )

    # robot
    robot_cfg: RobotisHandRobotCfg = RobotisHandRobotCfg()
    actuated_joint_names = ACTUATED_JOINT_NAMES
    fingertip_body_names = FINGERTIP_BODY_NAMES

    # object
    object_cfg: RobotisHandObjectCfg = RobotisHandObjectCfg()

    # goal marker — reuses the DexCube visual
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=str(BENCHMARK_CUBE_USD),
                scale=(0.8, 0.8, 0.8),
            )
        },
    )

    # scene
    scene: RobotisHandSceneCfg = RobotisHandSceneCfg()

    # reset noise
    reset_position_noise = 0.01
    reset_rotation_noise = 1.0
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.0

    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0003
    reach_goal_bonus = 250.0
    fall_penalty = 0.0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.2
    success_hold_steps: int = 1
    terminate_on_success: bool = False
    reset_target_on_success: bool = True
    success_requires_not_fallen: bool = False
    reward_mode: str = "proximity"
    max_consecutive_success = 0
    success_count_threshold: int = 1
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0
