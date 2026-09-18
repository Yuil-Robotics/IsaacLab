# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Yuil Dog flat-ground locomotion task with MoE-CTS RobotLab reward system."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from . import low_profile_rewards, robotlab_rewards, rough_mdp
from .asset_cfg import (
    YUIL_DOG_FOOT_NAMES,
    YUIL_DOG_JOINT_NAMES,
    YUIL_DOG_TROT_FOOT_PAIRS,
)
from .yuil_dog_cfg import (
    _apply_yuil_dog_randomization,
    _apply_yuil_dog_robot,
    _apply_yuil_dog_terminations,
    _apply_yuil_dog_velocity_command,
)

YUIL_DOG_FLAT_ROBOTLAB_TRAIN_TASK_ID = "Isaac-Velocity-Flat-Yuil-Dog-RobotLab-v0"
YUIL_DOG_FLAT_ROBOTLAB_PLAY_TASK_ID = "Isaac-Velocity-Flat-Yuil-Dog-RobotLab-Play-v0"
YUIL_DOG_FLAT_ROBOTLAB_EVAL_TASK_ID = "Isaac-Velocity-Flat-Yuil-Dog-RobotLab-Eval-v0"
YUIL_DOG_FLAT_ROBOTLAB_RECOVERY_EVAL_TASK_ID = "Isaac-Velocity-Flat-Yuil-Dog-RobotLab-Recovery-Eval-v0"
YUIL_DOG_FLAT_ROBOTLAB_EXPERIMENT_NAME = "yuil_dog_flat_robotlab"

BASE_HEIGHT_MIN: float = 0.30
"""Minimum penalty-free clearance [m] from the base underside to the ground."""

BASE_HEIGHT_MAX: float = 0.32
"""Maximum penalty-free clearance [m] from the base underside to the ground."""

BASE_HEIGHT_TARGET: float = 0.5 * (BASE_HEIGHT_MIN + BASE_HEIGHT_MAX)
"""Midpoint [m] retained for reward terms that require a scalar height reference."""

BASE_BOTTOM_OFFSET_Z: float = -0.01205
"""Base underside Z offset [m] from the Yuil Dog base-link origin."""

BASE_ROOT_INITIAL_HEIGHT: float = 0.325
"""Initial base-link origin height [m] for the crouched standing pose."""

YUIL_DOG_FLAT_ROBOTLAB_HISTORY_LENGTH: int = 10
"""Number of observation steps stacked for the policy actor."""

YUIL_DOG_FLAT_ROBOTLAB_ACTOR_STEP_DIM: int = 45
"""Dimension of a single-step policy observation without base linear velocity."""

YUIL_DOG_FLAT_ROBOTLAB_FALL_HEIGHT: float = 0.18
"""Minimum base-link height [m] used for backend-independent playback termination."""

YUIL_DOG_FLAT_ROBOTLAB_PUSH_INTERVAL_RANGE_S: tuple[float, float] = (3.0, 8.0)
"""Non-constant randomized time interval [s] between external velocity pushes."""

YUIL_DOG_FLAT_ROBOTLAB_PUSH_VELOCITY_RANGE_XY: tuple[float, float] = (-0.5, 0.5)
"""Randomized horizontal base velocity impulse range [m/s] for push disturbance."""

YUIL_DOG_FLAT_ROBOTLAB_PUSH_ANGULAR_VELOCITY_RANGE: tuple[float, float] = (0.0, 0.0)
"""Disabled angular-velocity impulse range [rad/s] retained for compatibility."""

YUIL_DOG_FLAT_ROBOTLAB_PUSH_RECOVERY_WINDOW_S: float = 0.50
"""Legacy recovery-window duration [s] retained for compatibility."""

YUIL_DOG_FLAT_ROBOTLAB_EXTERNAL_FORCE_RANGE: tuple[float, float] = (0.0, 0.0)
"""Disabled external-force range [N] from the reference run."""

YUIL_DOG_FLAT_ROBOTLAB_EXTERNAL_TORQUE_RANGE: tuple[float, float] = (0.0, 0.0)
"""Disabled external-torque range [N·m] from the reference run."""

YUIL_DOG_FLAT_ROBOTLAB_EXTERNAL_WRENCH_INTERVAL_RANGE_S: tuple[float, float] = (10.0, 15.0)
"""Reference interval [s] between zero-valued external-wrench writes."""

YUIL_DOG_FLAT_ROBOTLAB_EXTERNAL_WRENCH_DURATION_RANGE_S: tuple[float, float] = (0.12, 0.45)
"""Legacy wrench-pulse duration [s] retained for compatibility."""

YUIL_DOG_FLAT_ROBOTLAB_EXTERNAL_WRENCH_RECOVERY_SETTLE_TIME_S: float = 0.40
"""Legacy recovery settling time [s] retained for compatibility."""

COMMAND_LIN_VEL_X_RANGE: tuple[float, float] = (-1.0, 1.0)
"""Full randomized command linear velocity range in X [m/s] matching 2026-09-09 reference."""

COMMAND_LIN_VEL_Y_RANGE: tuple[float, float] = (-0.5, 0.5)
"""Full randomized command linear velocity range in Y [m/s] matching 2026-09-09 reference."""

COMMAND_ANG_VEL_Z_RANGE: tuple[float, float] = (-1.0, 1.0)
"""Full randomized command angular velocity range in Z [rad/s] matching 2026-09-09 reference."""

YUIL_DOG_FLAT_ROBOTLAB_MAXIMUM_CADENCE_HZ: float = 3.0
"""Maximum penalty-free trot cycle frequency [Hz]."""

YUIL_DOG_FLAT_ROBOTLAB_MINIMUM_TARGET_CADENCE_HZ: float = 2.0
"""Minimum recommended trot-cycle frequency for non-zero motion commands [Hz]."""

YUIL_DOG_FLAT_ROBOTLAB_MAXIMUM_CADENCE_SPEED: float = 0.8
"""Equivalent command speed reaching the maximum target cadence [m/s]."""

YUIL_DOG_FLAT_ROBOTLAB_MINIMUM_LATERAL_FOOT_SEPARATION: float = 0.20
"""Minimum penalty-free FL-FR and RL-RR lateral foot separation [m]."""

YUIL_DOG_FLAT_ROBOTLAB_FOOT_SEPARATION_MARGIN: float = 0.10
"""Foot separation shortfall [m] corresponding to a unit normalized penalty."""


@configclass
class PrivilegedVelocityCfg(ObsGroup):
    """Current base linear velocity available only to the critic [m/s]."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

    def __post_init__(self) -> None:
        """Keep privileged state uncorrupted and unstacked."""
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 0


def _apply_yuil_dog_flat_robotlab_robot(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Configure the nominal robot and action contract used by the reference run."""
    _apply_yuil_dog_robot(env_cfg)


def _apply_yuil_dog_flat_robotlab_observations(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Configure the 450D actor and privileged critic contracts from the reference run."""
    env_cfg.observations.policy.base_lin_vel = None
    env_cfg.observations.policy.history_length = YUIL_DOG_FLAT_ROBOTLAB_HISTORY_LENGTH
    env_cfg.observations.policy.flatten_history_dim = True
    env_cfg.observations.privileged = PrivilegedVelocityCfg()


def _apply_yuil_dog_flat_terrain(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Configure a plane and a downward ray caster mounted on the base underside."""
    env_cfg.scene.terrain.class_type = "go2_sim2sim.terrain:LocalPlaneTerrainImporter"
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.20, 0.20, 0.20),
        roughness=0.8,
    )
    height_scanner = env_cfg.scene.height_scanner
    if height_scanner is None:
        raise RuntimeError("Expected a height scanner for base-bottom clearance measurement.")
    height_scanner.offset.pos = (0.0, 0.0, BASE_BOTTOM_OFFSET_Z)
    height_scanner.ray_alignment = "yaw"
    height_scanner.pattern_cfg.resolution = 0.1
    height_scanner.pattern_cfg.size = (0.0, 0.0)
    height_scanner.max_distance = 1.0
    height_scanner.debug_vis = False
    env_cfg.observations.policy.height_scan = None
    env_cfg.curriculum.terrain_levels = None


def _apply_yuil_dog_flat_robotlab_initial_pose(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Configure the symmetric crouched pose used as the policy action offset."""
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, BASE_ROOT_INITIAL_HEIGHT)
    env_cfg.scene.robot.init_state.joint_pos = {
        "FL_hip_roll": -0.05,
        "FR_hip_roll": 0.05,
        "RL_hip_roll": 0.05,
        "RR_hip_roll": -0.05,
        ".*L_hip_pitch": -0.7,
        ".*R_hip_pitch": 0.7,
        ".*L_knee_pitch": 0.7,
        ".*R_knee_pitch": -0.7,
    }


def _apply_yuil_dog_flat_robotlab_external_disturbances(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Apply the velocity pushes and disabled wrench from the reference run."""
    env_cfg.events.push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=YUIL_DOG_FLAT_ROBOTLAB_PUSH_INTERVAL_RANGE_S,
        params={
            "velocity_range": {
                "x": YUIL_DOG_FLAT_ROBOTLAB_PUSH_VELOCITY_RANGE_XY,
                "y": YUIL_DOG_FLAT_ROBOTLAB_PUSH_VELOCITY_RANGE_XY,
            },
        },
    )
    env_cfg.events.base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=YUIL_DOG_FLAT_ROBOTLAB_EXTERNAL_WRENCH_INTERVAL_RANGE_S,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": YUIL_DOG_FLAT_ROBOTLAB_EXTERNAL_FORCE_RANGE,
            "torque_range": YUIL_DOG_FLAT_ROBOTLAB_EXTERNAL_TORQUE_RANGE,
        },
    )


def _apply_yuil_dog_flat_robotlab_randomization(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Apply the contact and rigid-body randomization used by the reference run."""
    _apply_yuil_dog_randomization(env_cfg)


def _apply_yuil_dog_flat_robotlab_velocity_command(
    env_cfg: LocomotionVelocityRoughEnvCfg, debug_vis: bool = False
) -> None:
    """Configure the command distribution used by the reference run."""
    _apply_yuil_dog_velocity_command(env_cfg, debug_vis=debug_vis)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.10
    env_cfg.commands.base_velocity.rel_straight_envs = 0.20
    env_cfg.commands.base_velocity.rel_turning_envs = 0.20
    env_cfg.commands.base_velocity.rel_lateral_envs = 0.20
    env_cfg.commands.base_velocity.ranges.lin_vel_x = COMMAND_LIN_VEL_X_RANGE
    env_cfg.commands.base_velocity.ranges.lin_vel_y = COMMAND_LIN_VEL_Y_RANGE
    env_cfg.commands.base_velocity.ranges.ang_vel_z = COMMAND_ANG_VEL_Z_RANGE


def _apply_yuil_dog_flat_robotlab_rewards(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Configure the RobotLab reward suite with explicit swing-foot clearance."""
    # Clear all default reward terms to avoid interference
    for attr in list(vars(env_cfg.rewards).keys()):
        if not attr.startswith("_"):
            setattr(env_cfg.rewards, attr, None)

    joint_names = list(YUIL_DOG_JOINT_NAMES)
    foot_names = list(YUIL_DOG_FOOT_NAMES)
    robot_asset_cfg = SceneEntityCfg("robot")
    all_joints_asset_cfg = SceneEntityCfg("robot", joint_names=joint_names, preserve_order=True)
    hip_joints_asset_cfg = SceneEntityCfg("robot", joint_names=[".*_hip_roll"])
    leg_joints_asset_cfg = SceneEntityCfg("robot", joint_names=[".*_hip_pitch", ".*_knee_pitch"])
    foot_asset_cfg = SceneEntityCfg("robot", body_names=foot_names, preserve_order=True)
    foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=foot_names)
    height_sensor_cfg = SceneEntityCfg("height_scanner")
    undesired_contact_sensor_cfg = SceneEntityCfg("contact_forces", body_names=[".*_hip_pitch", ".*_knee_pitch"])

    # 1. track_lin_vel_xy_exp (+2.0): XY velocity command tracking
    env_cfg.rewards.track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5, "asset_cfg": robot_asset_cfg},
    )

    # 2. track_ang_vel_z_exp (+1.0): Yaw angular velocity command tracking
    env_cfg.rewards.track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5, "asset_cfg": robot_asset_cfg},
    )

    # 3. lin_vel_z_l2 (-2.0 -> -1.0): Vertical bouncing penalty (relaxed by curriculum)
    env_cfg.rewards.lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
        params={"asset_cfg": robot_asset_cfg},
    )

    # 4. ang_vel_xy_l2 (-0.05): Roll/pitch oscillation penalty
    env_cfg.rewards.ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": robot_asset_cfg},
    )

    # 5. joint_acc_l2 (-1e-7): Joint acceleration penalty
    env_cfg.rewards.joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": all_joints_asset_cfg},
    )

    # 6. joint_power (-2e-5): Total mechanical joint power penalty
    env_cfg.rewards.joint_power = RewTerm(
        func=robotlab_rewards.joint_power,
        weight=-2.0e-5,
        params={"asset_cfg": all_joints_asset_cfg},
    )

    # 7. joint_torques_l2 (-1e-4): Applied joint torque magnitude penalty
    env_cfg.rewards.joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-4,
        params={"asset_cfg": all_joints_asset_cfg},
    )

    # 8. base_height_l2 (-5.0 -> -20.0): Keep base-bottom clearance in the 0.30-0.32 m range
    env_cfg.rewards.base_height_l2 = RewTerm(
        func=robotlab_rewards.base_bottom_height_range_l2,
        weight=-20.0,
        params={
            "minimum_height": BASE_HEIGHT_MIN,
            "maximum_height": BASE_HEIGHT_MAX,
            "sensor_cfg": height_sensor_cfg,
        },
    )

    # 9. action_rate_l2 (-0.01): First-order action rate penalty
    env_cfg.rewards.action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # 10. action_smoothness_l2 (-0.01): Second-order action rate penalty
    env_cfg.rewards.action_smoothness_l2 = RewTerm(
        func=robotlab_rewards.action_smoothness_l2,
        weight=-0.01,
    )

    # 11. undesired_contacts (-1.0): Thigh and calf ground collision penalty
    env_cfg.rewards.undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": undesired_contact_sensor_cfg, "threshold": 5.0},
    )

    # 12. joint_pos_limits (-2.0): Joint hardware limit proximity penalty
    env_cfg.rewards.joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": all_joints_asset_cfg},
    )

    # 13. feet_regulation (-0.10): Low-altitude foot dragging / scuffing penalty
    env_cfg.rewards.feet_regulation = RewTerm(
        func=robotlab_rewards.feet_regulation,
        weight=-0.10,
        params={
            "base_height_target": BASE_HEIGHT_TARGET,
            "asset_cfg": foot_asset_cfg,
            "sensor_cfg": height_sensor_cfg,
        },
    )

    # 14. feet_lateral_separation_l2 (-0.50): Keep front/rear left-right foot gaps at least 0.20 m
    env_cfg.rewards.feet_lateral_separation_l2 = RewTerm(
        func=robotlab_rewards.feet_lateral_separation_l2,
        weight=-0.5,
        params={
            "minimum_separation": YUIL_DOG_FLAT_ROBOTLAB_MINIMUM_LATERAL_FOOT_SEPARATION,
            "separation_margin": YUIL_DOG_FLAT_ROBOTLAB_FOOT_SEPARATION_MARGIN,
            "asset_cfg": foot_asset_cfg,
        },
    )

    # 15. foot_clearance (+0.30): Dense guidance toward the reference 5 cm swing trajectory
    env_cfg.rewards.foot_clearance = RewTerm(
        func=rough_mdp.RoughTerrainFootClearanceReward,
        weight=0.30,
        params={
            "asset_cfg": foot_asset_cfg,
            "sensor_cfg": foot_sensor_cfg,
            "target_height": 0.05,
            "std": 0.03,
            "motion_start": 0.05,
            "motion_full": 0.20,
            "yaw_scale": 0.25,
            "recovery_velocity_threshold": 0.4,
            "command_name": "base_velocity",
            "flight_grace_time": 0.04,
            "flight_full_time": 0.10,
        },
    )

    # 16. gait (+0.60): Diagonal trot timing preference for translation and yaw motion
    env_cfg.rewards.gait = RewTerm(
        func=rough_mdp.RoughTerrainGaitReward,
        weight=0.6,
        params={
            "std": 0.10,
            "max_err": 0.3,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": YUIL_DOG_TROT_FOOT_PAIRS,
            "asset_cfg": robot_asset_cfg,
            "sensor_cfg": foot_sensor_cfg,
            "motion_start": 0.05,
            "motion_full": 0.20,
            "yaw_scale": 0.25,
            "flight_grace_time": 0.04,
            "flight_full_time": 0.10,
        },
    )

    # 17. cadence_uniformity (-0.10): Keep consecutive diagonal-pair touchdown intervals even
    env_cfg.rewards.cadence_uniformity = RewTerm(
        func=rough_mdp.TrotCadenceUniformityPenalty,
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "synced_feet_pair_names": YUIL_DOG_TROT_FOOT_PAIRS,
            "sensor_cfg": foot_sensor_cfg,
            "motion_start": 0.05,
            "motion_full": 0.20,
            "yaw_scale": 0.25,
            "interval_tolerance": 0.05,
            "min_touchdown_interval": 0.04,
            "max_touchdown_interval": 0.60,
            "command_change_threshold": 0.05,
        },
    )

    # 18. cadence_target (-0.15): Map combined command magnitude smoothly onto 2.0-3.0 Hz
    env_cfg.rewards.cadence_target = RewTerm(
        func=rough_mdp.CommandSpeedTrotCadenceTargetPenalty,
        weight=-0.15,
        params={
            "command_name": "base_velocity",
            "synced_feet_pair_names": YUIL_DOG_TROT_FOOT_PAIRS,
            "sensor_cfg": foot_sensor_cfg,
            "motion_start": 0.05,
            "motion_full": 0.20,
            "yaw_scale": 0.25,
            "minimum_cycle_frequency": YUIL_DOG_FLAT_ROBOTLAB_MINIMUM_TARGET_CADENCE_HZ,
            "maximum_cycle_frequency": YUIL_DOG_FLAT_ROBOTLAB_MAXIMUM_CADENCE_HZ,
            "maximum_frequency_speed": YUIL_DOG_FLAT_ROBOTLAB_MAXIMUM_CADENCE_SPEED,
            "interval_tolerance": 0.05,
            "min_touchdown_interval": 0.04,
            "max_touchdown_interval": 0.60,
            "command_change_threshold": 0.05,
        },
    )

    # 19. cadence_ceiling (-0.50): Penalize only trot cycles faster than 3.0 Hz
    env_cfg.rewards.cadence_ceiling = RewTerm(
        func=low_profile_rewards.TrotCadenceCeilingPenalty,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "synced_feet_pair_names": YUIL_DOG_TROT_FOOT_PAIRS,
            "sensor_cfg": foot_sensor_cfg,
            "motion_start": 0.05,
            "motion_full": 0.20,
            "yaw_scale": 0.25,
            "maximum_cycle_frequency": YUIL_DOG_FLAT_ROBOTLAB_MAXIMUM_CADENCE_HZ,
            "interval_tolerance": 0.06,
            "min_touchdown_interval": 0.04,
            "max_touchdown_interval": 0.60,
            "command_change_threshold": 0.05,
        },
    )

    # 20. hip_pos_penalty_l1 (-0.10): Suppress hip-roll motion
    env_cfg.rewards.hip_pos_penalty_l1 = RewTerm(
        func=robotlab_rewards.hip_pos_penalty_l1,
        weight=-0.10,
        params={
            "command_name": "base_velocity",
            "asset_cfg": hip_joints_asset_cfg,
            "stand_still_scale": 1.0,
        },
    )

    # 21. joint_pos_penalty_l1 (-0.01): Thigh/calf pitch deviation penalty from default
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

    # 22. stand_still_joint_pos_l1 (-0.05): Restore the symmetric default pose at zero command
    env_cfg.rewards.stand_still_joint_pos_l1 = RewTerm(
        func=robotlab_rewards.stand_still_joint_pos_l1,
        weight=-0.05,
        params={
            "command_name": "base_velocity",
            "asset_cfg": all_joints_asset_cfg,
            "command_threshold": 0.05,
            "yaw_scale": 0.25,
        },
    )

    # 23. flat_orientation_l2 (-2.0): Penalize static roll and pitch tilt
    env_cfg.rewards.flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-2.0,
        params={"asset_cfg": robot_asset_cfg},
    )

    # 24. feet_impact_vel_l2 (-0.60): Penalize touchdown speed above 0.35 m/s
    env_cfg.rewards.feet_impact_vel_l2 = RewTerm(
        func=robotlab_rewards.FeetImpactVelocityPenalty,
        weight=-0.6,
        params={
            "asset_cfg": foot_asset_cfg,
            "sensor_cfg": foot_sensor_cfg,
            "velocity_threshold": 0.35,
        },
    )

    # 25. feet_contact_force_limit (-0.01): Penalize force above the reference 120 N threshold
    env_cfg.rewards.feet_contact_force_limit = RewTerm(
        func=robotlab_rewards.feet_contact_force_limit,
        weight=-0.01,
        params={
            "sensor_cfg": foot_sensor_cfg,
            "threshold": 120.0,
        },
    )


def _apply_yuil_dog_flat_robotlab_curriculum(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Configure linear curriculum schedules matching the MoE-CTS RobotLab reference."""
    # Relax vertical-motion regularization as recovery stepping becomes stable.
    env_cfg.curriculum.lin_vel_z_l2 = CurrTerm(
        func=robotlab_rewards.gradual_reward_weight_modification,
        params={
            "term_name": "lin_vel_z_l2",
            "initial_weight": -1.0,
            "final_weight": -1.0,
            "start_it": 0,
            "end_it": 1500,
        },
    )
    # Tighten height regulation to the final reference-run weight.
    env_cfg.curriculum.base_height_l2 = CurrTerm(
        func=robotlab_rewards.gradual_reward_weight_modification,
        params={
            "term_name": "base_height_l2",
            "initial_weight": -20.0,
            "final_weight": -20.0,
            "start_it": 0,
            "end_it": 3000,
        },
    )
    # Introduce the new speed-dependent cadence target gradually when resuming a trained policy.
    env_cfg.curriculum.cadence_target = CurrTerm(
        func=robotlab_rewards.gradual_reward_weight_modification,
        params={
            "term_name": "cadence_target",
            "initial_weight": -0.15,
            "final_weight": -0.15,
            "start_it": 0,
            "end_it": 1000,
        },
    )


def _apply_yuil_dog_flat_robotlab_play_terminations(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Use a backend-independent fall condition for playback and Sim2Sim evaluation."""
    env_cfg.terminations.base_contact = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": YUIL_DOG_FLAT_ROBOTLAB_FALL_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


def _use_nominal_actuation(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Keep the nominal actuator setup used by training."""
    env_cfg.events.actuator_gains = None
    env_cfg.events.joint_friction = None


def _restore_reference_disturbances(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Re-enable the reference-run disturbances for playback and recovery evaluation."""
    _apply_yuil_dog_flat_robotlab_external_disturbances(env_cfg)


@configclass
class YuilDogFlatRobotLabTrainEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Yuil Dog flat-ground training task with MoE-CTS RobotLab reward architecture."""

    def __post_init__(self) -> None:
        """Initialize the flat scene, robot kinematics, commands, and RobotLab rewards."""
        super().__post_init__()
        self.scene.num_envs = 4096
        _apply_yuil_dog_flat_robotlab_robot(self)
        _apply_yuil_dog_flat_robotlab_initial_pose(self)
        _apply_yuil_dog_flat_terrain(self)
        _apply_yuil_dog_flat_robotlab_observations(self)
        _apply_yuil_dog_flat_robotlab_randomization(self)
        _apply_yuil_dog_flat_robotlab_external_disturbances(self)
        _apply_yuil_dog_flat_robotlab_velocity_command(self, debug_vis=False)
        _apply_yuil_dog_flat_robotlab_rewards(self)
        _apply_yuil_dog_flat_robotlab_curriculum(self)
        _apply_yuil_dog_terminations(self)
        self.terminations.terrain_out_of_bounds = None


@configclass
class YuilDogFlatRobotLabPlayEnvCfg(YuilDogFlatRobotLabTrainEnvCfg):
    """Yuil Dog flat-ground recovery playback task with RobotLab configuration."""

    def __post_init__(self) -> None:
        """Configure reference-run disturbances on a small flat scene."""
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.commands.base_velocity.debug_vis = True
        self.observations.policy.enable_corruption = False
        _restore_reference_disturbances(self)
        _use_nominal_actuation(self)
        _apply_yuil_dog_flat_robotlab_play_terminations(self)


@configclass
class YuilDogFlatRobotLabEvalEnvCfg(YuilDogFlatRobotLabTrainEnvCfg):
    """Yuil Dog flat-ground deterministic evaluation task with RobotLab configuration."""

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
        _use_nominal_actuation(self)
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
        _apply_yuil_dog_flat_robotlab_play_terminations(self)


@configclass
class YuilDogFlatRobotLabRecoveryEvalEnvCfg(YuilDogFlatRobotLabEvalEnvCfg):
    """Nominal deterministic-reset evaluation with reference-run disturbances."""

    def __post_init__(self) -> None:
        """Retain clean observations and re-enable controlled recovery disturbances."""
        super().__post_init__()
        _restore_reference_disturbances(self)


@configclass
class YuilDogFlatRobotLabPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    """Yuil Dog flat-ground PPO runner configuration for the RobotLab reward task."""

    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy", "privileged"],
    }
    experiment_name = YUIL_DOG_FLAT_ROBOTLAB_EXPERIMENT_NAME
    max_iterations = 5000
    save_interval = 50
    clip_actions = 3.5

    def __post_init__(self) -> None:
        """Apply experiment name, asymmetric observation mapping, and training budget."""
        super().__post_init__()
        self.obs_groups = {
            "actor": ["policy"],
            "critic": ["policy", "privileged"],
        }
        self.experiment_name = YUIL_DOG_FLAT_ROBOTLAB_EXPERIMENT_NAME
        self.max_iterations = 5000
        self.save_interval = 50
        self.clip_actions = 3.5
