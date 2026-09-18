# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone Yuil Dog rough-terrain task using dedicated architecture and kinematics."""

from __future__ import annotations

import copy

import isaaclab.sim as sim_utils
from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors.ray_caster.patterns import grid_pattern
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import GridPatternCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2RoughPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from . import rough_mdp
from .asset_cfg import (
    YUIL_DOG_BASE_PRIM_PATH,
    YUIL_DOG_CFG,
    YUIL_DOG_FOOT_NAMES,
    YUIL_DOG_JOINT_NAMES,
    YUIL_DOG_TROT_FOOT_PAIRS,
)
from .rough_cfg import CUSTOM_ROUGH_TERRAINS_CFG
from .visualization import CURRENT_VELOCITY_MARKER_CFG, GOAL_VELOCITY_MARKER_CFG

YUIL_DOG_ROUGH_TRAIN_TASK_ID = "Isaac-Velocity-Rough-Yuil-Dog-v0"
YUIL_DOG_ROUGH_PLAY_TASK_ID = "Isaac-Velocity-Rough-Yuil-Dog-Play-v0"
YUIL_DOG_ROUGH_EVAL_TASK_ID = "Isaac-Velocity-Rough-Yuil-Dog-Eval-v0"
YUIL_DOG_ROUGH_EXPERIMENT_NAME = "yuil_dog_rough_leg_major"
YUIL_DOG_CONTACT_SENSOR_PRIM_PATH = f"{YUIL_DOG_BASE_PRIM_PATH}/.*"
YUIL_DOG_PHYSX_CONTACT_SENSOR = "go2_sim2sim.hierarchical_contact_sensor:HierarchicalContactSensor"

# Hardware-safe joint position clamping (radians)
# Limits derived from USD mechanical limits with safety margins:
# - Hip Roll: FL/RR [-25°, +30°], FR/RL [-30°, +25°] -> clamped to [-0.38, 0.46] / [-0.46, 0.38]
# - Hip Pitch: Left [-120°, +80°], Right [-80°, +120°] -> clamped to [-1.80, 1.20] / [-1.20, 1.80]
# - Knee Pitch: Left [0°, +100°], Right [-100°, 0°]
#   -> clamped to [0.05, 1.57] / [-1.57, -0.05] (prevents 0° hyperextension)
YUIL_DOG_JOINT_POSITION_CLIP: dict[str, tuple[float, float]] = {
    "FL_hip_roll": (-0.38, 0.46),
    "FR_hip_roll": (-0.46, 0.38),
    "RL_hip_roll": (-0.46, 0.38),
    "RR_hip_roll": (-0.38, 0.46),
    ".*L_hip_pitch": (-1.80, 1.20),
    ".*R_hip_pitch": (-1.20, 1.80),
    ".*L_knee_pitch": (0.05, 1.57),
    ".*R_knee_pitch": (-1.57, -0.05),
}


def _apply_yuil_dog_robot(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Configure the Yuil Dog articulation, action contract, and contact sensor."""
    env_cfg.scene.robot = YUIL_DOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    if env_cfg.scene.robot.spawn.articulation_props is None:
        env_cfg.scene.robot.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False
        )
    else:
        env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False

    joint_names = list(YUIL_DOG_JOINT_NAMES)
    env_cfg.actions.joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=joint_names,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
        clip=YUIL_DOG_JOINT_POSITION_CLIP,
    )

    joint_asset_cfg = SceneEntityCfg("robot", joint_names=joint_names, preserve_order=True)
    env_cfg.observations.policy.joint_pos.params["asset_cfg"] = joint_asset_cfg
    env_cfg.observations.policy.joint_vel.params["asset_cfg"] = joint_asset_cfg

    # Blind proprioceptive contract for zero-shot real deployment:
    # Retain height_scanner in the scene strictly for base_height_l2 reward computation,
    # but remove height_scan from policy observations so the policy remains blind.
    if env_cfg.scene.height_scanner is not None:
        env_cfg.scene.height_scanner.prim_path = YUIL_DOG_BASE_PRIM_PATH
        env_cfg.scene.height_scanner.pattern_cfg = GridPatternCfg(func=grid_pattern, resolution=0.1, size=[0.1, 0.1])
    env_cfg.observations.policy.height_scan = None

    # The nested-body workaround is PhysX-specific. Keep the backend-native
    # contact sensor classes for Newton/MJWarp and other physics presets.
    for preset_name, contact_cfg in vars(env_cfg.scene.contact_forces).items():
        if hasattr(contact_cfg, "prim_path"):
            contact_cfg.prim_path = YUIL_DOG_CONTACT_SENSOR_PRIM_PATH
        if preset_name in ("default", "physx") and hasattr(contact_cfg, "class_type"):
            contact_cfg.class_type = YUIL_DOG_PHYSX_CONTACT_SENSOR


def _apply_yuil_dog_terrain(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Apply scaled rough terrain generator without curriculum."""
    env_cfg.scene.terrain.terrain_generator = copy.deepcopy(CUSTOM_ROUGH_TERRAINS_CFG)
    env_cfg.scene.terrain.terrain_generator.curriculum = False
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.20, 0.20, 0.20),
        roughness=0.8,
    )
    env_cfg.curriculum.terrain_levels = None


def _apply_yuil_dog_velocity_command(env_cfg: LocomotionVelocityRoughEnvCfg, debug_vis: bool = False) -> None:
    """Configure fixed-interval velocity commands with dedicated motion modes."""
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.class_type = "go2_sim2sim.velocity_command:VisibleVelocityCommand"
    command_cfg.debug_vis = debug_vis
    command_cfg.rel_standing_envs = 0.10
    command_cfg.rel_straight_envs = 0.20
    command_cfg.rel_turning_envs = 0.20
    command_cfg.rel_lateral_envs = 0.20
    command_cfg.heading_command = False
    command_cfg.rel_heading_envs = 0.0
    command_cfg.resampling_time_range = (10.0, 10.0)
    command_cfg.ranges.lin_vel_x = (-1.0, 1.0)
    command_cfg.ranges.lin_vel_y = (-0.5, 0.5)
    command_cfg.ranges.ang_vel_z = (-1.0, 1.0)
    command_cfg.ranges.heading = None
    command_cfg.goal_vel_visualizer_cfg = GOAL_VELOCITY_MARKER_CFG.copy()
    command_cfg.current_vel_visualizer_cfg = CURRENT_VELOCITY_MARKER_CFG.copy()


def _apply_yuil_dog_randomization(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Apply Yuil Dog domain randomization without actuator gain/friction leakage."""
    base_asset_cfg = SceneEntityCfg("robot", body_names="base_link")
    env_cfg.events.physics_material.params.update(
        {
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.2, 0.8),
            "restitution_range": (0.0, 0.05),
            "make_consistent": True,
        }
    )
    if hasattr(env_cfg.events, "add_base_mass") and env_cfg.events.add_base_mass is not None:
        env_cfg.events.add_base_mass.params["asset_cfg"] = base_asset_cfg
    if hasattr(env_cfg.events, "base_com") and env_cfg.events.base_com is not None:
        base_com = env_cfg.events.base_com
        if hasattr(base_com, "params") and "asset_cfg" in base_com.params:
            base_com.params["asset_cfg"] = base_asset_cfg
        for sub in vars(base_com).values():
            if hasattr(sub, "params") and "asset_cfg" in sub.params:
                sub.params["asset_cfg"] = base_asset_cfg
    if hasattr(env_cfg.events, "base_external_force_torque") and env_cfg.events.base_external_force_torque is not None:
        env_cfg.events.base_external_force_torque.params["asset_cfg"] = base_asset_cfg
    if hasattr(env_cfg.events, "push_robot") and env_cfg.events.push_robot is not None:
        pass

    # Yuil Dog uses nominal RS03 and RS04 actuator configurations
    env_cfg.events.actuator_gains = None
    env_cfg.events.joint_friction = None


def _apply_yuil_dog_rewards(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Configure Yuil Dog reward suite with morphology-matched entities and terms."""
    # Remove upstream baseline terms to avoid objective conflicts
    env_cfg.rewards.track_lin_vel_xy_exp = None
    env_cfg.rewards.track_ang_vel_z_exp = None
    env_cfg.rewards.lin_vel_z_l2 = None
    env_cfg.rewards.ang_vel_xy_l2 = None
    env_cfg.rewards.dof_torques_l2 = None
    env_cfg.rewards.dof_acc_l2 = None
    env_cfg.rewards.action_rate_l2 = None
    env_cfg.rewards.feet_air_time = None
    env_cfg.rewards.flat_orientation_l2 = None
    env_cfg.rewards.dof_pos_limits = None

    foot_names = list(YUIL_DOG_FOOT_NAMES)
    joint_names = list(YUIL_DOG_JOINT_NAMES)
    joint_asset_cfg = SceneEntityCfg("robot", joint_names=joint_names, preserve_order=True)
    foot_asset_cfg = SceneEntityCfg("robot", body_names=foot_names)
    foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=foot_names)

    # Velocity tracking
    env_cfg.rewards.base_linear_velocity = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=4.0,
        params={"std": 0.5, "command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    env_cfg.rewards.base_angular_velocity = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"std": 0.5, "command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )

    # Stepping and clearance
    env_cfg.rewards.air_time = None
    env_cfg.rewards.air_time_variance = RewTerm(
        func=spot_mdp.air_time_variance_penalty,
        weight=-10.0,
        params={"sensor_cfg": foot_sensor_cfg},
    )
    env_cfg.rewards.foot_slip = RewTerm(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": foot_asset_cfg,
            "sensor_cfg": foot_sensor_cfg,
            "threshold": 5.0,
        },
    )
    env_cfg.rewards.foot_clearance = None
    env_cfg.rewards.gait = RewTerm(
        func=rough_mdp.RoughTerrainGaitReward,
        weight=2.5,
        params={
            "std": 0.2,
            "max_err": 0.3,
            "velocity_threshold": 0.5,
            "motion_start": 0.10,
            "motion_full": 0.40,
            "flight_grace_time": 0.04,
            "flight_full_time": 0.10,
            "synced_feet_pair_names": YUIL_DOG_TROT_FOOT_PAIRS,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )
    env_cfg.rewards.flight_phase = RewTerm(
        func=rough_mdp.flight_phase_penalty,
        weight=-3.0,
        params={
            "grace_time": 0.04,
            "full_time": 0.10,
            "sensor_cfg": foot_sensor_cfg,
        },
    )

    # Base stability
    env_cfg.rewards.base_motion = None
    env_cfg.rewards.base_height_l2 = RewTerm(
        func=rough_mdp.rough_base_height_l2,
        weight=-40.0,
        params={
            "target_height": 0.33,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    env_cfg.rewards.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    env_cfg.rewards.ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    env_cfg.rewards.base_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Posture and joint safety
    env_cfg.rewards.joint_pos = RewTerm(
        func=spot_mdp.joint_position_penalty,
        weight=-0.3,
        params={
            "asset_cfg": joint_asset_cfg,
            "stand_still_scale": 3.0,
            "velocity_threshold": 0.5,
        },
    )
    env_cfg.rewards.dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": joint_asset_cfg},
    )
    env_cfg.rewards.hip_joint_pos = None
    env_cfg.rewards.hip_straight_pos = None

    # Action smoothness and motor protection
    env_cfg.rewards.action_smoothness = RewTerm(func=spot_mdp.action_smoothness_penalty, weight=-1.0)
    env_cfg.rewards.joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": joint_asset_cfg},
    )
    env_cfg.rewards.joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-4,
        params={"asset_cfg": joint_asset_cfg},
    )
    env_cfg.rewards.joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": joint_asset_cfg},
    )
    env_cfg.rewards.hip_joint_vel = RewTerm(
        func=rough_mdp.straight_motion_hip_velocity_penalty,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll"]),
            "vy_threshold": 0.05,
            "wz_threshold": 0.05,
        },
    )
    env_cfg.rewards.hip_joint_torques = None

    # Collision avoidance
    env_cfg.rewards.undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_hip_roll", ".*_hip_pitch", ".*_knee_pitch"],
            ),
            "threshold": 5.0,
        },
    )


def _apply_yuil_dog_terminations(env_cfg: LocomotionVelocityRoughEnvCfg) -> None:
    """Configure terminations for base collision and boundary departure."""
    env_cfg.terminations.base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 10.0},
    )
    env_cfg.terminations.hip_contact = None
    env_cfg.terminations.terrain_out_of_bounds = DoneTerm(
        func=locomotion_mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 1.0},
        time_out=False,
    )


@configclass
class YuilDogRoughTrainEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Standalone Yuil Dog training task for rough-terrain locomotion."""

    def __post_init__(self) -> None:
        """Initialize standalone Yuil Dog scene, sensors, commands, and rewards."""
        super().__post_init__()
        self.scene.num_envs = 4096
        _apply_yuil_dog_robot(self)
        _apply_yuil_dog_terrain(self)
        _apply_yuil_dog_randomization(self)
        _apply_yuil_dog_velocity_command(self, debug_vis=False)
        _apply_yuil_dog_rewards(self)
        _apply_yuil_dog_terminations(self)


@configclass
class YuilDogRoughPlayEnvCfg(YuilDogRoughTrainEnvCfg):
    """Standalone Yuil Dog policy playback task."""

    def __post_init__(self) -> None:
        """Configure deterministic playback on a small terrain grid."""
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.debug_vis = True
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class YuilDogRoughEvalEnvCfg(YuilDogRoughTrainEnvCfg):
    """Standalone deterministic Yuil Dog evaluation task."""

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
class YuilDogRoughPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    """Standalone Yuil Dog PPO runner settings."""

    experiment_name = YUIL_DOG_ROUGH_EXPERIMENT_NAME
    max_iterations = 3000
    save_interval = 50
    clip_actions = 3.5

    def __post_init__(self) -> None:
        """Apply Yuil Dog experiment name and iteration budget."""
        super().__post_init__()
        self.experiment_name = YUIL_DOG_ROUGH_EXPERIMENT_NAME
        self.max_iterations = 3000
        self.save_interval = 50
        self.clip_actions = 3.5
