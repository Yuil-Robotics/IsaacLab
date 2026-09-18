# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project-owned Go2 rough-terrain locomotion task and PPO configurations."""

from __future__ import annotations

import copy
import math
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2RoughPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
    UnitreeGo2RoughEnvCfg_PLAY,
)

from . import rough_mdp
from .asset_cfg import (
    get_active_foot_names,
    get_active_joint_names,
    get_active_robot_cfg,
    get_active_trot_foot_pairs,
)
from .sensor_patterns import (
    create_lidar_cfg,
)
from .sim2real_cfg import Sim2RealDCMotorCfg, Sim2RealJointPositionActionCfg
from .visualization import CURRENT_VELOCITY_MARKER_CFG, GOAL_VELOCITY_MARKER_CFG

ROUGH_TRAIN_TASK_ID = "Isaac-Velocity-Rough-Go2-Sim2Sim-v0"
ROUGH_PLAY_TASK_ID = "Isaac-Velocity-Rough-Go2-Sim2Sim-Play-v0"
ROUGH_EVAL_TASK_ID = "Isaac-Velocity-Rough-Go2-Sim2Sim-Eval-v0"
ROUGH_EXPERIMENT_NAME = "go2_rough_spot_rewards_sim2sim_dr"

# Custom rough terrain configuration scaled for quadruped agility & robustness
CUSTOM_ROUGH_TERRAINS_CFG = copy.deepcopy(ROUGH_TERRAINS_CFG)
CUSTOM_ROUGH_TERRAINS_CFG.curriculum = False
CUSTOM_ROUGH_TERRAINS_CFG.sub_terrains.pop("pyramid_stairs", None)
CUSTOM_ROUGH_TERRAINS_CFG.sub_terrains.pop("pyramid_stairs_inv", None)
CUSTOM_ROUGH_TERRAINS_CFG.sub_terrains["boxes"].proportion = 0.35
CUSTOM_ROUGH_TERRAINS_CFG.sub_terrains["boxes"].grid_height_range = (0.025, 0.10)
CUSTOM_ROUGH_TERRAINS_CFG.sub_terrains["boxes"].grid_width = 0.45
CUSTOM_ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].proportion = 0.35
CUSTOM_ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
CUSTOM_ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].noise_step = 0.01
CUSTOM_ROUGH_TERRAINS_CFG.sub_terrains["hf_pyramid_slope"].proportion = 0.15
CUSTOM_ROUGH_TERRAINS_CFG.sub_terrains["hf_pyramid_slope"].slope_range = (0.0, 0.35)
CUSTOM_ROUGH_TERRAINS_CFG.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.15
CUSTOM_ROUGH_TERRAINS_CFG.sub_terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.35)


def _apply_rough_robot(env_cfg: UnitreeGo2RoughEnvCfg) -> None:
    """Apply the active robot asset and deterministic joint order for rough terrain."""
    robot_cfg = get_active_robot_cfg()
    env_cfg.scene.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Resolve leg-to-leg contacts physically so the contact sensor can also
    # penalize policies that drive non-adjacent links into one another.
    if env_cfg.scene.robot.spawn.articulation_props is None:
        env_cfg.scene.robot.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True
        )
    else:
        env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True
    if "base_legs" in env_cfg.scene.robot.actuators:
        env_cfg.scene.robot.actuators["base_legs"].armature = 0.02

    joint_names = list(get_active_joint_names())
    env_cfg.actions.joint_pos.joint_names = joint_names
    env_cfg.actions.joint_pos.preserve_order = True
    env_cfg.actions.joint_pos.scale = 0.25

    joint_asset_cfg = SceneEntityCfg("robot", joint_names=joint_names, preserve_order=True)
    env_cfg.observations.policy.joint_pos.params["asset_cfg"] = joint_asset_cfg
    env_cfg.observations.policy.joint_vel.params["asset_cfg"] = joint_asset_cfg


GREEN_RAY_CASTER_MARKER_CFG = RAY_CASTER_MARKER_CFG.copy()
GREEN_RAY_CASTER_MARKER_CFG.markers["hit"].visual_material.diffuse_color = (0.0, 1.0, 0.0)


def _apply_rough_sensors(
    env_cfg: UnitreeGo2RoughEnvCfg,
    debug_vis: bool = False,
    orientation: Literal["forward", "downward"] = "forward",
) -> None:
    """Configure sensors for blind proprioceptive locomotion and optional LiDAR.

    Options:
      - 'forward'  : Front-facing horizontal 3D LiDAR for navigation & obstacle mapping (Default).
      - 'downward' : Chin-mounted downward 3D LiDAR for ground terrain & step profiling.

    See DOWNWARD_LIDAR_MEMO and FORWARD_LIDAR_MEMO in sensor_patterns.py for full specifications.
    """
    # Blind locomotion contract (48 dims) for zero-shot real robot deployment
    env_cfg.scene.height_scanner = None
    env_cfg.observations.policy.height_scan = None

    if debug_vis:
        env_cfg.scene.lidar = create_lidar_cfg(
            orientation=orientation,
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=debug_vis,
        )
    else:
        env_cfg.scene.lidar = None


def _apply_rough_terrain(env_cfg: UnitreeGo2RoughEnvCfg) -> None:
    """Apply scaled rough terrain generator without curriculum, training across all terrains simultaneously."""
    env_cfg.scene.terrain.terrain_generator = copy.deepcopy(CUSTOM_ROUGH_TERRAINS_CFG)
    env_cfg.scene.terrain.terrain_generator.curriculum = False
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.20, 0.20, 0.20),
        roughness=0.8,
    )
    # Disable level-based curriculum so all terrains and difficulties are trained simultaneously
    env_cfg.curriculum.terrain_levels = None


def _apply_rough_velocity_command(env_cfg: UnitreeGo2RoughEnvCfg, debug_vis: bool) -> None:
    """Configure 10% standing, three 20% pure modes, and 30% general commands."""
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.class_type = "go2_sim2sim.velocity_command:VisibleVelocityCommand"
    command_cfg.debug_vis = debug_vis
    # Exactly 1/10 (10%) of environments receive [0.0, 0.0, 0.0] standing command
    command_cfg.rel_standing_envs = 0.10
    # 20% receive pure [vx, 0.0, 0.0] forward/backward commands.
    command_cfg.rel_straight_envs = 0.20
    # 20% receive pure [0.0, 0.0, wz] turning commands.
    command_cfg.rel_turning_envs = 0.20
    # 20% receive pure [0.0, vy, 0.0] lateral commands.
    command_cfg.rel_lateral_envs = 0.20
    command_cfg.heading_command = True
    command_cfg.heading_control_stiffness = 0.5
    command_cfg.resampling_time_range = (10.0, 10.0)
    command_cfg.ranges.lin_vel_x = (-1.0, 1.0)
    command_cfg.ranges.lin_vel_y = (-0.5, 0.5)
    command_cfg.ranges.ang_vel_z = (-1.0, 1.0)
    command_cfg.ranges.heading = (-math.pi, math.pi)
    command_cfg.goal_vel_visualizer_cfg = GOAL_VELOCITY_MARKER_CFG.copy()
    command_cfg.current_vel_visualizer_cfg = CURRENT_VELOCITY_MARKER_CFG.copy()


def _apply_rough_randomization(env_cfg: UnitreeGo2RoughEnvCfg) -> None:
    """Apply domain randomization tailored for rough terrain contact dynamics."""
    joint_names = list(get_active_joint_names())
    base_actuator_cfg = env_cfg.scene.robot.actuators["base_legs"]
    env_cfg.scene.robot.actuators["base_legs"] = Sim2RealDCMotorCfg(
        joint_names_expr=base_actuator_cfg.joint_names_expr,
        effort_limit=base_actuator_cfg.effort_limit,
        saturation_effort=base_actuator_cfg.saturation_effort,
        velocity_limit=base_actuator_cfg.velocity_limit,
        effort_limit_sim=base_actuator_cfg.effort_limit_sim,
        velocity_limit_sim=base_actuator_cfg.velocity_limit_sim,
        stiffness=base_actuator_cfg.stiffness,
        damping=base_actuator_cfg.damping,
        armature=base_actuator_cfg.armature,
        friction=base_actuator_cfg.friction,
        dynamic_friction=base_actuator_cfg.dynamic_friction,
        viscous_friction=base_actuator_cfg.viscous_friction,
        min_delay=0,
        max_delay=3,
        effort_scale_range=(0.8, 1.0),
    )
    env_cfg.actions.joint_pos = Sim2RealJointPositionActionCfg(
        asset_name="robot",
        joint_names=joint_names,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
        min_delay=0,
        max_delay=1,
    )
    env_cfg.events.physics_material.params.update(
        {
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.2, 0.8),
            "restitution_range": (0.0, 0.05),
            "make_consistent": True,
        }
    )
    env_cfg.events.actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=joint_names, preserve_order=True),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    env_cfg.events.joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=joint_names, preserve_order=True),
            "friction_distribution_params": (0.0, 0.1),
            "operation": "abs",
            "distribution": "uniform",
        },
    )


def _apply_rough_rewards(env_cfg: UnitreeGo2RoughEnvCfg) -> None:
    """Apply terrain-relative rewards with 10% zero-command balance shaping."""
    # Remove upstream baseline rewards to prevent objective conflict
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
    env_cfg.rewards.base_height_l2 = None

    foot_names = list(get_active_foot_names())
    joint_names = list(get_active_joint_names())
    trot_pairs = get_active_trot_foot_pairs()

    # -- Primary velocity tracking objectives. Squared kernels keep poor tracking
    # clearly separated from merely surviving on difficult terrain.
    env_cfg.rewards.base_linear_velocity = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=4.0,
        params={
            "std": 0.5,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    env_cfg.rewards.base_angular_velocity = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={
            "std": 0.5,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # -- Stepping and clearance. These terms blend smoothly into stance support
    # at low commands and never use absolute world height as a terrain reference.
    env_cfg.rewards.air_time = None
    env_cfg.rewards.air_time_variance = None
    env_cfg.rewards.foot_slip = RewTerm(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=foot_names),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=foot_names),
            "threshold": 5.0,
        },
    )
    env_cfg.rewards.foot_clearance = None

    # -- Gait coordination. A trot remains a preference at meaningful planar
    # speed, while low-speed turns and difficult footholds may use other patterns.
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
            "synced_feet_pair_names": trot_pairs,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    # -- Suppress four-foot hopping directly. Very brief all-foot contact loss is
    # tolerated for sensor transitions and obstacle edges, then penalized smoothly.
    env_cfg.rewards.flight_phase = RewTerm(
        func=rough_mdp.flight_phase_penalty,
        weight=-3.0,
        params={
            "grace_time": 0.04,
            "full_time": 0.10,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=foot_names),
        },
    )

    # -- Base stability. Squared penalties are gentle around normal slope and
    # step adaptation but still grow rapidly for unstable motion.
    env_cfg.rewards.base_motion = None
    env_cfg.rewards.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    env_cfg.rewards.ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    env_cfg.rewards.base_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # -- Posture and joint safety. Avoid hip-specific duplication so the legs can
    # widen or shift on slopes, while directly protecting soft joint limits.
    env_cfg.rewards.joint_pos = RewTerm(
        func=spot_mdp.joint_position_penalty,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=joint_names, preserve_order=True),
            "stand_still_scale": 3.0,
            "velocity_threshold": 0.5,
        },
    )
    env_cfg.rewards.dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names, preserve_order=True)},
    )
    env_cfg.rewards.hip_joint_pos = None
    env_cfg.rewards.hip_straight_pos = None

    # -- Action smoothness and motor protection
    env_cfg.rewards.action_smoothness = RewTerm(func=spot_mdp.action_smoothness_penalty, weight=-1.0)
    env_cfg.rewards.joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names, preserve_order=True)},
    )
    env_cfg.rewards.joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names, preserve_order=True)},
    )
    env_cfg.rewards.joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names, preserve_order=True)},
    )
    env_cfg.rewards.hip_joint_vel = RewTerm(
        func=rough_mdp.straight_motion_hip_velocity_penalty,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
            "vy_threshold": 0.05,
            "wz_threshold": 0.05,
        },
    )
    env_cfg.rewards.hip_joint_torques = None

    # -- Collision avoidance on obstacles and stairs
    env_cfg.rewards.undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_hip", ".*_thigh", ".*_calf"],
            ),
            "threshold": 5.0,
        },
    )


def _apply_rough_terminations(env_cfg: UnitreeGo2RoughEnvCfg) -> None:
    """Terminate on severe base contact or approaching the terrain edge."""
    env_cfg.terminations.base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    env_cfg.terminations.terrain_out_of_bounds = DoneTerm(
        func=locomotion_mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 1.0},
        time_out=False,
    )


# -----------------------------------------------------------------------------
# Environment Configurations
# -----------------------------------------------------------------------------


@configclass
class Go2RoughTrainEnvCfg(UnitreeGo2RoughEnvCfg):
    """Go2 rough-terrain velocity tracking task for PhysX training."""

    def __post_init__(self) -> None:
        """Apply project rough training defaults."""
        super().__post_init__()
        self.scene.num_envs = 4096
        _apply_rough_robot(self)
        _apply_rough_sensors(self, debug_vis=False)
        _apply_rough_terrain(self)
        _apply_rough_randomization(self)
        _apply_rough_velocity_command(self, debug_vis=False)
        _apply_rough_rewards(self)
        _apply_rough_terminations(self)


@configclass
class Go2RoughPlayEnvCfg(UnitreeGo2RoughEnvCfg_PLAY):
    """Go2 rough-terrain policy visualization on diverse sub-terrains."""

    def __post_init__(self) -> None:
        """Apply project rough play defaults."""
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        _apply_rough_robot(self)
        _apply_rough_sensors(self, debug_vis=False)
        _apply_rough_terrain(self)
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
        _apply_rough_velocity_command(self, debug_vis=True)
        _apply_rough_rewards(self)
        _apply_rough_terminations(self)

        # Disable stochastic disturbances for deterministic play
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.actuator_gains = None


@configclass
class Go2RoughEvalEnvCfg(UnitreeGo2RoughEnvCfg_PLAY):
    """Deterministic evaluation configuration for rough terrain locomotion."""

    def __post_init__(self) -> None:
        """Remove stochastic perturbations for benchmark evaluation."""
        super().__post_init__()
        self.scene.num_envs = 256
        self.commands.base_velocity.debug_vis = False
        _apply_rough_robot(self)
        _apply_rough_sensors(self, debug_vis=False)
        _apply_rough_terrain(self)
        _apply_rough_rewards(self)
        _apply_rough_terminations(self)

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
        self.events.actuator_gains = None
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
class Go2RoughPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    """Project PPO runner configuration for rough-terrain learning."""

    experiment_name = ROUGH_EXPERIMENT_NAME
    max_iterations = 3000
    save_interval = 50
    clip_actions = 3.5

    def __post_init__(self) -> None:
        """Apply project experiment name and iteration budget."""
        super().__post_init__()
        self.experiment_name = ROUGH_EXPERIMENT_NAME
        self.max_iterations = 3000
        self.save_interval = 50
        self.clip_actions = 3.5
