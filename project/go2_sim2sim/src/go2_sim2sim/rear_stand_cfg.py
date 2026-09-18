# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 rear-foot balancing task configuration."""

from __future__ import annotations

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.sensors import MultiMeshRayCasterCfg, patterns
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)

from . import rear_stand_mdp
from .asset_cfg import GO2_CFG, GO2_JOINT_NAMES
from .sensor_patterns import dual_sector_lidar_pattern

REAR_STAND_TRAIN_TASK_ID = "Isaac-Rear-Stand-Go2-Sim2Sim-v0"
REAR_STAND_PLAY_TASK_ID = "Isaac-Rear-Stand-Go2-Sim2Sim-Play-v0"
REAR_STAND_EXPERIMENT_NAME = "go2_rear_bent_balance"

GO2_FRONT_FOOT_NAMES = ("FL_foot", "FR_foot")
GO2_REAR_FOOT_NAMES = ("RL_foot", "RR_foot")


def _apply_rear_stand_robot(env_cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Apply the project asset and deterministic policy joint ordering."""
    env_cfg.scene.robot = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.scene.robot.actuators["base_legs"].armature = 0.02
    env_cfg.scene.terrain.class_type = "go2_sim2sim.terrain:LocalPlaneTerrainImporter"
    joint_names = list(GO2_JOINT_NAMES)
    env_cfg.actions.joint_pos.joint_names = joint_names
    env_cfg.actions.joint_pos.preserve_order = True
    joint_cfg = SceneEntityCfg("robot", joint_names=joint_names, preserve_order=True)
    env_cfg.observations.policy.joint_pos.params["asset_cfg"] = joint_cfg
    env_cfg.observations.policy.joint_vel.params["asset_cfg"] = joint_cfg


from typing import Literal

from .sensor_patterns import (
    create_downward_lidar_cfg,
    create_lidar_cfg,
    dual_sector_lidar_pattern,
)


def _apply_rear_stand_sensors(
    env_cfg: UnitreeGo2FlatEnvCfg,
    debug_vis: bool = False,
    orientation: Literal["forward", "downward"] = "downward",
) -> None:
    """Attach 3D LiDAR to Go2 robot.

    Options:
      - 'downward' : Chin-mounted downward 3D LiDAR for ground terrain & step profiling (Default for rear stand).
      - 'forward'  : Front-facing horizontal 3D LiDAR.
    """
    if debug_vis:
        env_cfg.scene.lidar = create_lidar_cfg(
            orientation=orientation,
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=debug_vis,
        )
    else:
        env_cfg.scene.lidar = None


def _disable_velocity_objective(env_cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Keep a zero command for manager compatibility without rewarding locomotion."""
    command = env_cfg.commands.base_velocity
    command.heading_command = False
    command.rel_heading_envs = 0.0
    command.rel_standing_envs = 1.0
    command.debug_vis = False
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.ranges.heading = None
    env_cfg.observations.policy.velocity_commands = None


def _apply_rear_stand_rewards(env_cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Replace locomotion rewards with rear-foot balance rewards."""
    for term_name in vars(env_cfg.rewards):
        if not term_name.startswith("_"):
            setattr(env_cfg.rewards, term_name, None)

    rear_feet = SceneEntityCfg("contact_forces", body_names=list(GO2_REAR_FOOT_NAMES))
    front_feet = SceneEntityCfg("contact_forces", body_names=list(GO2_FRONT_FOOT_NAMES))
    non_foot_bodies = SceneEntityCfg(
        "contact_forces", body_names=["base", ".*_hip", ".*_thigh", ".*_calf"]
    )
    front_feet_asset = SceneEntityCfg("robot", body_names=list(GO2_FRONT_FOOT_NAMES))
    front_joints = SceneEntityCfg(
        "robot",
        joint_names=["F[L,R]_hip_joint", "F[L,R]_thigh_joint", "F[L,R]_calf_joint"],
    )
    hip_joints = SceneEntityCfg("robot", joint_names=[".*_hip_joint"])
    rear_knee_joints = SceneEntityCfg("robot", joint_names=["R[L,R]_calf_joint"])
    joints = SceneEntityCfg("robot", joint_names=list(GO2_JOINT_NAMES), preserve_order=True)

    # Dense shaping makes the objective learnable from the nominal four-foot pose.
    env_cfg.rewards.rear_support = RewTerm(
        func=rear_stand_mdp.desired_contact_any,
        weight=2.0,
        params={"sensor_cfg": rear_feet, "threshold": 5.0},
    )
    env_cfg.rewards.front_feet_air = RewTerm(
        func=rear_stand_mdp.desired_air_fraction,
        weight=2.0,
        params={"sensor_cfg": front_feet, "threshold": 5.0},
    )
    # This is the dominant task term: partial contact patterns receive no bonus.
    env_cfg.rewards.rear_stand_pattern = RewTerm(
        func=rear_stand_mdp.rear_stand_contact_pattern,
        weight=8.0,
        params={
            "rear_sensor_cfg": rear_feet,
            "front_sensor_cfg": front_feet,
            "forbidden_sensor_cfg": non_foot_bodies,
            "threshold": 5.0,
            "min_rear_contacts": 1,
        },
    )
    # A roughly 70-degree nose-up pose distinguishes standing from merely lifting the front paws.
    env_cfg.rewards.base_orientation = RewTerm(
        func=rear_stand_mdp.base_orientation_exp,
        weight=4.0,
        params={"target_gravity": (-0.94, 0.0, -0.342), "std": 0.45},
    )
    env_cfg.rewards.base_height = RewTerm(
        func=rear_stand_mdp.base_height_exp,
        weight=3.0,
        params={"target_height": 0.50, "std": 0.12},
    )
    # Keep the abduction joints near their nominal pose so the legs do not
    # spread sideways. This remains soft so a recovery step is still possible.
    env_cfg.rewards.hip_posture = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": hip_joints},
    )
    # Keep some bend in both rear knees without prescribing one exact angle.
    env_cfg.rewards.rear_knee_posture = RewTerm(
        func=rear_stand_mdp.joint_pos_outside_range_l2,
        weight=-1.5,
        params={"asset_cfg": rear_knee_joints, "lower": -1.8, "upper": -1.0},
    )

    standing_contact_params = {
        "rear_sensor_cfg": rear_feet,
        "front_sensor_cfg": front_feet,
        "forbidden_sensor_cfg": non_foot_bodies,
        "threshold": 5.0,
    }
    env_cfg.rewards.front_feet_motion = RewTerm(
        func=rear_stand_mdp.body_lin_vel_l2_when_rear_standing,
        weight=-0.5,
        params={"asset_cfg": front_feet_asset, **standing_contact_params},
    )
    env_cfg.rewards.front_joint_velocity = RewTerm(
        func=rear_stand_mdp.joint_vel_l2_when_rear_standing,
        weight=-0.02,
        params={"asset_cfg": front_joints, **standing_contact_params},
    )

    # Regularizers suppress violent solutions without discouraging horizontal motion.
    env_cfg.rewards.base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    env_cfg.rewards.action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    env_cfg.rewards.joint_torques = RewTerm(
        func=mdp.joint_torques_l2, weight=-2.0e-5, params={"asset_cfg": joints}
    )
    env_cfg.rewards.joint_acceleration = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": joints}
    )
    env_cfg.rewards.non_foot_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={"sensor_cfg": non_foot_bodies, "threshold": 1.0},
    )


@configclass
class Go2RearStandTrainEnvCfg(UnitreeGo2FlatEnvCfg):
    """Train Go2 to balance on both rear feet on flat ground."""

    def __post_init__(self) -> None:
        """Apply rear-standing training settings."""
        super().__post_init__()
        self.scene.num_envs = 4096
        self.episode_length_s = 12.0
        _apply_rear_stand_robot(self)
        _apply_rear_stand_sensors(self, debug_vis=False)
        _disable_velocity_objective(self)
        _apply_rear_stand_rewards(self)

        # Begin from the familiar four-foot pose and learn the transition through dense shaping.
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (-0.1, 0.1),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.1, 0.1),
        }
        self.events.push_robot = None
        self.terminations.low_base = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.22})
        self.terminations.non_foot_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf"]
                ),
                "threshold": 5.0,
            },
        )

        self.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=list(GO2_JOINT_NAMES), preserve_order=True
                ),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
            },
        )


@configclass
class Go2RearStandPlayEnvCfg(Go2RearStandTrainEnvCfg):
    """Visualize a learned rear-foot balancing policy."""

    def __post_init__(self) -> None:
        """Disable training perturbations for visualization."""
        super().__post_init__()
        self.scene.num_envs = 16
        _apply_rear_stand_sensors(self, debug_vis=False)
        self.observations.policy.enable_corruption = False
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.actuator_gains = None


@configclass
class Go2RearStandPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    """PPO configuration isolated from locomotion checkpoints."""

    experiment_name = REAR_STAND_EXPERIMENT_NAME
    max_iterations = 1500
    save_interval = 50

    def __post_init__(self) -> None:
        """Restore rear-standing values after upstream initialization."""
        super().__post_init__()
        self.experiment_name = REAR_STAND_EXPERIMENT_NAME
        self.max_iterations = 1500
        self.save_interval = 50
