# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assemble the self-contained Yuil Dog MoE-CTS task."""

import copy

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import patterns
from isaaclab.utils.configclass import configclass

from ...env_cfg import Go2EnvCfg
from ...mdp.actions import JointPositionActionHistory
from ...mdp.commands_cts import Go2RLGymCommandCfg
from ...mdp.terrains import TERRAIN_CFG
from ...robots.yuil_dog.asset_cfg import YUIL_DOG_CFG
from ...robots.yuil_dog.joints_cfg import YUIL_DOG_BASE_PRIM_PATH, YUIL_DOG_JOINT_NAMES, YUIL_DOG_JOINT_POSITION_CLIP
from .curriculum_cfg import configure_curriculum
from .events_cfg import configure_events
from .observations_cfg import ObservationsCTSCfg
from .rewards_cfg import RewardsCTSCfg

_JOINTS = SceneEntityCfg("robot", joint_names=list(YUIL_DOG_JOINT_NAMES), preserve_order=True)


@configclass
class YuilDogMoECTSEnvCfg(Go2EnvCfg):
    """Reference rough terrain and CTS observations with validated Yuil hardware settings."""

    observations: ObservationsCTSCfg = ObservationsCTSCfg()
    rewards: RewardsCTSCfg = RewardsCTSCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 25.0
        self.commands.base_velocity = Go2RLGymCommandCfg()
        configure_curriculum(self)
        self.scene.robot = YUIL_DOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False
        self.scene.terrain.terrain_generator = copy.deepcopy(TERRAIN_CFG)
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.seed = 42
        self.scene.terrain.max_init_terrain_level = 5
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            class_type=JointPositionActionHistory,
            asset_name="robot",
            joint_names=list(YUIL_DOG_JOINT_NAMES),
            preserve_order=True,
            scale=0.25,
            use_default_offset=True,
            clip=copy.deepcopy(YUIL_DOG_JOINT_POSITION_CLIP),
        )
        self.observations.policy.joint_pos.params["asset_cfg"] = copy.deepcopy(_JOINTS)
        self.observations.policy.joint_vel.params["asset_cfg"] = copy.deepcopy(_JOINTS)
        self.scene.height_scanner.prim_path = YUIL_DOG_BASE_PRIM_PATH
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0))
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.height_scanner_small = self.scene.height_scanner.copy()
        self.scene.height_scanner_small.pattern_cfg = patterns.GridPatternCfg(resolution=0.1, size=(0.4, 0.3))
        self.scene.contact_forces.prim_path = f"{YUIL_DOG_BASE_PRIM_PATH}/.*"
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.contact_forces.class_type = (
            "go2_moe_cts.sensors.hierarchical_contact_sensor:HierarchicalContactSensor"
        )
        configure_events(self)
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names="base_link")


@configclass
class YuilDogMoECTSPlayEnvCfg(YuilDogMoECTSEnvCfg):
    """Student playback on the full rough-terrain family without observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class YuilDogMoECTSEvalEnvCfg(YuilDogMoECTSPlayEnvCfg):
    """Seeded evaluation with fixed nominal dynamics and deterministic reset state."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        self.seed = 42
        self.scene.terrain.terrain_generator.curriculum = False
        self.curriculum.terrain_levels = None
        self.curriculum.base_linear_velocity = None
        self.curriculum.base_height = None
        self.commands.base_velocity.dynamic_resample_commands = False
        self.commands.base_velocity.command_range_curriculum = []
        self.commands.base_velocity.zero_command_curriculum = None
        self.commands.base_velocity.resampling_time = 1e9
        for name in (
            "physics_material",
            "add_base_mass",
            "base_com",
            "scale_limb_mass",
            "actuator_gains",
            "motor_zero_offset",
        ):
            setattr(self.events, name, None)
        self.events.reset_robot_joints.params.update(position_range=(1.0, 1.0), velocity_range=(0.0, 0.0))
        self.events.reset_base.params["pose_range"] = {key: (0.0, 0.0) for key in ("x", "y", "z", "yaw")}
        self.events.reset_base.params["velocity_range"] = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.commands.base_velocity.resampling_time_range = (1e9, 1e9)
