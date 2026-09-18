# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project-owned Reach environment and PPO configuration for the Yuil arm."""

from __future__ import annotations

import math
import sys

import gymnasium as gym

from isaaclab.sim import SimulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.manager_based.manipulation.deploy.reach.config.ur_10e.agents.rsl_rl_ppo_cfg import (
    URReachPPORunnerCfg,
)
from isaaclab_tasks.manager_based.manipulation.deploy.reach.reach_env_cfg import ReachEnvCfg
from isaaclab_tasks.utils import PresetCfg

import isaaclab_tasks.manager_based.manipulation.deploy.mdp as mdp

from .asset_cfg import BASE_LINK_NAME, END_EFFECTOR_LINK_NAME, JOINT_NAMES, YUIL_3KG_ROBOT_CFG

TRAIN_TASK_ID = "Isaac-Reach-Yuil-3kg-v0"
PLAY_TASK_ID = "Isaac-Reach-Yuil-3kg-Play-v0"
EXPERIMENT_NAME = "reach_yuil_3kg_gravity"


@configclass
class Yuil3kgPhysicsCfg(PresetCfg):
    """Physics backends used to compare the same Yuil USD model."""

    default: PhysxCfg = PhysxCfg(enable_external_forces_every_iteration=True)
    physx: PhysxCfg = PhysxCfg(enable_external_forces_every_iteration=True)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=64,
            nconmax=32,
            iterations=100,
            ls_iterations=50,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=True,
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=False,
    )


@configclass
class Yuil3kgReachEnvCfg(ReachEnvCfg):
    """Reach training configuration using the company-provided robot USD."""

    # Author gravity explicitly at scene level so PhysX and Newton/MJWarp use
    # the same acceleration.
    sim: SimulationCfg = SimulationCfg(gravity=(0.0, 0.0, -9.81), physics=Yuil3kgPhysicsCfg())

    def __post_init__(self) -> None:
        """Apply Yuil robot, control, workspace, and visualization settings."""
        super().__post_init__()

        self.scene.num_envs = 4096
        # Keep this project independent of remote Isaac content. This fixed-base
        # Reach baseline does not require contact geometry.
        self.scene.ground = None
        self.scene.table = None
        self.scene.robot = YUIL_3KG_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        joint_names = list(JOINT_NAMES)
        # Keep the gravity-on critical-damping gains fixed for the baseline.
        # Friction randomization is disabled until real joint-friction data is
        # available; otherwise it obscures gravity and drive validation.
        self.events.robot_joint_stiffness_and_damping = None
        self.events.joint_friction = None

        self.scene.ee_frame_wrt_base_frame = FrameTransformerCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{BASE_LINK_NAME}",
            source_frame_offset=OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{END_EFFECTOR_LINK_NAME}",
                    name="end_effector",
                )
            ],
        )
        tracked_frame = SceneEntityCfg("ee_frame_wrt_base_frame")
        self.rewards.end_effector_keypoint_tracking.params["asset_cfg"] = tracked_frame
        self.rewards.end_effector_keypoint_tracking_exp.params["asset_cfg"] = tracked_frame
        self.rewards.action_rate.weight = -0.02

        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=joint_names,
            scale=0.05,
            use_zero_offset=True,
        )

        self.commands.ee_pose.body_name = END_EFFECTOR_LINK_NAME
        self.commands.ee_pose.debug_vis = True
        self.commands.ee_pose.resampling_time_range = (4.0, 4.0)

        # Provisional workspace around the zero-joint endplate pose
        # (-0.0866, -0.3464, 0.3386) m. Validate against the real safe workspace.
        self.target_pos_centre = (0.0, -0.30, 0.35)
        self.target_pos_range = (0.25, 0.25, 0.20)
        for axis, centre, radius in zip(
            ("pos_x", "pos_y", "pos_z"),
            self.target_pos_centre,
            self.target_pos_range,
        ):
            setattr(self.commands.ee_pose.ranges, axis, (centre - radius, centre + radius))

        # Nominal orientation of j6_adapter_endplate at the zero-joint pose.
        self.target_rot_centre = (-3.0 * math.pi / 4.0, -math.pi / 2.0, -3.0 * math.pi / 4.0)
        self.target_rot_range = (math.pi / 6.0, math.pi / 6.0, math.pi / 3.0)
        for axis, centre, radius in zip(
            ("roll", "pitch", "yaw"),
            self.target_rot_centre,
            self.target_rot_range,
        ):
            setattr(self.commands.ee_pose.ranges, axis, (centre - radius, centre + radius))

        goal_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Yuil3kg/TargetFrame")
        goal_marker_cfg.markers["frame"].scale = (0.10, 0.10, 0.10)
        self.commands.ee_pose.goal_pose_visualizer_cfg = goal_marker_cfg

        current_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Yuil3kg/EndEffectorFrame")
        current_marker_cfg.markers["frame"].scale = (0.075, 0.075, 0.075)
        self.commands.ee_pose.current_pose_visualizer_cfg = current_marker_cfg


@configclass
class Yuil3kgReachEnvCfgPlay(Yuil3kgReachEnvCfg):
    """Evaluation configuration with fewer environments and clean observations."""

    def __post_init__(self) -> None:
        """Apply evaluation-specific settings."""
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.5
        self.observations.policy.enable_corruption = False


@configclass
class Yuil3kgReachPPORunnerCfg(URReachPPORunnerCfg):
    """Project PPO configuration."""

    num_steps_per_env = 64
    max_iterations = 1500
    save_interval = 50
    experiment_name = EXPERIMENT_NAME


def register_tasks() -> list[str]:
    """Register project Gym tasks and preserve Hydra arguments."""
    if TRAIN_TASK_ID not in gym.registry:
        gym.register(
            id=TRAIN_TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}:Yuil3kgReachEnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}:Yuil3kgReachPPORunnerCfg",
            },
        )
    if PLAY_TASK_ID not in gym.registry:
        gym.register(
            id=PLAY_TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}:Yuil3kgReachEnvCfgPlay",
                "rsl_rl_cfg_entry_point": f"{__name__}:Yuil3kgReachPPORunnerCfg",
            },
        )
    return sys.argv[1:]
