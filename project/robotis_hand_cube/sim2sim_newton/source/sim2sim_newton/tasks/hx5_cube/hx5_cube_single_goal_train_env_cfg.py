# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-goal Sim2Real training configuration for the Robotis HX5 cube task."""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .hx5_cube_env_cfg import DEX_CUBE_DENSITY, DEX_CUBE_SIDE_LENGTH, RobotisHandEventCfg
from .hx5_cube_robust_train_env_cfg import RobotisHandRobustTrainEnvCfg
from .randomization import RandomizeCubeSizeMassInertia


@configclass
class RobotisHandSingleGoalEventCfg(RobotisHandEventCfg):
    """Additional object and contact randomization used only by the A task."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
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
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
            "make_consistent": True,
        },
    )
    # Replace the inherited independent mass randomizer with one correlated
    # size/mass/inertia event at constant density.
    object_scale_mass = None
    object_size_mass_inertia = EventTerm(
        func=RandomizeCubeSizeMassInertia,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "side_length_range": (0.047, 0.049),
            "density": DEX_CUBE_DENSITY,
            "nominal_side_length": DEX_CUBE_SIDE_LENGTH,
        },
    )
    object_center_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_com,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "com_range": {
                "x": (-0.001, 0.001),
                "y": (-0.001, 0.001),
                "z": (-0.001, 0.001),
            },
        },
    )


@configclass
class RobotisHandSingleGoalTrainEnvCfg(RobotisHandRobustTrainEnvCfg):
    """Terminate after one stable goal while preserving the robust physics setup."""

    events: RobotisHandSingleGoalEventCfg = RobotisHandSingleGoalEventCfg()

    terminate_on_success: bool = True
    reset_target_on_success: bool = False
    success_requires_not_fallen: bool = True

    # A-task reward: reward improvement instead of time spent near the goal.
    reward_mode: str = "single_goal_progress"
    rotation_progress_scale: float = 20.0
    orientation_error_penalty_scale: float = -0.5
    distance_safe_radius: float = 0.06
    distance_penalty_scale: float = -100.0
    action_rate_penalty_scale: float = -0.01
    # Preserve raw policy outputs for the bound penalty while clipping only the
    # command sent through the simulated actuator path.
    action_clip_value: float = 1.0
    action_saturation_threshold: float = 0.9
    action_saturation_penalty_scale: float = -0.01
    joint_velocity_penalty_scale: float = -0.01
    # Allow a wider orientation target and hold it for 0.5 s at the 30 Hz policy rate.
    success_tolerance: float = 0.4
    success_hold_steps: int = 15
    # Retained for configuration compatibility; angular speed is no longer a
    # success condition. Success only requires orientation tolerance + hold.
    success_max_object_angvel: float = 2.0
    hold_reward_scale: float = 1.0
    time_penalty: float = -0.01
    timeout_penalty: float = -50.0

    # Extended Sim2Real actuator randomization. Each value is sampled per
    # environment at reset and held for the complete episode.
    enable_extended_sim2real_randomization: bool = True
    joint_zero_offset_range: tuple[float, float] = (-0.01, 0.01)
    joint_effort_limit_scale_range: tuple[float, float] = (0.9, 1.1)
    joint_velocity_limit_scale_range: tuple[float, float] = (0.9, 1.1)

    # Total command-path latency at the 30 Hz policy rate. A one-step delay is
    # 33.3 ms. Jitter is represented by occasionally holding the last applied
    # command for one additional policy step.
    action_delay_step_range: tuple[int, int] = (0, 1)
    action_hold_probability: float = 0.05

    # Keep commanded and reset joint positions 3% away from each physical end
    # stop. The policy still uses its full normalized action range [-1, 1].
    joint_limit_safety_margin_fraction: float = 0.03
    # Match the V8 controller: apply 80% of the newly generated target and
    # retain 20% of the previous target once per policy step.
    act_moving_average: float = 0.8
    # Limit each joint-position target change to 0.08 rad per 30 Hz policy step.
    joint_target_velocity_limit: float = 2.4

    def __post_init__(self) -> None:
        """Configure robust actuators and additional A-task GPU contact capacity."""
        super().__post_init__()
        # The runtime derives in_hand_pos from the object's default position and
        # subtracts 0.04 m on z, yielding (0.0, -0.44, 0.56) as in V8.
        for object_cfg in (self.object_cfg.physx, self.object_cfg.default):
            object_cfg.init_state.pos = (0.0, -0.44, 0.60)
        for robot_cfg in (self.robot_cfg.physx, self.robot_cfg.default):
            robot_cfg.actuators["hand"].velocity_limit_sim = 4.8
        for physics_cfg in (self.sim.physics.physx, self.sim.physics.default):
            physics_cfg.gpu_max_rigid_patch_count = 2**18


@configclass
class RobotisHandSingleGoalNominalTrainEnvCfg(RobotisHandSingleGoalTrainEnvCfg):
    """Initial curriculum stage without Sim2Real randomization or latency.

    The task reset distribution, target orientations, reward, observations, and
    action spaces remain identical to :class:`RobotisHandSingleGoalTrainEnvCfg`
    so its checkpoints can be resumed directly in the randomized stage.
    """

    events = None
    observation_noise_model = None
    action_noise_model = None
    enable_cube_pose_obs_noise: bool = False
    enable_cube_pose_sample_hold: bool = False
    enable_extended_sim2real_randomization: bool = False
    action_delay_step_range: tuple[int, int] = (0, 0)
    action_hold_probability: float = 0.0


@configclass
class RobotisHandSingleGoalPlayEnvCfg(RobotisHandSingleGoalTrainEnvCfg):
    """Single-goal playback configuration with periodic terminal statistics."""

    play_stats_every_steps: int = 30


@configclass
class RobotisHandSingleGoalNominalPlayEnvCfg(RobotisHandSingleGoalPlayEnvCfg):
    """Single-goal PhysX evaluation without Sim2Real domain randomization.

    Initial hand/object state and goal sampling remain enabled because they are
    part of the task distribution rather than physical domain randomization.
    The actuator, cube, and contact properties therefore stay at their nominal
    values for a like-for-like comparison with Newton nominal playback.
    """

    events = None
    observation_noise_model = None
    action_noise_model = None
    enable_cube_pose_obs_noise: bool = False
    enable_cube_pose_sample_hold: bool = False
    enable_extended_sim2real_randomization: bool = False
    action_delay_step_range: tuple[int, int] = (0, 0)
    action_hold_probability: float = 0.0
