# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robust PhysX training environment for the Robotis HX5 cube task with domain randomization."""

from __future__ import annotations

import math

from isaaclab.utils.configclass import configclass

from .hx5_cube_env_cfg import SIM2REAL_ACTUATOR_ARMATURE, RobotisHandEnvCfg, RobotisHandEventCfg


@configclass
class RobotisHandRobustTrainEnvCfg(RobotisHandEnvCfg):
    """Robust PhysX training environment with domain randomization.

    Extends :class:`RobotisHandEnvCfg` with:

    * Physics randomization (friction, mass, motor gains) via
      :class:`RobotisHandEventCfg`.
    * Deterministic joint-position targets without action noise.
    * Cube-pose-specific observation noise to simulate 6D pose estimation error.
    """

    # Domain randomization events
    events: RobotisHandEventCfg = RobotisHandEventCfg()

    # A goal must remain inside the orientation tolerance for 10 consecutive
    # 30 Hz policy steps (1/3 second). Dropping the cube incurs an explicit
    # penalty in addition to terminating the episode.
    success_hold_steps: int = 10
    fall_penalty: float = -50.0

    # Keep policy outputs deterministic when converting them to absolute joint
    # position targets. Motor dynamics are still covered by gain and armature
    # randomization below.
    action_noise_model = None

    # Do not corrupt all 149 policy observations with one unit-agnostic noise
    # scale. Cube pose noise is applied before constructing cube-derived terms.
    observation_noise_model = None
    enable_cube_pose_obs_noise: bool = True
    cube_position_obs_noise_std: tuple[float, float, float] = (0.005, 0.005, 0.010)
    cube_position_obs_bias_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-0.005, 0.005),
        (-0.005, 0.005),
        (-0.010, 0.010),
    )
    cube_orientation_obs_noise_std: float = math.radians(3.0)
    cube_orientation_obs_bias_max: float = math.radians(2.0)
    cube_pose_obs_noise_clip_sigma: float = 3.0
    # A 30 Hz policy receives a new 6D pose every two or three steps,
    # matching a 10-15 Hz pose estimator. Intermediate observations are held.
    enable_cube_pose_sample_hold: bool = True
    cube_pose_update_interval_steps_range: tuple[int, int] = (2, 3)
    enable_cube_velocity_from_pose: bool = True
    cube_velocity_obs_low_pass_alpha: float = 0.35
    cube_linear_velocity_obs_max: float = 3.0
    cube_angular_velocity_obs_max: float = 30.0

    def __post_init__(self) -> None:
        """Set the provisional physical actuator inertia for Sim2Real training."""
        interval_low, interval_high = self.cube_pose_update_interval_steps_range
        if interval_low < 1 or interval_high < interval_low:
            raise ValueError(
                "cube_pose_update_interval_steps_range must contain positive, monotonically increasing values."
            )
        # PresetCfg stores independent ``physx`` and ``default`` instances.
        # Update both so direct construction and Hydra preset resolution agree.
        for robot_cfg in (self.robot_cfg.physx, self.robot_cfg.default):
            robot_cfg.actuators["hand"].armature = SIM2REAL_ACTUATOR_ARMATURE
