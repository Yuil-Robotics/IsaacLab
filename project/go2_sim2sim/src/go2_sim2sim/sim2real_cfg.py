# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration types for sim-to-real action and actuator models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.actuators import DCMotorCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .sim2real import Sim2RealDCMotor, Sim2RealJointPositionAction


@configclass
class Sim2RealJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for episode-randomized policy action delay and calibration bias."""

    class_type: type[Sim2RealJointPositionAction] | str = (
        "go2_sim2sim.sim2real:Sim2RealJointPositionAction"
    )

    min_delay: int = 0
    """Minimum policy delay [policy steps]."""

    max_delay: int = 2
    """Maximum policy delay [policy steps]."""

    joint_bias_range: tuple[float, float] | None = (-0.03, 0.03)
    """Joint position calibration bias range [rad]. None disables bias randomization."""


@configclass
class Sim2RealDCMotorCfg(DCMotorCfg):
    """Configuration for delayed and torque-randomized DC motors."""

    class_type: type[Sim2RealDCMotor] | str = "go2_sim2sim.sim2real:Sim2RealDCMotor"

    min_delay: int = 0
    """Minimum motor command delay [physics steps]."""

    max_delay: int = 3
    """Maximum motor command delay [physics steps]."""

    effort_scale_range: tuple[float, float] = (0.85, 1.15)
    """Available torque scale range relative to nominal torque."""

    motor_strength_range: tuple[float, float] | None = (0.9, 1.1)
    """Motor output torque strength multiplier range. None disables strength scaling."""
