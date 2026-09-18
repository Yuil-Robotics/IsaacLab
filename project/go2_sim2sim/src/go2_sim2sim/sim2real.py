# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-to-real action and actuator models for locomotion training."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.actuators import DCMotor
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import DelayBuffer
from isaaclab.utils.types import ArticulationActions

from .sim2real_cfg import Sim2RealDCMotorCfg, Sim2RealJointPositionActionCfg


class Sim2RealJointPositionAction(JointPositionAction):
    """Joint-position action with an episode-randomized policy delay and calibration bias."""

    cfg: Sim2RealJointPositionActionCfg

    def __init__(self, cfg: Sim2RealJointPositionActionCfg, env) -> None:
        """Initialize the delayed action term.

        Args:
            cfg: Delayed joint-position action configuration.
            env: Manager-based environment that owns the action term.
        """
        if cfg.min_delay < 0 or cfg.max_delay < cfg.min_delay:
            raise ValueError(
                "Policy delay bounds must satisfy 0 <= min_delay <= max_delay. "
                f"Received ({cfg.min_delay}, {cfg.max_delay})."
            )
        if cfg.joint_bias_range is not None:
            if not cfg.joint_bias_range[0] <= cfg.joint_bias_range[1]:
                raise ValueError(
                    "Joint bias range bounds must be ordered. "
                    f"Received {cfg.joint_bias_range}."
                )
        super().__init__(cfg, env)
        self._delay_buffer = DelayBuffer(cfg.max_delay, self.num_envs, device=self.device)
        self._joint_bias = torch.zeros((self.num_envs, self.action_dim), device=self.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Scale, bias, and delay policy actions before applying joint targets."""
        super().process_actions(actions)
        if self.cfg.joint_bias_range is not None:
            self._processed_actions = self._processed_actions + self._joint_bias
            if self.cfg.clip is not None:
                self._processed_actions = torch.clamp(
                    self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
                )
        self._processed_actions = self._delay_buffer.compute(self._processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resample policy delay and joint calibration bias for reset environments."""
        super().reset(env_ids)
        batch_ids, num_envs = _resolve_reset_ids(env_ids, self.num_envs)
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self.device,
        )
        self._delay_buffer.set_time_lag(time_lags, batch_ids)
        self._delay_buffer.reset(batch_ids)

        if self.cfg.joint_bias_range is not None:
            self._joint_bias[batch_ids] = torch.empty(
                (num_envs, self.action_dim), device=self.device
            ).uniform_(*self.cfg.joint_bias_range)
        else:
            self._joint_bias[batch_ids] = 0.0


class Sim2RealDCMotor(DCMotor):
    """DC motor with randomized command delay, available torque, and motor strength."""

    cfg: Sim2RealDCMotorCfg

    def __init__(self, cfg: Sim2RealDCMotorCfg, *args, **kwargs) -> None:
        """Initialize delay buffers and nominal torque limits.

        Args:
            cfg: Sim-to-real DC motor configuration.
            *args: Positional arguments forwarded to :class:`isaaclab.actuators.DCMotor`.
            **kwargs: Keyword arguments forwarded to :class:`isaaclab.actuators.DCMotor`.
        """
        if cfg.min_delay < 0 or cfg.max_delay < cfg.min_delay:
            raise ValueError(
                "Motor delay bounds must satisfy 0 <= min_delay <= max_delay. "
                f"Received ({cfg.min_delay}, {cfg.max_delay})."
            )
        if not 0.0 < cfg.effort_scale_range[0] <= cfg.effort_scale_range[1]:
            raise ValueError(
                "Motor effort scale bounds must be positive and ordered. "
                f"Received {cfg.effort_scale_range}."
            )
        if cfg.motor_strength_range is not None:
            if not 0.0 < cfg.motor_strength_range[0] <= cfg.motor_strength_range[1]:
                raise ValueError(
                    "Motor strength range bounds must be positive and ordered. "
                    f"Received {cfg.motor_strength_range}."
                )

        super().__init__(cfg, *args, **kwargs)
        self._position_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._velocity_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._effort_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)

        self._default_effort_limit = self.effort_limit.clone()
        self._default_saturation_effort = torch.full_like(self.effort_limit, float(cfg.saturation_effort))
        self._saturation_effort = self._default_saturation_effort.clone()
        self._motor_strength = torch.ones((self._num_envs, self.num_joints), device=self._device)
        self._update_velocity_at_effort_limit()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resample motor delay, torque capacity, and motor strength for reset environments."""
        super().reset(env_ids)
        batch_ids, num_envs = _resolve_reset_ids(env_ids, self._num_envs)
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        self._position_delay_buffer.set_time_lag(time_lags, batch_ids)
        self._velocity_delay_buffer.set_time_lag(time_lags, batch_ids)
        self._effort_delay_buffer.set_time_lag(time_lags, batch_ids)
        self._position_delay_buffer.reset(batch_ids)
        self._velocity_delay_buffer.reset(batch_ids)
        self._effort_delay_buffer.reset(batch_ids)

        effort_scales = torch.empty((num_envs, 1), device=self._device).uniform_(*self.cfg.effort_scale_range)
        self.effort_limit[batch_ids] = self._default_effort_limit[batch_ids] * effort_scales
        self._saturation_effort[batch_ids] = self._default_saturation_effort[batch_ids] * effort_scales
        self._update_velocity_at_effort_limit(batch_ids)

        if self.cfg.motor_strength_range is not None:
            self._motor_strength[batch_ids] = torch.empty(
                (num_envs, self.num_joints), device=self._device
            ).uniform_(*self.cfg.motor_strength_range)
        else:
            self._motor_strength[batch_ids] = 1.0

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        """Delay motor commands and scale output torque using the motor strength model."""
        control_action.joint_positions = self._position_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self._velocity_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self._effort_delay_buffer.compute(control_action.joint_efforts)
        action = super().compute(control_action, joint_pos, joint_vel)
        if action.joint_efforts is not None and self.cfg.motor_strength_range is not None:
            action.joint_efforts = action.joint_efforts * self._motor_strength
            self.applied_effort = action.joint_efforts
        return action

    def _update_velocity_at_effort_limit(self, env_ids: Sequence[int] | None = None) -> None:
        """Update torque-speed curve intersections for selected environments."""
        if env_ids is None:
            env_ids = slice(None)
        self._vel_at_effort_lim[env_ids] = self.velocity_limit[env_ids] * (
            1.0 + self.effort_limit[env_ids] / self._saturation_effort[env_ids]
        )


def _resolve_reset_ids(
    env_ids: Sequence[int] | None,
    num_envs: int,
) -> tuple[Sequence[int] | slice, int]:
    """Resolve reset indices and their batch size."""
    if env_ids is None or isinstance(env_ids, slice):
        return slice(None), num_envs
    return env_ids, len(env_ids)
