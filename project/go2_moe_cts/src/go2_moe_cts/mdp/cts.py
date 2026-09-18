# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated compatibility imports; use the role-specific MDP modules."""

import warnings

from .actions import JointPositionActionHistory
from .events import randomize_motor_zero_offset
from .observations_cts import foot_contact_force_norm, joint_acc, joint_effort, single_policy_obs
from .rewards_cts import action_smoothness_l2, feet_regulation, hip_pos_penalty_l1

warnings.warn(
    "go2_moe_cts.mdp.cts is deprecated; use mdp.actions, mdp.observations_cts, mdp.rewards_cts or mdp.events.",
    DeprecationWarning,
    stacklevel=2,
)
__all__ = [
    "JointPositionActionHistory",
    "randomize_motor_zero_offset",
    "foot_contact_force_norm",
    "joint_acc",
    "joint_effort",
    "single_policy_obs",
    "action_smoothness_l2",
    "feet_regulation",
    "hip_pos_penalty_l1",
]
