# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree Go2 asset configuration for go2_moe_cts.

Uses isaaclab_assets UNITREE_GO2_CFG with DCMotor actuators (stiffness=25, damping=0.5).
Joint order follows the leg-major convention (FL/FR/RL/RR × hip/thigh/calf).
"""

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

##
# Joint / body name constants
##

GO2_JOINT_NAMES: tuple[str, ...] = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)
"""Go2 leg-major joint order used by the policy (FL-first, hip→thigh→calf)."""

GO2_FOOT_NAMES: tuple[str, ...] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
"""Go2 foot body names."""

GO2_BASE_HEIGHT_TARGET: float = 0.38
"""Target base height [m] for rewards and height-scan helper."""

##
# Articulation config
##

GO2_CFG = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
"""Go2 articulation config with env-namespace prim path."""
