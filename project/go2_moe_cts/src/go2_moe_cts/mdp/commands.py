# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity command configuration for go2_moe_cts.

Uses the standard IsaacLab UniformVelocityCommandCfg with Go2-matched ranges.
"""

from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.utils.configclass import configclass


@configclass
class Go2VelocityCommandCfg(UniformVelocityCommandCfg):
    """Velocity command sampled uniformly for Go2 locomotion training.

    Ranges match go2_rl_robotlab:
    - Linear x:  [-1.0, 1.0] m/s
    - Linear y:  [-0.5, 0.5] m/s
    - Angular z: [-1.0, 1.0] rad/s

    10% standing envs, heading command disabled (pure velocity tracking).
    """

    asset_name: str = "robot"
    resampling_time_range: tuple = (10.0, 10.0)
    rel_standing_envs: float = 0.10
    rel_heading_envs: float = 0.0
    heading_command: bool = False
    debug_vis: bool = False

    ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-0.5, 0.5),
        ang_vel_z=(-1.0, 1.0),
        heading=None,
    )
