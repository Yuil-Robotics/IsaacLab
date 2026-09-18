# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain progression and scheduled reward weights."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from ...mdp import curriculums_cts


def configure_curriculum(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Configure terrain levels and the height/vertical-velocity reward schedules."""
    env_cfg.curriculum.terrain_levels = CurrTerm(
        func=curriculums_cts.terrain_levels_tracking,
        params={"promote_threshold": 0.7, "demote_threshold": 0.5},
    )
    env_cfg.curriculum.base_linear_velocity = CurrTerm(
        func=curriculums_cts.gradual_reward_weight_modification,
        params={
            "term_name": "lin_vel_z_l2",
            "initial_weight": -2.0,
            "final_weight": 0.0,
            "start_it": 0,
            "end_it": 1500,
        },
    )
    env_cfg.curriculum.base_height = CurrTerm(
        func=curriculums_cts.gradual_reward_weight_modification,
        params={
            "term_name": "base_height_l2",
            "initial_weight": -1.0,
            "final_weight": -10.0,
            "start_it": 0,
            "end_it": 5000,
        },
    )
