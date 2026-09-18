# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv


def gradual_reward_weight_modification(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    initial_weight: float,
    final_weight: float,
    start_it: int,
    end_it: int,
):
    """Curriculum that gradually modifies a reward weight between an initial and final value over a range of steps."""
    current_it = env.common_step_counter // 24
    if current_it < start_it:
        return

    if current_it >= end_it:
        new_weight = final_weight
    else:
        new_weight = (current_it - start_it) / (end_it - start_it) * (final_weight - initial_weight) + initial_weight

    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.weight = new_weight
    env.reward_manager.set_term_cfg(term_name, term_cfg)


def terrain_levels_vel_gym(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> float:
    """
    使用 max_move_distance 而非 reset 时的瞬间位移, 比较标准基于 commands_xy_accumulation
    """
    terrain = env.scene.terrain
    command = env.command_manager.get_term("base_velocity")

    max_move_dist = command.max_move_distance[env_ids]
    cmd_accum = command.commands_xy_accumulation[env_ids]

    resampling_time = command.cfg.resampling_time
    zero_prob = command.zero_command_prob

    terrain_cfg = terrain.cfg.terrain_generator
    sub_terrain_border_width = getattr(terrain_cfg, "sub_terrain_border_width", 0.0) or 0.0
    terrain_length = max(0.0, terrain_cfg.size[0] - 2.0 * sub_terrain_border_width)
    move_up = max_move_dist > terrain_length / 2
    target_dist = torch.norm(cmd_accum, dim=1) * (resampling_time * (1 - zero_prob))
    move_down = (max_move_dist < target_dist * 0.5) * ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)

    return torch.mean(terrain.terrain_levels.float())  # type: ignore


def terrain_levels_tracking(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    promote_threshold: float = 0.7,
    demote_threshold: float = 0.5,
) -> dict[str, float]:
    """Update terrain level using mean episode command-following score.

    The final pre-reset state is included. No samples means hold. Scores at
    0.5 hold and scores at 0.7 promote (floating-point tolerance 1e-6).
    """
    if not 0 <= demote_threshold < promote_threshold <= 1:
        raise ValueError("Tracking thresholds must satisfy 0 <= demote < promote <= 1.")
    command = env.command_manager.get_term("base_velocity")
    command.record_tracking(env_ids)
    count = command.tracking_sample_count[env_ids]
    score = command.tracking_score_sum[env_ids] / count.clamp_min(1)
    valid = count > 0
    move_up = valid & (score + 1e-6 >= promote_threshold)
    move_down = valid & (score + 1e-6 < demote_threshold)
    env.scene.terrain.update_env_origins(env_ids, move_up, move_down)
    return {
        "mean_level": env.scene.terrain.terrain_levels.float().mean().item(),
        "tracking_pct": score[valid].mean().item() * 100 if valid.any() else 0.0,
        "promoted_count": move_up.sum().item(),
        "demoted_count": move_down.sum().item(),
        "held_count": (~(move_up | move_down)).sum().item(),
        "evaluated_count": valid.sum().item(),
    }
