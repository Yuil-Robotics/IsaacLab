# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout diagnostics and completed-episode statistics for CTS training."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rsl_rl.env import VecEnv


class TrainingMetrics:
    """Track complete episodes, excluding the first partial episode after startup."""

    def __init__(self, env: VecEnv, teacher_ids: torch.Tensor, student_ids: torch.Tensor):
        self.env = env
        self.raw = env.unwrapped
        self.groups = {"teacher": teacher_ids, "student": student_ids}
        self.returns = torch.zeros(env.num_envs, device=env.device)
        self.lengths = torch.zeros(env.num_envs, device=env.device)
        # Initial random episode ages and resume do not provide a complete trajectory.
        self.complete = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.teacher_mask = torch.zeros_like(self.complete)
        self.teacher_mask[teacher_ids] = True
        self.episodes = {name: deque(maxlen=100) for name in ("all", "teacher", "student")}
        self.begin()

    def begin(self) -> None:
        """Clear iteration-local accumulators."""
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def add(self, key: str, value: float, count: int = 1) -> None:
        """Accumulate an average with explicit sample count."""
        self.sums[key] += value * count
        self.counts[key] += count

    def observe(self) -> None:
        """Sample pre-step tracking errors [m/s and rad/s] without reset contamination."""
        if not hasattr(self.raw, "scene"):
            return
        robot = self.raw.scene["robot"]
        command = self.raw.command_manager.get_command("base_velocity")
        linear = robot.data.root_lin_vel_b.torch[:, :2]
        angular = robot.data.root_ang_vel_b.torch[:, 2]
        errors = {
            "linear_error_m_s": (command[:, :2] - linear).norm(dim=-1),
            "yaw_error_rad_s": (command[:, 2] - angular).abs(),
        }
        for name, values in errors.items():
            self.add(f"Tracking/{name}", values.mean().item())
            for group, ids in self.groups.items():
                self.add(f"Tracking/{group}_{name}", values[ids].mean().item())

    def step(self, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Accumulate rewards and classify completed episodes at each reset."""
        rewards = rewards.to(self.returns.device).flatten()
        done = dones.to(self.returns.device).bool().flatten()
        self.returns += rewards
        self.lengths += 1
        self.add("Train/mean_step_reward", rewards.mean().item())
        for group, ids in self.groups.items():
            self.add(f"Train/{group}_mean_step_reward", rewards[ids].mean().item())
        manager = getattr(self.raw, "reward_manager", None)
        if manager is not None:
            # IsaacLab 6 stores weighted reward rates here (before multiplying by step_dt).
            rates = manager._step_reward.mean(dim=0).detach().cpu().tolist()
            for name, rate in zip(manager.active_terms, rates):
                self.add(f"RewardRate/{name}", rate)
        term = getattr(self.raw, "termination_manager", None)
        timeouts = extras.get("time_outs")
        if term is not None:
            timeouts = term.time_outs
            failed = term.terminated.to(done.device).bool()
            for name in term.active_terms:
                self.sums[f"TerminationCount/{name}"] += (term.get_term(name).to(done.device) & done).sum().item()
                self.counts[f"TerminationCount/{name}"] = 1
        else:
            failed = done if timeouts is None else done & ~timeouts.to(done.device).bool()
        success = None if timeouts is None else timeouts.to(done.device).bool() & ~failed
        eligible = done & self.complete
        survival = torch.zeros_like(self.returns) if success is None else success.float()
        # Transfer finished records together to avoid synchronizing once per environment.
        records = torch.stack((self.returns, self.lengths, survival, self.teacher_mask.float()), dim=-1)
        for episode_return, length, survived, teacher in records[eligible].cpu().tolist():
            record = (episode_return, length, None if success is None else survived)
            self.episodes["all"].append(record)
            self.episodes["teacher" if teacher else "student"].append(record)
        self.complete[done] = True
        self.returns[done] = 0
        self.lengths[done] = 0
        # extras['log'] persists between resets: consume it only on an actual reset.
        reset_count = int(done.sum().item())
        if reset_count:
            for key, value in extras.get("log", {}).items():
                if key.startswith("Episode_Termination/"):
                    continue  # Use exact per-step termination signals instead.
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    self.add(key, value.item(), reset_count)
                elif isinstance(value, (int, float)):
                    self.add(key, value, reset_count)

    def values(self) -> dict[str, float]:
        """Return rollout means and rolling statistics over up to 100 full episodes."""
        result = {key: total / self.counts[key] for key, total in self.sums.items()}
        for group, records in self.episodes.items():
            result[f"Episode/{group}_count"] = len(records)
            if not records:
                continue
            result[f"Episode/{group}_return"] = sum(r[0] for r in records) / len(records)
            length = sum(r[1] for r in records) / len(records)
            result[f"Episode/{group}_length_steps"] = length
            if hasattr(self.raw, "step_dt"):
                result[f"Episode/{group}_length_s"] = length * self.raw.step_dt
            known = [r[2] for r in records if r[2] is not None]
            if known:
                result[f"Episode/{group}_timeout_survival_pct"] = 100 * sum(known) / len(known)
        return result


def print_training_summary(iteration: int, target: int, metrics: dict[str, float], log_dir: Path | None) -> None:
    """Print PPO-style diagnostics with explicit units and survival definition."""
    print("\n" + "=" * 88)
    print(f"{'MoE-CTS learning iteration':>43}: {iteration} / {target}")
    print(f"{'Log directory':>43}: {log_dir}")
    labels = {
        "Perf/fps": "Throughput [env steps/s]",
        "Perf/rollout_s": "Rollout time [s]",
        "Perf/update_s": "Update time [s]",
        "Perf/eta_s": "Estimated remaining time [s]",
        "Train/mean_step_reward": "Mean step reward (teacher + student)",
        "Train/teacher_mean_step_reward": "Teacher mean step reward",
        "Train/student_mean_step_reward": "Student mean step reward",
        "Train/learning_rate": "PPO learning rate",
        "Train/student_learning_rate": "Student encoder learning rate",
        "Policy/action_std": "Mean action noise std",
    }
    for key, label in labels.items():
        if key in metrics:
            print(f"{label:>43}: {metrics[key]:.6g}")
    for group in ("all", "teacher", "student"):
        prefix = f"Episode/{group}_"
        if not metrics.get(prefix + "count", 0):
            print(f"{group + ' full episodes':>43}: pending (first partial episode excluded)")
        else:
            for suffix in ("count", "return", "length_steps", "length_s", "timeout_survival_pct"):
                if prefix + suffix in metrics:
                    print(f"{group + ' episode ' + suffix:>43}: {metrics[prefix + suffix]:.6g}")
    print("  Survival = timeout without failure / completed full episodes; rolling last 100 per group.")
    for prefix, title in (
        ("Tracking/", "Velocity tracking: lower error is better"),
        ("RewardRate/", "Weighted reward terms / simulation second (rollout mean)"),
        ("TerminationCount/", "Termination events in this rollout (causes may overlap)"),
        ("Curriculum/terrain_levels/", "Terrain curriculum (reset summaries)"),
        ("Loss/", "PPO and CTS losses"),
        ("MoE/", "Student expert mean gating weights (distillation batches)"),
    ):
        print(f"--- {title} ---")
        for key, value in metrics.items():
            if key.startswith(prefix):
                print(f"{key.removeprefix(prefix):>43}: {value:.6g}")
    print("=" * 88, flush=True)
