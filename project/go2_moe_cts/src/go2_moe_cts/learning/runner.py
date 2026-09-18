# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacLab 6 / RSL-RL VecEnv bridge for the reference MoE-CTS algorithm."""

from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import torch

from .algorithm import MoECTS
from .exporter import StudentPolicy
from .metrics import TrainingMetrics, print_training_summary
from .policy import ActorCriticMoECTS
from .storage import RolloutStorageCTS

if TYPE_CHECKING:
    from rsl_rl.env import VecEnv


class OnPolicyRunnerCTS:
    """Collect teacher/student rollouts, optimize, checkpoint, and export the student."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        inference_only: bool = False,
    ):
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("The project MoE-CTS runner currently supports one training process.")
        self.inference_only = inference_only
        self.env = env
        self.cfg = copy.deepcopy(train_cfg)
        self.device = device
        self.log_dir = Path(log_dir) if log_dir else None
        self.current_learning_iteration = 0
        obs = env.get_observations().to(device)
        policy = ActorCriticMoECTS(obs, self.cfg["obs_groups"], env.num_actions, **self.cfg["policy"]).to(device)
        if inference_only:
            self.alg = SimpleNamespace(policy=policy)
            self.last_losses = {}
            return
        ratio = self.cfg["algorithm"]["teacher_env_ratio"]
        teacher_count = int(env.num_envs * ratio)
        if not 0 < ratio < 1 or not 0 < teacher_count < env.num_envs:
            raise ValueError("Training needs at least one teacher and one student environment.")
        steps = self.cfg["num_steps_per_env"]
        batches = self.cfg["algorithm"]["num_mini_batches"]
        if (
            steps <= 0
            or batches <= 0
            or any(n * steps % batches for n in (teacher_count, env.num_envs - teacher_count))
        ):
            raise ValueError("Teacher and student rollout sizes must each be divisible by num_mini_batches.")
        storage = RolloutStorageCTS("rl", env.num_envs, teacher_count, steps, obs, [env.num_actions], device)
        self.alg = MoECTS(policy, storage, env.num_envs, device=device, **self.cfg["algorithm"])
        self.last_losses: dict[str, float] = {}
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        shapes = {key: tuple(value.shape) for key, value in obs.items()}
        print(f"[MoE-CTS] teacher={teacher_count}, student={env.num_envs - teacher_count}, obs={shapes}")

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run additional PPO and student-distillation iterations."""
        if self.inference_only:
            raise RuntimeError("An inference-only runner cannot train.")
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations().to(self.device)
        self.alg.policy.train()
        writer = None
        if self.log_dir:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(str(self.log_dir))
        tracker = TrainingMetrics(self.env, self.alg.teacher_env_idxs, self.alg.student_env_idxs)
        target = self.current_learning_iteration + num_learning_iterations
        started = time.perf_counter()
        for_count = 0
        try:
            for _ in range(num_learning_iterations):
                tracker.begin()
                rollout_start = time.perf_counter()
                with torch.inference_mode():
                    for _ in range(self.cfg["num_steps_per_env"]):
                        tracker.observe()
                        actions = self.alg.act(obs)
                        obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                        obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                        if (
                            not all(torch.isfinite(value).all() for value in obs.values())
                            or not torch.isfinite(rewards).all()
                        ):
                            raise FloatingPointError("Non-finite rollout observations or rewards.")
                        self.alg.process_env_step(obs, rewards, dones, extras)
                        tracker.step(rewards, dones, extras)
                    self.alg.compute_returns(obs)
                rollout_s = time.perf_counter() - rollout_start
                update_start = time.perf_counter()
                self.last_losses = self.alg.update()
                update_s = time.perf_counter() - update_start
                for_count += 1
                if not all(np.isfinite(value) for value in self.last_losses.values()):
                    raise FloatingPointError(f"Non-finite MoE-CTS losses: {self.last_losses}")
                self.current_learning_iteration += 1
                iteration = self.current_learning_iteration
                metrics = {f"Loss/{k}": v for k, v in self.last_losses.items()}
                metrics.update(tracker.values())
                metrics["Train/learning_rate"] = self.alg.learning_rate
                metrics["Train/student_learning_rate"] = self.alg.optimizer_stu_enc.param_groups[0]["lr"]
                metrics["Policy/action_std"] = self.alg.policy.action_std.mean().item()
                metrics["Perf/rollout_s"] = rollout_s
                metrics["Perf/update_s"] = update_s
                metrics["Perf/fps"] = self.env.num_envs * self.cfg["num_steps_per_env"] / (rollout_s + update_s)
                metrics["Perf/eta_s"] = (time.perf_counter() - started) / for_count * (target - iteration)
                for expert, usage in enumerate(self.alg.last_expert_usage):
                    metrics[f"MoE/expert_{expert}_weight"] = usage
                if self.log_dir:
                    with (self.log_dir / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps({"iteration": iteration, **metrics}) + "\n")
                if writer:
                    for key, value in metrics.items():
                        writer.add_scalar(key, value, iteration)
                print_training_summary(iteration, target, metrics, self.log_dir)
                if self.log_dir and iteration % self.cfg["save_interval"] == 0:
                    self.save(str(self.log_dir / f"model_{iteration}.pt"))
            if self.log_dir:
                self.save(str(self.log_dir / f"model_{self.current_learning_iteration}.pt"))
        finally:
            if writer:
                writer.close()

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save model, both optimizers, iteration and random-number generator states."""
        torch.save(
            {
                "format": "go2_moe_cts_v1",
                "model_state_dict": self.alg.policy.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "optimizer_stu_enc_state_dict": self.alg.optimizer_stu_enc.state_dict(),
                "learning_rate": self.alg.learning_rate,
                "iter": self.current_learning_iteration,
                "config": self.cfg,
                "infos": infos,
                "rng": {
                    "torch": torch.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = True) -> dict | None:
        """Restore a local CTS checkpoint; flat PPO checkpoints are incompatible."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("format") != "go2_moe_cts_v1":
            raise ValueError("Expected a MoE-CTS checkpoint. A Phase-1 PPO checkpoint cannot be resumed as CTS.")
        for key in ("policy", "obs_groups"):
            if checkpoint["config"][key] != self.cfg[key]:
                raise ValueError(f"Checkpoint {key} does not match the requested configuration.")
        self.alg.policy.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and not self.inference_only:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.alg.optimizer_stu_enc.load_state_dict(checkpoint["optimizer_stu_enc_state_dict"])
            self.alg.learning_rate = checkpoint["learning_rate"]
            rng = checkpoint["rng"]
            torch.set_rng_state(rng["torch"].cpu())
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
            if torch.cuda.is_available() and rng["cuda"] is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
        self.current_learning_iteration = checkpoint["iter"]
        if hasattr(self.env.unwrapped, "common_step_counter"):
            self.env.unwrapped.common_step_counter = self.current_learning_iteration * self.cfg["num_steps_per_env"]
        return checkpoint["infos"]

    def get_inference_policy(self, device: str | None = None):
        """Return student inference accepting the environment's observation dictionary."""
        self.alg.policy.eval()
        module = StudentPolicy(self.alg.policy).to(device or self.device).eval()

        class Inference:
            def __call__(self, obs):
                return module(obs["policy"])

            def reset(self, dones=None):
                module.reset(dones)

        return Inference()

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the blind student with a 450D history input and 12D action output."""
        Path(path).mkdir(parents=True, exist_ok=True)
        module = StudentPolicy(self.alg.policy).cpu().eval()
        torch.jit.script(module).save(str(Path(path) / filename))
        self._export_contract(path)

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx") -> None:
        """Export the same student interface to ONNX."""
        Path(path).mkdir(parents=True, exist_ok=True)
        module = StudentPolicy(self.alg.policy).cpu().eval()
        torch.onnx.export(
            module,
            torch.zeros(1, module.history_dim),
            str(Path(path) / filename),
            input_names=["obs"],
            output_names=["actions"],
            opset_version=17,
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
            dynamo=False,
        )
        self._export_contract(path)

    def _export_contract(self, path: str) -> None:
        contract = {
            "algorithm": "MoE-CTS student",
            "history_layout": "term-major, oldest-to-newest",
            "input_dim": self.alg.policy.num_actor_obs,
            "action_dim": self.env.num_actions,
            "observation_terms": [
                "base_ang_vel",
                "projected_gravity",
                "velocity_commands",
                "joint_pos",
                "joint_vel",
                "last_action",
            ],
            "observation_scales": [0.25, 1.0, 1.0, 1.0, 0.05, 1.0],
        }
        env_cfg = getattr(self.env.unwrapped, "cfg", None)
        if env_cfg is not None:
            contract.update(
                {
                    "control_dt": env_cfg.sim.dt * env_cfg.decimation,
                    "joint_names": env_cfg.actions.joint_pos.joint_names,
                    "action_scale": env_cfg.actions.joint_pos.scale,
                    "joint_position_clip": env_cfg.actions.joint_pos.clip,
                    "default_joint_pos": env_cfg.scene.robot.init_state.joint_pos,
                }
            )
        (Path(path) / "policy_contract.json").write_text(json.dumps(contract, indent=2))

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Record the repository revision for the training script."""
        if self.log_dir:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(repo_file_path).parent,
                capture_output=True,
                text=True,
                check=False,
            )
            (self.log_dir / "git_revision.txt").write_text(result.stdout)
