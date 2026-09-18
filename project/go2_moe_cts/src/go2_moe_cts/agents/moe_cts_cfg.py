# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Concurrent teacher/student settings from go2_rl_robotlab, commit 28b4516."""

from isaaclab.utils.configclass import configclass


@configclass
class MoECTSRunnerCfg:
    """Project-local runner configuration, independent of the RSL-RL 5 model API."""

    class_name: str = "OnPolicyRunnerCTS"
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 300000
    save_interval: int = 500
    experiment_name: str = "yuil_dog_rough_moe_cts"
    run_name: str = ""
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    clip_actions: float = 100.0
    obs_groups: dict = {"policy": ["policy"], "critic": ["critic"]}
    policy: dict = {
        "expert_num": 8,
        "latent_dim": 32,
        "norm_type": "l2norm",
        "teacher_encoder_hidden_dims": [512, 256],
        "student_encoder_hidden_dims": [512, 256, 256],
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
        "init_noise_std": 1.0,
        "noise_std_type": "log",
    }
    algorithm: dict = {
        "teacher_env_ratio": 0.75,
        "load_balance_coef": 0.01,
        "student_encoder_learning_rate": 1e-3,
        "learning_rate": 1e-3,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "clip_param": 0.2,
        "value_loss_coef": 1.0,
        "entropy_coef": 0.01,
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "schedule": "adaptive",
    }
