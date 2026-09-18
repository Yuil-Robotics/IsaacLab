# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration serialized with the July 2026 Robotis benchmark."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class RobotisHandPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Reproduce the PPO hyperparameters used by the reference benchmark."""

    seed = 42
    num_steps_per_env = 16
    max_iterations = 50000
    save_interval = 250
    experiment_name = "robotis_hand_state_based_small_cube"
    clip_actions = None
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[1024, 512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.4, std_type="scalar"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[1024, 512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
    )


@configclass
class RobotisHandSingleGoalPPORunnerCfg(RobotisHandPPORunnerCfg):
    """Save the A-task training state every 100 learning iterations."""

    save_interval = 100
    # Preserve raw policy outputs for the environment-side action-bound reward.
    clip_actions = None
