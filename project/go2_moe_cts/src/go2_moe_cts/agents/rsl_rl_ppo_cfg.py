# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO runner configuration for go2_moe_cts.

Uses the IsaacLab 6.1 RSL-RL API:
- ``RslRlMLPModelCfg`` with ``GaussianDistributionCfg`` for actor
- ``obs_groups`` dict to map actor/critic to observation group names
- Actor reads ``policy`` group; Critic reads ``policy`` + ``critic`` groups

"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Go2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """On-policy PPO runner with an asymmetric actor-critic.

    - Actor reads the 450-dim ``policy`` observation history.
    - Critic reads ``policy`` + ``critic`` obs groups (includes privileged terrain info).
    """

    num_steps_per_env: int = 24
    max_iterations: int = 15000
    save_interval: int = 200
    experiment_name: str = "go2_moe_cts_45x10"
    empirical_normalization: bool = False

    # Asymmetric obs groups: actor uses policy, critic uses policy+critic
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy", "critic"],
    }

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        """Restore obs_groups after configclass resolution."""
        super().__post_init__()
        self.obs_groups = {
            "actor": ["policy"],
            "critic": ["policy", "critic"],
        }
        self.experiment_name = "go2_moe_cts_45x10"
