# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rough-terrain task with actor observation history and privileged velocity."""

from __future__ import annotations

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass

from .rough_cfg import (
    Go2RoughEvalEnvCfg,
    Go2RoughPlayEnvCfg,
    Go2RoughPPORunnerCfg,
    Go2RoughTrainEnvCfg,
)

ROUGH_HISTORY_TRAIN_TASK_ID = "Isaac-Velocity-Rough-Go2-History-v0"
ROUGH_HISTORY_PLAY_TASK_ID = "Isaac-Velocity-Rough-Go2-History-Play-v0"
ROUGH_HISTORY_EVAL_TASK_ID = "Isaac-Velocity-Rough-Go2-History-Eval-v0"
ROUGH_HISTORY_EXPERIMENT_NAME = "go2_rough_history_5step_sim2real"
ROUGH_HISTORY_LENGTH = 5
ROUGH_HISTORY_ACTOR_STEP_DIM = 45


@configclass
class PrivilegedVelocityCfg(ObsGroup):
    """Current base linear velocity available only to the critic [m/s]."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

    def __post_init__(self) -> None:
        """Keep privileged state uncorrupted and unstacked."""
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 0


def _apply_history_observations(env_cfg: Go2RoughTrainEnvCfg) -> None:
    """Remove actor linear velocity and stack five policy observations."""
    env_cfg.observations.policy.base_lin_vel = None
    env_cfg.observations.policy.history_length = ROUGH_HISTORY_LENGTH
    env_cfg.observations.policy.flatten_history_dim = True
    env_cfg.observations.privileged = PrivilegedVelocityCfg()


@configclass
class Go2RoughHistoryTrainEnvCfg(Go2RoughTrainEnvCfg):
    """Training task using 225 actor inputs and privileged critic velocity."""

    def __post_init__(self) -> None:
        """Apply the history observation contract after the rough task defaults."""
        super().__post_init__()
        _apply_history_observations(self)


@configclass
class Go2RoughHistoryPlayEnvCfg(Go2RoughPlayEnvCfg):
    """Visualization task using the deployable history-only actor input."""

    def __post_init__(self) -> None:
        """Apply the same actor input contract used during training."""
        super().__post_init__()
        _apply_history_observations(self)


@configclass
class Go2RoughHistoryEvalEnvCfg(Go2RoughEvalEnvCfg):
    """Deterministic evaluation task for the history policy."""

    def __post_init__(self) -> None:
        """Apply the same actor input contract used during training."""
        super().__post_init__()
        _apply_history_observations(self)


@configclass
class Go2RoughHistoryPPORunnerCfg(Go2RoughPPORunnerCfg):
    """Asymmetric PPO runner for history actor and privileged critic inputs."""

    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy", "privileged"],
    }
    experiment_name = ROUGH_HISTORY_EXPERIMENT_NAME

    def __post_init__(self) -> None:
        """Restore the separate experiment and asymmetric observation mapping."""
        super().__post_init__()
        self.obs_groups = {
            "actor": ["policy"],
            "critic": ["policy", "privileged"],
        }
        self.experiment_name = ROUGH_HISTORY_EXPERIMENT_NAME
