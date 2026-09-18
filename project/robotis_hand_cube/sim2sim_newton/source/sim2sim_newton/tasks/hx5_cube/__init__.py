# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robotis HX5 cube reorientation task registration."""

import gymnasium as gym

from . import agents
from .hx5_cube_env_cfg import RobotisHandEnvCfg
from .hx5_cube_newton_env_cfg import Hx5CubeNewtonEnvCfg, Hx5CubeNewtonSingleGoalEnvCfg
from .hx5_cube_robust_train_env_cfg import RobotisHandRobustTrainEnvCfg
from .hx5_cube_single_goal_train_env_cfg import (
    RobotisHandSingleGoalNominalTrainEnvCfg,
    RobotisHandSingleGoalNominalPlayEnvCfg,
    RobotisHandSingleGoalPlayEnvCfg,
    RobotisHandSingleGoalTrainEnvCfg,
)
from .hx5_cube_v7_newton_env_cfg import Hx5CubeV7NewtonEnvCfg

##
# Shared runtime entry points
##

_benchmark_env_cls = f"{__name__}.hx5_cube_physx_env:RobotisHandBenchmarkEnv"
_single_goal_play_env_cls = f"{__name__}.hx5_cube_physx_env:RobotisHandSingleGoalPlayEnv"
_single_goal_hold_play_env_cls = f"{__name__}.hx5_cube_physx_env:RobotisHandSingleGoalHoldPlayEnv"

##
# Register Gym environments.
##

# -- PhysX training (base, no DR)
gym.register(
    id="Isaac-Repose-Cube-Robotis-Direct-v0",
    entry_point=_benchmark_env_cls,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hx5_cube_env_cfg:RobotisHandEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RobotisHandPPORunnerCfg",
    },
)

# -- PhysX training (with domain randomization)
gym.register(
    id="Isaac-Repose-Cube-Robotis-Robust-Train-v0",
    entry_point=_benchmark_env_cls,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hx5_cube_robust_train_env_cfg:RobotisHandRobustTrainEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RobotisHandPPORunnerCfg",
    },
)

# -- PhysX single-goal Sim2Real training (with domain randomization)
gym.register(
    id="Isaac-Repose-Cube-Robotis-Single-Goal-Train-v0",
    entry_point=_benchmark_env_cls,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hx5_cube_single_goal_train_env_cfg:RobotisHandSingleGoalTrainEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RobotisHandSingleGoalPPORunnerCfg",
    },
)

# -- PhysX single-goal curriculum stage 1 (without Sim2Real randomization)
gym.register(
    id="Isaac-Repose-Cube-Robotis-Single-Goal-Nominal-Train-v0",
    entry_point=_benchmark_env_cls,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hx5_cube_single_goal_train_env_cfg:RobotisHandSingleGoalNominalTrainEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RobotisHandSingleGoalPPORunnerCfg",
    },
)

# -- PhysX single-goal evaluation with exclusive terminal statistics
gym.register(
    id="Isaac-Repose-Cube-Robotis-Single-Goal-Play-v0",
    entry_point=_single_goal_play_env_cls,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hx5_cube_single_goal_train_env_cfg:RobotisHandSingleGoalPlayEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RobotisHandSingleGoalPPORunnerCfg",
    },
)

# -- PhysX single-goal nominal evaluation without domain randomization
gym.register(
    id="Isaac-Repose-Cube-Robotis-Single-Goal-Nominal-Play-v0",
    entry_point=_single_goal_play_env_cls,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hx5_cube_single_goal_train_env_cfg:RobotisHandSingleGoalNominalPlayEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RobotisHandSingleGoalPPORunnerCfg",
    },
)

# -- PhysX A/B evaluation: hold the reset joint pose for 15 policy steps
gym.register(
    id="Isaac-Repose-Cube-Robotis-Single-Goal-Hold15-Play-v0",
    entry_point=_single_goal_hold_play_env_cls,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hx5_cube_single_goal_train_env_cfg:RobotisHandSingleGoalPlayEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RobotisHandSingleGoalPPORunnerCfg",
    },
)

# -- Newton/MJWarp play (sim-to-sim policy playback)
gym.register(
    id="Isaac-Repose-Cube-Robotis-HX5-Newton-Play-v0",
    entry_point=f"{__name__}.hx5_cube_env:Hx5CubeNewtonEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hx5_cube_newton_env_cfg:Hx5CubeNewtonEnvCfg",
    },
)

# -- Newton/MJWarp single-goal evaluation
gym.register(
    id="Isaac-Repose-Cube-Robotis-HX5-Newton-Single-Goal-Play-v0",
    entry_point=f"{__name__}.hx5_cube_env:Hx5CubeNewtonEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hx5_cube_newton_env_cfg:Hx5CubeNewtonSingleGoalEnvCfg",
    },
)

# -- Newton/MJWarp V7 continuous reorientation playback
gym.register(
    id="Isaac-Repose-Cube-Robotis-HX5-Newton-V7-Play-v0",
    entry_point=f"{__name__}.hx5_cube_v7_env:Hx5CubeV7NewtonEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hx5_cube_v7_newton_env_cfg:Hx5CubeV7NewtonEnvCfg",
    },
)

__all__ = [
    "RobotisHandEnvCfg",
    "RobotisHandRobustTrainEnvCfg",
    "RobotisHandSingleGoalNominalTrainEnvCfg",
    "RobotisHandSingleGoalNominalPlayEnvCfg",
    "RobotisHandSingleGoalPlayEnvCfg",
    "RobotisHandSingleGoalTrainEnvCfg",
    "Hx5CubeNewtonEnvCfg",
    "Hx5CubeNewtonSingleGoalEnvCfg",
    "Hx5CubeV7NewtonEnvCfg",
]
