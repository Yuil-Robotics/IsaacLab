# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree Go2 locomotion with deployable observation history and privileged critic observations.

Task IDs
--------
- ``Isaac-Velocity-Rough-Go2-MoECTS-v0``   : Training task (4096 envs, curriculum).
- ``Isaac-Velocity-Rough-Go2-MoECTS-Play-v0`` : Evaluation with 16 envs, debug vis.
- ``Isaac-Velocity-Rough-Go2-MoECTS-Eval-v0`` : Deterministic benchmark (256 envs).
"""

import gymnasium as gym

from .agents.rsl_rl_ppo_cfg import Go2PPORunnerCfg
from .env_cfg import Go2EnvCfg, Go2EvalEnvCfg, Go2PlayEnvCfg

##
# Register tasks
##

gym.register(
    id="Isaac-Velocity-Rough-Go2-MoECTS-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2EnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:Go2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Go2-MoECTS-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2PlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:Go2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Go2-MoECTS-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2EvalEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:Go2PPORunnerCfg",
    },
)

# Keep Phase-1 task IDs intact. The Yuil CTS task uses a separate runner and checkpoint format.
for _suffix, _cfg in (
    ("", "YuilDogMoECTSEnvCfg"),
    ("-Play", "YuilDogMoECTSPlayEnvCfg"),
    ("-Eval", "YuilDogMoECTSEvalEnvCfg"),
):
    gym.register(
        id=f"Isaac-Velocity-Rough-Yuil-Dog-MoECTS{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.yuil_dog.env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.moe_cts_cfg:MoECTSRunnerCfg",
        },
    )
