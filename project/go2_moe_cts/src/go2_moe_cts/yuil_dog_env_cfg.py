# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated compatibility path for the Yuil Dog task configuration."""

import warnings

from .tasks.yuil_dog.env_cfg import YuilDogMoECTSEnvCfg, YuilDogMoECTSEvalEnvCfg, YuilDogMoECTSPlayEnvCfg
from .tasks.yuil_dog.observations_cfg import ObservationsCTSCfg
from .tasks.yuil_dog.rewards_cfg import RewardsCTSCfg

warnings.warn(
    "go2_moe_cts.yuil_dog_env_cfg is deprecated; use go2_moe_cts.tasks.yuil_dog.env_cfg.",
    DeprecationWarning,
    stacklevel=2,
)
__all__ = [
    "YuilDogMoECTSEnvCfg",
    "YuilDogMoECTSEvalEnvCfg",
    "YuilDogMoECTSPlayEnvCfg",
    "ObservationsCTSCfg",
    "RewardsCTSCfg",
]
