# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint actions and per-environment action history."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv, mdp


class JointPositionActionHistory(mdp.JointPositionAction):
    """Position action retaining two previous normalized actions for smoothness."""

    def __init__(self, cfg: mdp.JointPositionActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.previous = torch.zeros_like(self.raw_actions)
        self.previous_previous = torch.zeros_like(self.raw_actions)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Advance history once per policy step before processing the new action."""
        self.previous_previous.copy_(self.previous)
        self.previous.copy_(self.raw_actions)
        super().process_actions(actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset action history for environments that ended their episode."""
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.previous[ids] = 0
        self.previous_previous[ids] = 0
