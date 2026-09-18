# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stateless student export using the existing 450D term-major history contract."""

import copy

import torch
from torch import nn

from .policy import ActorCriticMoECTS


class StudentPolicy(nn.Module):
    """Deployable student with history input and normalized joint-position actions.

    The caller maintains term-major history, oldest to newest within each term.
    No privileged observation or teacher is retained in the exported module.
    """

    def __init__(self, policy: ActorCriticMoECTS):
        super().__init__()
        self.encoder = copy.deepcopy(policy.student_moe_encoder)
        self.actor = copy.deepcopy(policy.actor)
        self.history_normalizer = copy.deepcopy(policy.actor_obs_normalizer)
        self.single_normalizer = copy.deepcopy(policy.single_obs_normalizer)
        if policy.state_dependent_std:
            raise ValueError("Student export requires state-independent action noise.")
        self.history_dim = policy.num_actor_obs
        if self.history_dim % policy.num_single_obs:
            raise ValueError("History dimension must be a multiple of the single-frame dimension.")
        history_length = self.history_dim // policy.num_single_obs
        indices = []
        offset = 0
        for width in (3, 3, 3, policy.num_actions, policy.num_actions, policy.num_actions):
            indices.extend(range(offset + (history_length - 1) * width, offset + history_length * width))
            offset += history_length * width
        if offset != self.history_dim:
            raise ValueError("Unsupported proprioceptive observation layout.")
        self.register_buffer("latest_indices", torch.tensor(indices, dtype=torch.long))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Return joint-position actions for a batch of observation histories."""
        single = self.single_normalizer(history.index_select(-1, self.latest_indices))
        latent, _ = self.encoder(self.history_normalizer(history))
        return self.actor(torch.cat((latent, single), dim=-1))

    @torch.jit.export
    def reset(self, dones: torch.Tensor | None = None) -> None:
        """No internal state; the caller resets history when an episode ends."""
