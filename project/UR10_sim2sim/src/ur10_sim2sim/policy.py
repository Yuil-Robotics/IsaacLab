# Copyright (c) 2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Policy loading and observation construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def build_observation(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    target_pos: np.ndarray,
    target_quat_xyzw: np.ndarray,
) -> np.ndarray:
    """Build the 19-element observation used by the UR10e Reach policy."""
    parts = [
        np.asarray(joint_pos, dtype=np.float32),
        np.asarray(joint_vel, dtype=np.float32),
        np.asarray(target_pos, dtype=np.float32),
        np.asarray(target_quat_xyzw, dtype=np.float32),
    ]
    expected_sizes = (6, 6, 3, 4)
    if tuple(part.size for part in parts) != expected_sizes:
        raise ValueError(
            f"Invalid observation component sizes: {tuple(part.size for part in parts)}; expected {expected_sizes}."
        )
    return np.concatenate(parts)


class TorchScriptPolicy:
    """CPU inference wrapper for an Isaac Lab exported TorchScript policy."""

    def __init__(self, path: str | Path):
        policy_path = Path(path).expanduser().resolve()
        if not policy_path.is_file():
            raise FileNotFoundError(f"Policy does not exist: {policy_path}")
        self._module = torch.jit.load(str(policy_path), map_location="cpu")
        self._module.eval()

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Evaluate one 19-element observation and return six actions."""
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (19,):
            raise ValueError(f"Expected observation shape (19,), got {obs.shape}.")
        with torch.inference_mode():
            action = self._module(torch.from_numpy(obs).unsqueeze(0))
        action_np = action.squeeze(0).detach().cpu().numpy()
        if action_np.shape != (6,):
            raise ValueError(f"Expected policy action shape (6,), got {action_np.shape}.")
        if not np.all(np.isfinite(action_np)):
            raise FloatingPointError(f"Policy produced non-finite action: {action_np}")
        return action_np
