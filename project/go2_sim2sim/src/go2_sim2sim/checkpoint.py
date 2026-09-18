# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint path resolution for Go2 policy tools."""

from __future__ import annotations

import re
from pathlib import Path

_CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt$")


def _checkpoint_iteration(path: Path) -> int:
    """Return the iteration encoded in a checkpoint filename."""
    match = _CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Invalid checkpoint filename: {path.name}")
    return int(match.group(1))


def resolve_checkpoint(path: Path) -> Path:
    """Resolve a checkpoint file or select the latest checkpoint under a directory.

    When :paramref:`path` directly contains checkpoints, the highest numbered
    ``model_<iteration>.pt`` file is selected. Otherwise, the lexicographically
    latest run directory containing checkpoints is selected first, followed by
    its highest numbered checkpoint.

    Args:
        path: Checkpoint file, run directory, or experiment log directory.

    Returns:
        Resolved checkpoint file path.

    Raises:
        FileNotFoundError: If the path or a matching checkpoint does not exist.
    """
    resolved_path = path.expanduser().resolve()
    if resolved_path.is_file():
        return resolved_path
    if not resolved_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {resolved_path}")
    if not resolved_path.is_dir():
        raise FileNotFoundError(f"Checkpoint path is not a file or directory: {resolved_path}")

    direct_checkpoints = [candidate for candidate in resolved_path.glob("model_*.pt") if candidate.is_file()]
    if direct_checkpoints:
        return max(
            direct_checkpoints,
            key=lambda candidate: (_checkpoint_iteration(candidate), candidate.stat().st_mtime_ns),
        )

    nested_checkpoints = [candidate for candidate in resolved_path.rglob("model_*.pt") if candidate.is_file()]
    if not nested_checkpoints:
        raise FileNotFoundError(f"No model_<iteration>.pt checkpoint found under: {resolved_path}")
    latest_run_dir = max({candidate.parent for candidate in nested_checkpoints}, key=lambda directory: str(directory))
    run_checkpoints = [candidate for candidate in nested_checkpoints if candidate.parent == latest_run_dir]
    return max(run_checkpoints, key=lambda candidate: (_checkpoint_iteration(candidate), candidate.stat().st_mtime_ns))
