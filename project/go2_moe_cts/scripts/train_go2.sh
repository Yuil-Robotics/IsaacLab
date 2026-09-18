#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Train Go2 locomotion with 45-dim × 10-frame observations.
# Usage: ./scripts/train_go2.sh [--num_envs 4096] [--gui] [--resume] [extra args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ISAACLAB_DIR="$(dirname "$(dirname "$PROJECT_DIR")")"

cd "$ISAACLAB_DIR"

"$ISAACLAB_DIR/isaaclab.sh" -p "$PROJECT_DIR/scripts/training/train.py" \
    --task Isaac-Velocity-Rough-Go2-MoECTS-v0 \
    --num_envs 4096 \
    "$@"
