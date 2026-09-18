#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Play the latest Go2 checkpoint with the Kit GUI.
# Usage: ./scripts/play_go2.sh [--checkpoint PATH] [--num_envs N] [extra args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ISAACLAB_DIR="$(dirname "$(dirname "$PROJECT_DIR")")"

cd "$ISAACLAB_DIR"

"$ISAACLAB_DIR/isaaclab.sh" -p "$PROJECT_DIR/scripts/evaluation/play.py" \
    --task Isaac-Velocity-Rough-Go2-MoECTS-Play-v0 \
    --num_envs 16 \
    --viz kit \
    "$@"
