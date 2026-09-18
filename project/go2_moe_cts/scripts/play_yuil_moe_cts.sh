#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ISAACLAB_DIR="$(dirname "$(dirname "$PROJECT_DIR")")"
cd "$ISAACLAB_DIR"
exec ./isaaclab.sh -p "$SCRIPT_DIR/evaluation/play.py" --task Isaac-Velocity-Rough-Yuil-Dog-MoECTS-Play-v0 --viz kit "$@"
