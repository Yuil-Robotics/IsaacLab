#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export ROBOT_TYPE="yuil_dog"
export GO2_ROUGH_PLAY_TASK_ID="Isaac-Velocity-Flat-Yuil-Dog-Low-Profile-Play-v0"
export GO2_ROUGH_PLAY_CHECKPOINT_ROOT="${PROJECT_ROOT}/logs/rsl_rl/yuil_dog_flat_low_profile"
export GO2_ROUGH_PLAY_LABEL="Yuil-Dog-Flat-Low-Profile"

exec "${SCRIPT_DIR}/play_rough.sh" "$@"
