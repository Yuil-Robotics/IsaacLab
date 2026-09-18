#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export GO2_ROUGH_TRAIN_TASK_ID="Isaac-Velocity-Rough-Go2-History-v0"
export GO2_ROUGH_CHECKPOINT_ROOT="${PROJECT_ROOT}/logs/rsl_rl/go2_rough_history_5step_sim2real"
export GO2_ROUGH_TRAIN_LABEL="Go2-Rough-History"

exec "${SCRIPT_DIR}/train_rough.sh" "$@"
