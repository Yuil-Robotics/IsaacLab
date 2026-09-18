#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export ROBOT_TYPE="go2"
export GO2_SIM2SIM_CHECKPOINT_ROOT="${PROJECT_ROOT}/logs/rsl_rl/go2_rough_history_5step_sim2real"
export GO2_SIM2SIM_TASK="Isaac-Velocity-Rough-Go2-History-Eval-v0"
export GO2_SIM2SIM_LABEL="Go2-Rough-History Sim2Sim"
export GO2_SIM2SIM_RESULT_SUFFIX="_history"

exec "${SCRIPT_DIR}/run_sim2sim.sh" "$@"
