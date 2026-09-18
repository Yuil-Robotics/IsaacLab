#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Spawn the Yuil Dog robot in mid-air for interactive pose inspection.
# Usage:
#   bash scripts/view_pose.sh [--spawn_height 1.2] [--headless]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$(cd "${PROJECT_ROOT}/../.." && pwd)}"

if [[ ! -f "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
    echo "ERROR: Cannot locate isaaclab.sh at ${ISAACLAB_ROOT}/isaaclab.sh"
    echo "       Set ISAACLAB_ROOT to the IsaacLab repository root."
    exit 1
fi

exec "${ISAACLAB_ROOT}/isaaclab.sh" -p "${SCRIPT_DIR}/view_pose.py" "$@"
