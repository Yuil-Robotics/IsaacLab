#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

prepare_project_environment
ensure_go2_asset

echo "[Go2 USD Editor] Launching Isaac Sim GUI with Go2 USD asset..."
"${ISAACLAB_ROOT}/isaaclab.sh" -p "${SCRIPT_DIR}/edit_usd.py" --visualizer kit "$@"
