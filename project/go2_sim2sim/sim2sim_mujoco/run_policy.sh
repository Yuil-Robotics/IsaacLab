#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd -- "${TASK_DIR}/../../.." && pwd)"

cd "${TASK_DIR}"
exec "${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/run_policy_client.py --rate_hz 50 "$@"
