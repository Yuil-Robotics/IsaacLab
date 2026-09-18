#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd -- "${PROJECT_DIR}/../../.." && pwd)"
ISAACLAB_LAUNCHER="${ISAACLAB_ROOT}/isaaclab.sh"

if [[ ! -x "${ISAACLAB_LAUNCHER}" ]]; then
  echo "Isaac Lab launcher is missing or not executable: ${ISAACLAB_LAUNCHER}" >&2
  exit 1
fi

exec "${ISAACLAB_LAUNCHER}" -p "${PROJECT_DIR}/scripts/play.py" "$@"
