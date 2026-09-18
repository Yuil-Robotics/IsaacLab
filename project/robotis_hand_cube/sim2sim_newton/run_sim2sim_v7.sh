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

# Set V7_NUM_ENVS for a convenient batch default, or pass --num_envs directly.
# An explicit CLI argument takes precedence and preserves normal argparse help/errors.
V7_NUM_ENVS="${V7_NUM_ENVS:-1}"
num_envs_is_explicit=false
for arg in "$@"; do
  if [[ "${arg}" == "--num_envs" || "${arg}" == --num_envs=* ]]; then
    num_envs_is_explicit=true
    break
  fi
done

default_args=()
if [[ "${num_envs_is_explicit}" == false ]]; then
  if [[ ! "${V7_NUM_ENVS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "V7_NUM_ENVS must be a positive integer, got: ${V7_NUM_ENVS}" >&2
    exit 2
  fi
  default_args=(--num_envs "${V7_NUM_ENVS}")
fi

exec "${ISAACLAB_LAUNCHER}" -p "${PROJECT_DIR}/scripts/play.py" --v7 "${default_args[@]}" "$@"
