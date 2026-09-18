#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [checkpoint.pt]" >&2
  exit 2
fi

CHECKPOINT_ARGS=()
if [[ $# -eq 1 ]]; then
  CHECKPOINT_PATH="$(realpath -- "$1")"
  CHECKPOINT_ARGS=(--checkpoint "${CHECKPOINT_PATH}")
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${ISAACLAB_ROOT}/isaaclab.sh" play \
  --rl_library rsl_rl \
  --task Isaac-Reach-UR10e-Sim2Sim-Play-v0 \
  --external_callback ur10_sim2sim.isaaclab_cfg.register_tasks \
  --num_envs 1 \
  --viz none \
  --video \
  --video_length 1 \
  "env.commands.ee_pose.debug_vis=false" \
  "${CHECKPOINT_ARGS[@]}"

echo "Export complete. Find policy.pt and policy.onnx under the run's exported/ directory."
