#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# The 2026-09-16_17-03-34 run developed persistent action saturation and rapid
# stepping. Restart recovery fine-tuning from the last verified stable policy.
STABLE_RUN="${YUIL_DOG_RECOVERY_LOAD_RUN:-2026-09-16_11-08-38}"
STABLE_CHECKPOINT="${YUIL_DOG_RECOVERY_CHECKPOINT:-model_24995.pt}"

exec "${SCRIPT_DIR}/train_yuil_dog_robotlab.sh" \
  --load_run "${STABLE_RUN}" \
  --checkpoint "${STABLE_CHECKPOINT}" \
  "$@"
