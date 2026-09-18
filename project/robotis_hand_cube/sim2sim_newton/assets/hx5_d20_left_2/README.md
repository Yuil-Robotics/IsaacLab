<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
-->

# Robot USD placement

The temporary HX5 USD generated from the MuJoCo MJCF is available at:

`assets/hx5_d20_left_fixed/hx5_d20_left/hx5_d20_left.usda`

It was generated from `sim2sim-mujoco/model/MJCF/hx5_d20_left.xml` with the
IsaacLab MJCF converter and includes all referenced geometry and physics layers.

When the original asset arrives, copy its complete directory here, including
meshes and referenced USD layers, and select its main file with `--robot_usd`.

The file can instead remain elsewhere when `--robot_usd` or the
`ROBOTIS_HAND_USD` environment variable points to it.
