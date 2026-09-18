# Reference and licensing

Upstream: https://github.com/wertyuilife2/go2_rl_robotlab

Pinned reference revision: `28b4516d22617b11aeaf8ead63cc00b0c0bcd1bd`.

The following files were adapted from that revision:

| Local file | Upstream file |
|---|---|
| `learning/networks.py` | `source/rsl_rl/rsl_rl/networks/moe.py` |
| `learning/policy.py` | `source/rsl_rl/rsl_rl/modules/actor_critic_moe_cts.py` |
| `learning/algorithm.py` | `source/rsl_rl/rsl_rl/algorithms/moe_cts.py` |
| `learning/storage.py` | `source/rsl_rl/rsl_rl/storage/rollout_storage_cts.py` |
| `mdp/terrains.py` | `source/robot_lab/robot_lab/tasks/go2/mdp/terrains.py` |
| `mdp/commands_cts.py` | `source/robot_lab/robot_lab/tasks/go2/mdp/commands.py` |
| `mdp/terrain_utils.py` | `source/robot_lab/robot_lab/tasks/go2/mdp/utils.py` |
| `mdp/curriculums_cts.py` | Selected functions from `source/robot_lab/robot_lab/tasks/go2/mdp/curriculums.py` |

The `learning` implementation retains upstream ETH Zurich/NVIDIA copyright notices and
[BSD-3-Clause license](src/go2_moe_cts/learning/LICENSE.rsl_rl).
The RobotLab reference is distributed under [Apache-2.0](LICENSE.robotlab);
existing notices, including Ziqi Fan's notice on terrain utilities, are retained.
The files above live under `src/go2_moe_cts/`.

Local changes include IsaacLab 6 proxy-array access, local module imports, removal of unused
RND/symmetry dependencies, arbitrary teacher/student ratios, lazy mini-batch iteration to
avoid retaining all epochs on the GPU, and the independent checkpoint/export runner.
The terrain family, expert topology, gradient separation, and two-stage update objectives
follow the pinned reference. Hardware-specific adaptations are listed in README.md.


## Bundled Yuil Dog snapshot

Robot definition, nominal RS03/RS04 motor values, initial crouched pose, joint target limits,
and the hierarchical PhysX contact sensor were copied from the locally validated
`go2_sim2sim` project on 2026-09-18. The snapshot is now owned by this project:

- `src/go2_moe_cts/robots/yuil_dog/`: configuration and all ten USD layers, including embedded mesh data.
- `src/go2_moe_cts/sensors/hierarchical_contact_sensor.py`: local sensor implementation.

There are no runtime imports, symlinks, or default asset paths into the former sibling project.
Changes to that sibling no longer affect this project's robot. Existing copyright notices are retained.
