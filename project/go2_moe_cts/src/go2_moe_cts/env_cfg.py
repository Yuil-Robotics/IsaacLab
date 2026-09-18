# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 locomotion environment configuration.

Observation groups:
- **policy** : 45-dim deployable observation with 10-step history (450-dim Actor input).
- **critic** : policy history plus 163-dim privileged information.

Modelling choices mirror go2_rl_robotlab:
- Unitree DCMotor model (stiffness=25, damping=0.5)
- Fixed-sigma velocity tracking rewards (std=0.5)
- A large RayCaster used only by the critic during training
- Terrain curriculum containing only flat and random rough ground
"""

from __future__ import annotations

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp

from .assets import GO2_BASE_HEIGHT_TARGET, GO2_CFG, GO2_FOOT_NAMES, GO2_JOINT_NAMES
from .mdp import (
    Go2VelocityCommandCfg,
    height_scan_large,
    joint_pos_penalty_l1,
    joint_power,
    stand_still,
    track_ang_vel_z_exp,
    track_lin_vel_xy_exp,
)

##
# Terrain configuration (flat + random rough ground only)
##

TERRAIN_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=True,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.5,
            noise_range=(0.01, 0.08),
            noise_step=0.01,
            border_width=0.25,
        ),
    },
)
"""Flat and random-rough terrain generator with curriculum enabled."""


##
# Scene definition
##


@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    """Interactive scene with Go2 robot, privileged height scanner, and contact sensor."""

    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TERRAIN_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.18, 0.20, 0.22),
            roughness=0.9,
        ),
        debug_vis=False,
    )

    # Robot (Unitree Go2 with DCMotor model)
    robot: ArticulationCfg = GO2_CFG

    # Large height scanner → critic-only privileged observation (1.5×0.9 m, 160 rays)
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.5, 0.9)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # Contact sensor (all bodies, history_length=3 for air-time tracking)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # Sky light
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
        ),
    )


##
# MDP: Commands
##


@configclass
class CommandsCfg:
    """Velocity command specification."""

    base_velocity: Go2VelocityCommandCfg = Go2VelocityCommandCfg()


##
# MDP: Actions
##


@configclass
class ActionsCfg:
    """12-DOF joint position action (scale=0.25, relative to default pose)."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(GO2_JOINT_NAMES),
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )


##
# MDP: Observations
##

_JOINT_ASSET_CFG = SceneEntityCfg("robot", joint_names=list(GO2_JOINT_NAMES), preserve_order=True)
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=list(GO2_FOOT_NAMES))


@configclass
class ObservationsCfg:
    """Observation groups.

    - **policy**: ten frames of 45-dim deployable observations (450 dim).
    - **critic**: privileged terrain and velocity (163 dim), appended to policy history.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Ten-frame history of observations available on the physical robot."""

        # Base angular velocity (IMU) [rad/s], shape [3]
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        # Gravity projection in body frame, shape [3]
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
        )
        # Velocity command (lin_x, lin_y, ang_z), shape [3]
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
        )
        # Joint positions relative to default, shape [12]
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _JOINT_ASSET_CFG},
            noise=Unoise(n_min=-0.03, n_max=0.03),
            clip=(-100.0, 100.0),
        )
        # Joint velocities, shape [12]
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _JOINT_ASSET_CFG},
            noise=Unoise(n_min=-2.0, n_max=2.0),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        # Last action (for action history), shape [12]
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))

        # Corruption enabled during training, disabled during eval
        enable_corruption: bool = True
        concatenate_terms: bool = True
        history_length: int = 10
        flatten_history_dim: bool = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged terms appended to the policy observations for the critic."""

        # Large height scan (1.5×0.9 m, 160 rays)
        height_scan_large = ObsTerm(
            func=height_scan_large,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
        )
        # Base linear velocity (not measurable from an IMU alone)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        enable_corruption: bool = False
        concatenate_terms: bool = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


##
# MDP: Events (Domain Randomisation)
##


@configclass
class EventCfg:
    """Domain randomisation events."""

    # Physics material randomisation (friction, restitution)
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.2, 0.8),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
        },
    )

    # Base mass randomisation [-1, +3 kg]
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # Base CoM offset randomisation
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # Reset base pose (randomly on terrain)
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # Reset joints near default
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # External push during episode
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )


##
# MDP: Rewards
##


@configclass
class RewardsCfg:
    """Reward suite mirroring go2_rl_robotlab with 2× tracking weights.

    Velocity tracking uses **fixed sigma** (std=0.5) as in go2_rl_robotlab,
    not dynamic sigma.
    """

    # ---- Velocity tracking (positive) ----
    track_lin_vel_xy = RewTerm(
        func=track_lin_vel_xy_exp,
        weight=4.0,  # 2× go2_rl_gym baseline
        params={"std": 0.5, "command_name": "base_velocity"},
    )
    track_ang_vel_z = RewTerm(
        func=track_ang_vel_z_exp,
        weight=2.0,  # 2× go2_rl_gym baseline
        params={"std": 0.5, "command_name": "base_velocity"},
    )

    # ---- Base stability ----
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={
            "target_height": GO2_BASE_HEIGHT_TARGET,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # ---- Joint motion quality ----
    # Low weight: IsaacLab physics-step resolution makes raw acc noisy
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": _JOINT_ASSET_CFG},
    )
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-4,
        params={"asset_cfg": _JOINT_ASSET_CFG},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # ---- go2_rl_robotlab addition: L1 joint position deviation ----
    joint_pos_penalty_l1 = RewTerm(
        func=joint_pos_penalty_l1,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": _JOINT_ASSET_CFG,
            "stand_still_scale": 2.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )

    # ---- Energy ----
    joint_power = RewTerm(
        func=joint_power,
        weight=-2.0e-5,
        params={"asset_cfg": _JOINT_ASSET_CFG},
    )

    # ---- Stance ----
    stand_still = RewTerm(
        func=stand_still,
        weight=-1.0,
        params={"command_name": "base_velocity", "command_threshold": 0.06},
    )

    # ---- Feet ----
    feet_air_time = RewTerm(
        func=locomotion_mdp.feet_air_time,
        weight=0.5,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=locomotion_mdp.undesired_contacts,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "threshold": 1.0,
        },
    )


##
# MDP: Terminations
##


@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=locomotion_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )


##
# MDP: Curriculum
##


@configclass
class CurriculumCfg:
    """Terrain curriculum: advance when tracking rewards are high."""

    terrain_levels = CurrTerm(
        func=locomotion_mdp.terrain_levels_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


##
# Full environment config
##


@configclass
class Go2EnvCfg(ManagerBasedRLEnvCfg):
    """Go2 locomotion environment with deployable observation history.

    Designed for a future MoE-CTS Phase 2 extension:
    - ``policy`` and ``critic`` obs groups are separate.
    - Reward suite matches go2_rl_robotlab.
    """

    # --- Scene ---
    scene: Go2SceneCfg = Go2SceneCfg(num_envs=4096, env_spacing=2.5)

    # --- MDP ---
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Apply simulator and engine settings."""
        super().__post_init__()

        # Environment decimation: policy runs at 50 Hz (physics 200 Hz → decimation=4)
        self.decimation = 4

        # Simulator settings
        self.sim.dt = 0.005  # 200 Hz physics
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            enable_external_forces_every_iteration=True,
            gpu_max_rigid_patch_count=10 * 2**15,
        )

        # Episode length: 20 s → 1000 steps at 50 Hz
        self.episode_length_s = 20.0

        # Viewer
        self.viewer.eye = (10.0, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


@configclass
class Go2PlayEnvCfg(Go2EnvCfg):
    """Small playback environment for qualitative evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.debug_vis = True
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class Go2EvalEnvCfg(Go2EnvCfg):
    """Deterministic evaluation environment (no randomisation, no noise)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        self.observations.policy.enable_corruption = False
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.push_robot = None
        self.commands.base_velocity.resampling_time_range = (1e9, 1e9)
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
