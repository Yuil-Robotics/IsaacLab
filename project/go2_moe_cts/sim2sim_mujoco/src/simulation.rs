// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

use std::path::Path;
use std::sync::Arc;

use mujoco_rs::prelude::*;

use crate::state::{ACTION_DIM, FOOT_COUNT, OBSERVATION_DIM};

pub const PHYSICS_DT: f64 = 0.005;
pub const DECIMATION: usize = 4;
pub const POLICY_DT: f64 = PHYSICS_DT * DECIMATION as f64;
pub const STEP_OBSERVATION_DIM: usize = 45;
pub const HISTORY_LENGTH: usize = 10;

const ACTION_CLIP: f64 = 100.0;
const ACTION_SCALE: f64 = 0.25;
const FALL_HEIGHT: f64 = 0.18;
const RESET_HEIGHT: f64 = 0.325;
pub const BASE_BOTTOM_OFFSET_Z: f64 = -0.01205;

// This is the action and joint-observation order from YuilDogRobotLabEnvCfg.
pub const JOINT_NAMES: [&str; ACTION_DIM] = [
    "FL_hip_roll",
    "FL_hip_pitch",
    "FL_knee_pitch",
    "FR_hip_roll",
    "FR_hip_pitch",
    "FR_knee_pitch",
    "RL_hip_roll",
    "RL_hip_pitch",
    "RL_knee_pitch",
    "RR_hip_roll",
    "RR_hip_pitch",
    "RR_knee_pitch",
];
pub const FOOT_NAMES: [&str; FOOT_COUNT] = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"];

const DEFAULT_POSITION: [f64; ACTION_DIM] = [
    -0.05, -0.7, 0.7, 0.05, 0.7, -0.7, 0.05, -0.7, 0.7, -0.05, 0.7, -0.7,
];
const POSITION_LOWER: [f64; ACTION_DIM] = [
    -0.38, -1.8, 0.05, -0.46, -1.2, -1.57, -0.46, -1.8, 0.05, -0.38, -1.2, -1.57,
];
const POSITION_UPPER: [f64; ACTION_DIM] = [
    0.46, 1.2, 1.57, 0.38, 1.8, -0.05, 0.38, 1.2, 1.57, 0.46, 1.8, -0.05,
];
const STIFFNESS: [f64; ACTION_DIM] = [
    60.0, 85.0, 80.0, 60.0, 85.0, 80.0, 60.0, 85.0, 80.0, 60.0, 85.0, 80.0,
];
const DAMPING: [f64; ACTION_DIM] = [3.5, 2.2, 2.2, 3.5, 2.2, 2.2, 3.5, 2.2, 2.2, 3.5, 2.2, 2.2];
const CONTINUOUS_EFFORT: [f64; ACTION_DIM] = [
    30.0, 60.0, 60.0, 30.0, 60.0, 60.0, 30.0, 60.0, 60.0, 30.0, 60.0, 60.0,
];
const SATURATION_EFFORT: [f64; ACTION_DIM] = [
    60.0, 115.0, 115.0, 60.0, 115.0, 115.0, 60.0, 115.0, 115.0, 60.0, 115.0, 115.0,
];
const VELOCITY_LIMIT: [f64; ACTION_DIM] = [
    10.0, 20.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0, 20.0,
];

const OBSERVATION_TERMS: [(usize, usize); 6] =
    [(0, 3), (3, 3), (6, 3), (9, 12), (21, 12), (33, 12)];
const _: [(); OBSERVATION_DIM] = [(); STEP_OBSERVATION_DIM * HISTORY_LENGTH];

pub struct StepResult {
    pub observation: [f64; OBSERVATION_DIM],
    pub base_position: [f64; 3],
    pub base_height: f64,
    pub base_velocity: [f64; 3],
    pub action_rms: f64,
    pub foot_positions: [[f64; 3]; FOOT_COUNT],
    pub foot_heights: [f64; FOOT_COUNT],
    pub foot_contacts: [bool; FOOT_COUNT],
    pub foot_cycle_periods: [f64; FOOT_COUNT],
    pub contact_impulses: [f64; FOOT_COUNT],
    pub touchdown_impulses: [f64; FOOT_COUNT],
    pub motor_torques: [f64; ACTION_DIM],
    pub base_yaw: f64,
    pub external_force: [f64; 3],
    pub episode_step: u64,
    pub automatic_reset: bool,
    pub fell: bool,
}

pub struct QuadrupedSimulation {
    pub data: MjData<Arc<MjModel>>,
    joint_qpos: [usize; ACTION_DIM],
    joint_dof: [usize; ACTION_DIM],
    foot_body_ids: [usize; FOOT_COUNT],
    foot_geom_ids: [usize; FOOT_COUNT],
    ground_geom_id: usize,
    pub base_body_id: usize,
    free_qpos: usize,
    command: [f64; 3],
    latest_action: [f64; ACTION_DIM],
    latest_motor_torques: [f64; ACTION_DIM],
    foot_contacts: [bool; FOOT_COUNT],
    last_touchdown_times: [Option<f64>; FOOT_COUNT],
    foot_cycle_periods: [f64; FOOT_COUNT],
    contact_impulses: [f64; FOOT_COUNT],
    touchdown_impulses: [f64; FOOT_COUNT],
    external_force: [f64; 3],
    history: [[f64; STEP_OBSERVATION_DIM]; HISTORY_LENGTH],
    episode_step: u64,
}

pub fn load_model(path: &Path) -> MjModel {
    let model = MjModel::from_xml(path)
        .unwrap_or_else(|error| panic!("failed to load {}: {error}", path.display()));
    let timestep_error = (model.opt().timestep - PHYSICS_DT).abs();
    assert!(
        timestep_error < 1.0e-12,
        "model timestep must be {PHYSICS_DT} s, got {} s",
        model.opt().timestep
    );
    model
}

impl QuadrupedSimulation {
    pub fn new(model: Arc<MjModel>, command: [f64; 3]) -> Self {
        assert!(command.iter().all(|value| value.is_finite()));
        let joint_ids = JOINT_NAMES.map(|name| {
            model
                .name_to_id(MjtObj::mjOBJ_JOINT, name)
                .unwrap_or_else(|| panic!("required policy joint '{name}' was not found"))
        });
        for (index, joint_id) in joint_ids.iter().copied().enumerate() {
            assert_eq!(
                model.jnt_type()[joint_id],
                MjtJoint::mjJNT_HINGE,
                "policy joint '{}' is not a hinge",
                JOINT_NAMES[index]
            );
            let model_range = model.jnt_range()[joint_id];
            assert!(
                POSITION_LOWER[index] >= model_range[0] && POSITION_UPPER[index] <= model_range[1],
                "safe range for '{}' exceeds MJCF limits",
                JOINT_NAMES[index]
            );
        }

        let joint_qpos = joint_ids.map(|id| model.jnt_qposadr()[id] as usize);
        let joint_dof = joint_ids.map(|id| model.jnt_dofadr()[id] as usize);
        let foot_body_ids = FOOT_NAMES.map(|name| {
            model
                .name_to_id(MjtObj::mjOBJ_BODY, name)
                .unwrap_or_else(|| panic!("required foot body '{name}' was not found"))
        });
        let ground_geom_id = model
            .name_to_id(MjtObj::mjOBJ_GEOM, "ground")
            .expect("ground geom 'ground' was not found");
        let foot_geom_ids = foot_body_ids.map(|body_id| {
            model
                .geom_bodyid()
                .iter()
                .enumerate()
                .find_map(|(geom_id, &geom_body_id)| {
                    (geom_body_id as usize == body_id && model.geom_contype()[geom_id] != 0)
                        .then_some(geom_id)
                })
                .expect("a collidable foot geom was not found")
        });
        for &geom_id in &foot_geom_ids {
            assert_eq!(
                model.geom_type()[geom_id],
                MjtGeom::mjGEOM_SPHERE,
                "foot ground-clearance telemetry requires spherical foot geoms"
            );
        }
        let free_joint_id = model
            .name_to_id(MjtObj::mjOBJ_JOINT, "base_free_joint")
            .expect("floating-base joint 'base_free_joint' was not found");
        let free_qpos = model.jnt_qposadr()[free_joint_id] as usize;
        let base_body_id = model
            .name_to_id(MjtObj::mjOBJ_BODY, "base_link")
            .expect("base body 'base_link' was not found");
        let data = MjData::new(model);
        let mut simulation = Self {
            data,
            joint_qpos,
            joint_dof,
            foot_body_ids,
            foot_geom_ids,
            ground_geom_id,
            base_body_id,
            free_qpos,
            command,
            latest_action: [0.0; ACTION_DIM],
            latest_motor_torques: [0.0; ACTION_DIM],
            foot_contacts: [false; FOOT_COUNT],
            last_touchdown_times: [None; FOOT_COUNT],
            foot_cycle_periods: [0.0; FOOT_COUNT],
            contact_impulses: [0.0; FOOT_COUNT],
            touchdown_impulses: [0.0; FOOT_COUNT],
            external_force: [0.0; 3],
            history: [[0.0; STEP_OBSERVATION_DIM]; HISTORY_LENGTH],
            episode_step: 0,
        };
        simulation.set_command(command);
        simulation.reset_episode();
        simulation
    }

    pub fn set_command(&mut self, command: [f64; 3]) {
        assert!(command.iter().all(|value| value.is_finite()));
        self.command = [
            command[0].clamp(-1.0, 1.0),
            command[1].clamp(-0.5, 0.5),
            command[2].clamp(-1.0, 1.0),
        ];
    }

    pub fn set_external_force(&mut self, force: [f64; 3]) {
        assert!(force.iter().all(|value| value.is_finite()));
        self.external_force = force;
    }

    pub fn reset_episode(&mut self) {
        self.data.reset();
        {
            let qpos = self.data.qpos_mut();
            qpos[self.free_qpos..self.free_qpos + 3].copy_from_slice(&[0.0, 0.0, RESET_HEIGHT]);
            qpos[self.free_qpos + 3..self.free_qpos + 7].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            for (index, &address) in self.joint_qpos.iter().enumerate() {
                qpos[address] = DEFAULT_POSITION[index];
            }
        }
        self.data.qvel_mut().fill(0.0);
        self.data.qfrc_applied_mut().fill(0.0);
        self.data.xfrc_applied_mut().fill([0.0; 6]);
        self.latest_action.fill(0.0);
        self.latest_motor_torques.fill(0.0);
        self.foot_contacts.fill(false);
        self.last_touchdown_times.fill(None);
        self.foot_cycle_periods.fill(0.0);
        self.contact_impulses.fill(0.0);
        self.touchdown_impulses.fill(0.0);
        self.external_force.fill(0.0);
        self.episode_step = 0;
        self.data.forward();
        let observation = self.build_observation_step();
        self.history.fill(observation);
    }

    pub fn policy_step(&mut self, raw_action: [f64; ACTION_DIM]) -> StepResult {
        let clipped_action = raw_action.map(|value| value.clamp(-ACTION_CLIP, ACTION_CLIP));
        let target = std::array::from_fn(|index| {
            (DEFAULT_POSITION[index] + ACTION_SCALE * clipped_action[index])
                .clamp(POSITION_LOWER[index], POSITION_UPPER[index])
        });

        self.contact_impulses.fill(0.0);
        let mut touchdown_started = [false; FOOT_COUNT];
        for _ in 0..DECIMATION {
            let torque = self.compute_motor_torque(target);
            self.latest_motor_torques = torque;
            let applied = self.data.qfrc_applied_mut();
            applied.fill(0.0);
            for (index, &address) in self.joint_dof.iter().enumerate() {
                applied[address] = torque[index];
            }
            let applied = self.data.xfrc_applied_mut();
            applied.fill([0.0; 6]);
            applied[self.base_body_id][0..3].copy_from_slice(&self.external_force);
            self.data.step();

            let (contacts, impulses) = self.measure_ground_contacts();
            for foot_index in 0..FOOT_COUNT {
                let touchdown = !self.foot_contacts[foot_index] && contacts[foot_index];
                if touchdown {
                    let current_time = self.data.time();
                    if let Some(prev_time) = self.last_touchdown_times[foot_index] {
                        let dt = current_time - prev_time;
                        // Minimum cycle duration (0.15 s) rejects contact bounce/chatter within stance
                        if dt >= 0.15 {
                            self.foot_cycle_periods[foot_index] = dt;
                            self.last_touchdown_times[foot_index] = Some(current_time);
                        }
                    } else {
                        self.last_touchdown_times[foot_index] = Some(current_time);
                    }
                }
                touchdown_started[foot_index] |= touchdown;
                self.contact_impulses[foot_index] += impulses[foot_index];
            }
            self.foot_contacts = contacts;
        }
        for (foot_index, started) in touchdown_started.iter().copied().enumerate() {
            if started {
                self.touchdown_impulses[foot_index] = self.contact_impulses[foot_index];
            }
        }
        assert!(self.data.qpos().iter().all(|value| value.is_finite()));
        assert!(self.data.qvel().iter().all(|value| value.is_finite()));

        self.latest_action = clipped_action;
        self.episode_step += 1;
        let fell = self.base_height() < FALL_HEIGHT;
        let automatic_reset = fell;
        if automatic_reset {
            self.reset_episode();
        } else {
            self.history.rotate_left(1);
            self.history[HISTORY_LENGTH - 1] = self.build_observation_step();
        }
        let mut result = self.current_result();
        result.automatic_reset = automatic_reset;
        result.fell = fell;
        result
    }

    pub fn current_result(&self) -> StepResult {
        StepResult {
            observation: self.flatten_history(),
            base_position: self.base_position(),
            base_height: self.base_height(),
            base_velocity: self.base_velocity(),
            action_rms: (self
                .latest_action
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                / ACTION_DIM as f64)
                .sqrt(),
            foot_positions: self.foot_positions(),
            foot_heights: self.foot_heights(),
            foot_contacts: self.foot_contacts,
            foot_cycle_periods: self.foot_cycle_periods,
            contact_impulses: self.contact_impulses,
            touchdown_impulses: self.touchdown_impulses,
            motor_torques: self.latest_motor_torques,
            base_yaw: self.base_yaw(),
            external_force: self.external_force,
            episode_step: self.episode_step,
            automatic_reset: false,
            fell: false,
        }
    }

    fn compute_motor_torque(&self, target: [f64; ACTION_DIM]) -> [f64; ACTION_DIM] {
        let qpos = self.data.qpos();
        let qvel = self.data.qvel();
        std::array::from_fn(|index| {
            let position = qpos[self.joint_qpos[index]];
            let velocity = qvel[self.joint_dof[index]];
            let computed =
                STIFFNESS[index] * (target[index] - position) - DAMPING[index] * velocity;
            let bounded_velocity = velocity.clamp(
                -VELOCITY_LIMIT[index]
                    * (1.0 + CONTINUOUS_EFFORT[index] / SATURATION_EFFORT[index]),
                VELOCITY_LIMIT[index] * (1.0 + CONTINUOUS_EFFORT[index] / SATURATION_EFFORT[index]),
            );
            let maximum = (SATURATION_EFFORT[index]
                * (1.0 - bounded_velocity / VELOCITY_LIMIT[index]))
                .min(CONTINUOUS_EFFORT[index]);
            let minimum = (SATURATION_EFFORT[index]
                * (-1.0 - bounded_velocity / VELOCITY_LIMIT[index]))
                .max(-CONTINUOUS_EFFORT[index]);
            computed.clamp(minimum, maximum)
        })
    }

    fn build_observation_step(&self) -> [f64; STEP_OBSERVATION_DIM] {
        let mut observation = [0.0; STEP_OBSERVATION_DIM];
        let world_velocity =
            self.data
                .object_velocity(MjtObj::mjOBJ_BODY, self.base_body_id, false);
        let quaternion = self.data.xquat()[self.base_body_id];
        let angular_velocity = rotate_world_to_body(
            quaternion,
            [world_velocity[0], world_velocity[1], world_velocity[2]],
        );
        observation[0..3].copy_from_slice(&angular_velocity);

        observation[3..6].copy_from_slice(&rotate_world_to_body(quaternion, [0.0, 0.0, -1.0]));
        observation[6..9].copy_from_slice(&self.command);
        for (index, &address) in self.joint_qpos.iter().enumerate() {
            observation[9 + index] = self.data.qpos()[address] - DEFAULT_POSITION[index];
        }
        for (index, &address) in self.joint_dof.iter().enumerate() {
            observation[21 + index] = self.data.qvel()[address];
        }
        observation[33..45].copy_from_slice(&self.latest_action);
        // IsaacLab clips each raw observation before applying the CTS scales.
        for value in &mut observation {
            *value = value.clamp(-100.0, 100.0);
        }
        for value in &mut observation[0..3] {
            *value *= 0.25;
        }
        for value in &mut observation[21..33] {
            *value *= 0.05;
        }
        observation
    }

    fn flatten_history(&self) -> [f64; OBSERVATION_DIM] {
        let mut flattened = [0.0; OBSERVATION_DIM];
        let mut cursor = 0;
        for (start, size) in OBSERVATION_TERMS {
            for step in &self.history {
                flattened[cursor..cursor + size].copy_from_slice(&step[start..start + size]);
                cursor += size;
            }
        }
        debug_assert_eq!(cursor, OBSERVATION_DIM);
        flattened
    }

    fn base_position(&self) -> [f64; 3] {
        self.data.xpos()[self.base_body_id]
    }

    /// Robot base-bottom clearance above ground [m], matching IsaacLab's height_scanner reference.
    pub fn base_height(&self) -> f64 {
        let ground_height = self.data.geom_xpos()[self.ground_geom_id][2];
        let base_xpos = self.data.xpos()[self.base_body_id];
        let body_xmat = self.data.xmat()[self.base_body_id];
        (base_xpos[2] + body_xmat[8] * BASE_BOTTOM_OFFSET_Z) - ground_height
    }

    /// Robot base yaw heading in world coordinates [rad] (-pi to pi).
    pub fn base_yaw(&self) -> f64 {
        let body_xmat = self.data.xmat()[self.base_body_id];
        body_xmat[3].atan2(body_xmat[0])
    }

    #[allow(dead_code)]
    pub fn foot_cycle_periods(&self) -> [f64; FOOT_COUNT] {
        self.foot_cycle_periods
    }

    pub fn foot_positions(&self) -> [[f64; 3]; FOOT_COUNT] {
        self.foot_body_ids.map(|body_id| self.data.xpos()[body_id])
    }

    /// Foot positions expressed in base_link body frame [m] (X: forward, Y: left, Z: up).
    pub fn foot_relative_positions(&self) -> [[f64; 3]; FOOT_COUNT] {
        let base_xpos = self.data.xpos()[self.base_body_id];
        let body_xmat = self.data.xmat()[self.base_body_id];
        self.foot_body_ids.map(|body_id| {
            let foot_xpos = self.data.xpos()[body_id];
            let dx = foot_xpos[0] - base_xpos[0];
            let dy = foot_xpos[1] - base_xpos[1];
            let dz = foot_xpos[2] - base_xpos[2];
            [
                body_xmat[0] * dx + body_xmat[3] * dy + body_xmat[6] * dz,
                body_xmat[1] * dx + body_xmat[4] * dy + body_xmat[7] * dz,
                body_xmat[2] * dx + body_xmat[5] * dy + body_xmat[8] * dz,
            ]
        })
    }

    fn foot_heights(&self) -> [f64; FOOT_COUNT] {
        let ground_height = self.data.geom_xpos()[self.ground_geom_id][2];
        self.foot_geom_ids.map(|geom_id| {
            self.data.geom_xpos()[geom_id][2]
                - self.data.model().geom_size()[geom_id][0]
                - ground_height
        })
    }

    fn measure_ground_contacts(&self) -> ([bool; FOOT_COUNT], [f64; FOOT_COUNT]) {
        let mut contacts = [false; FOOT_COUNT];
        let mut impulses = [0.0; FOOT_COUNT];
        let geom_body_ids = self.data.model().geom_bodyid();
        for contact_id in 0..self.data.ncon() as usize {
            let contact = &self.data.contact()[contact_id];
            let (ground_geom, other_geom) = if contact.geom1 == self.ground_geom_id as i32 {
                (contact.geom1, contact.geom2)
            } else if contact.geom2 == self.ground_geom_id as i32 {
                (contact.geom2, contact.geom1)
            } else {
                continue;
            };
            debug_assert_eq!(ground_geom, self.ground_geom_id as i32);
            if other_geom < 0 {
                continue;
            }
            let other_body = geom_body_ids[other_geom as usize] as usize;
            let Some(foot_index) = self
                .foot_body_ids
                .iter()
                .position(|&body_id| body_id == other_body)
            else {
                continue;
            };
            contacts[foot_index] = true;
            impulses[foot_index] += self.data.contact_force(contact_id)[0].abs() * PHYSICS_DT;
        }
        (contacts, impulses)
    }

    fn base_velocity(&self) -> [f64; 3] {
        let world_velocity =
            self.data
                .object_velocity(MjtObj::mjOBJ_BODY, self.base_body_id, false);
        let quaternion = self.data.xquat()[self.base_body_id];
        rotate_world_to_body(
            quaternion,
            [world_velocity[3], world_velocity[4], world_velocity[5]],
        )
    }
}

fn rotate_world_to_body([w, x, y, z]: [f64; 4], vector: [f64; 3]) -> [f64; 3] {
    let rotation = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    std::array::from_fn(|column| {
        rotation[0][column] * vector[0]
            + rotation[1][column] * vector[1]
            + rotation[2][column] * vector[2]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_quaternion_preserves_world_vector() {
        assert_eq!(
            rotate_world_to_body([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
            [0.0, 0.0, -1.0]
        );
    }

    #[test]
    fn observation_contract_has_expected_size() {
        assert_eq!(STEP_OBSERVATION_DIM * HISTORY_LENGTH, OBSERVATION_DIM);
        assert_eq!(
            OBSERVATION_TERMS.iter().map(|term| term.1).sum::<usize>(),
            45
        );
    }

    #[test]
    fn cts_observation_scales_and_command_limits() {
        let model = Arc::new(load_model(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("model/MJCF/yuil_dog.xml"),
        ));
        let mut simulation = QuadrupedSimulation::new(model, [2.0, -2.0, 3.0]);
        assert_eq!(simulation.command, [1.0, -0.5, 1.0]);
        simulation.data.qvel_mut()[3..6].copy_from_slice(&[4.0, -8.0, 12.0]);
        for address in simulation.joint_dof {
            simulation.data.qvel_mut()[address] = 200.0;
        }
        simulation.latest_action.fill(8.0);
        simulation.data.forward();
        let observation = simulation.build_observation_step();
        assert_eq!(&observation[0..3], &[1.0, -2.0, 3.0]);
        assert_eq!(&observation[21..33], &[5.0; 12]);
        assert_eq!(&observation[33..45], &[8.0; 12]);
        simulation.reset_episode();
        simulation.policy_step([8.0; ACTION_DIM]);
        assert_eq!(simulation.latest_action, [8.0; ACTION_DIM]);
    }

    #[test]
    fn generated_model_satisfies_runtime_contract() {
        let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("model")
            .join("MJCF")
            .join("yuil_dog.xml");
        let model = Arc::new(load_model(&model_path));
        let simulation = QuadrupedSimulation::new(model, [0.5, 0.0, 0.0]);
        let result = simulation.current_result();
        assert!(result.observation.iter().all(|value| value.is_finite()));
        assert!(
            simulation
                .foot_positions()
                .iter()
                .flatten()
                .all(|value| value.is_finite())
        );
        assert!(result.foot_heights.iter().all(|value| value.is_finite()));
        assert_eq!(result.motor_torques, [0.0; ACTION_DIM]);
        assert_eq!(result.contact_impulses, [0.0; FOOT_COUNT]);
        assert!(result.base_yaw.abs() < 1e-6);
        assert_eq!(result.base_position, [0.0, 0.0, RESET_HEIGHT]);
        let expected_base_height = RESET_HEIGHT + BASE_BOTTOM_OFFSET_Z;
        assert!((result.base_height - expected_base_height).abs() < 1e-6);
        assert_eq!(result.foot_cycle_periods, [0.0; FOOT_COUNT]);
        for gravity in result.observation[30..60].chunks_exact(3) {
            assert_eq!(gravity, [0.0, 0.0, -1.0]);
        }
    }

    #[test]
    fn policy_step_reports_applied_force_and_finite_telemetry() {
        let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("model")
            .join("MJCF")
            .join("yuil_dog.xml");
        let model = Arc::new(load_model(&model_path));
        let mut simulation = QuadrupedSimulation::new(model, [0.0; 3]);
        let force = [25.0, -10.0, 5.0];
        simulation.set_external_force(force);

        let result = simulation.policy_step([0.0; ACTION_DIM]);

        assert_eq!(result.external_force, force);
        assert!(result.motor_torques.iter().all(|value| value.is_finite()));
        assert!(result.foot_heights.iter().all(|value| value.is_finite()));
        assert!(result.contact_impulses.iter().all(|value| *value >= 0.0));
        assert!(result.touchdown_impulses.iter().all(|value| *value >= 0.0));
    }

    #[test]
    fn episode_does_not_reset_at_the_previous_time_limit() {
        let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("model")
            .join("MJCF")
            .join("yuil_dog.xml");
        let model = Arc::new(load_model(&model_path));
        let mut simulation = QuadrupedSimulation::new(model, [0.0; 3]);
        simulation.episode_step = 999;

        let result = simulation.policy_step([0.0; ACTION_DIM]);

        assert!(!result.automatic_reset);
        assert_eq!(result.episode_step, 1_000);
    }

    #[test]
    fn base_height_and_foot_cycle_periods_telemetry() {
        let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("model")
            .join("MJCF")
            .join("yuil_dog.xml");
        let model = Arc::new(load_model(&model_path));
        let mut simulation = QuadrupedSimulation::new(model, [0.0; 3]);

        let initial_base_height = simulation.base_height();
        assert!((initial_base_height - (RESET_HEIGHT + BASE_BOTTOM_OFFSET_Z)).abs() < 1e-6);
        assert_eq!(simulation.foot_cycle_periods(), [0.0; FOOT_COUNT]);

        let result = simulation.policy_step([0.0; ACTION_DIM]);
        assert!(result.base_height > 0.20 && result.base_height < 0.40);
        assert_eq!(result.foot_cycle_periods, simulation.foot_cycle_periods());
    }
}
