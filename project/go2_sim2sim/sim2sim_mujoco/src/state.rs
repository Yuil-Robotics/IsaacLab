// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

pub const ACTION_DIM: usize = 12;
pub const OBSERVATION_DIM: usize = 450;
pub const FOOT_COUNT: usize = 4;
pub const GRAPH_HISTORY_LEN: usize = 150;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrailMode {
    BodyRelative,
    WorldFrame,
}

pub struct SimState {
    pub action: [f64; ACTION_DIM],
    pub action_version: u64,
    pub completed_action_version: u64,
    pub observation: [f64; OBSERVATION_DIM],
    pub command: [f64; 3],
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
    pub impulse_history: [VecDeque<f32>; FOOT_COUNT],
    pub torque_history: [VecDeque<f32>; ACTION_DIM],
    pub trail_mode: TrailMode,
    pub trail_length: usize,
    pub episode_step: u64,
    pub episode_generation: u64,
    pub fall_count: u64,
    pub reset_requested: bool,
    pub external_force_magnitude: f64,
    pub external_force_duration: f64,
    pub external_force_request: [f64; 3],
    pub external_force_request_version: u64,
    pub external_force_active: [f64; 3],
    pub external_force_steps_remaining: usize,
    pub shutdown_requested: bool,
}

impl SimState {
    pub fn new(command: [f64; 3]) -> Self {
        Self {
            action: [0.0; ACTION_DIM],
            action_version: 0,
            completed_action_version: 0,
            observation: [0.0; OBSERVATION_DIM],
            command,
            base_position: [0.0, 0.0, 0.315],
            base_height: 0.313,
            base_velocity: [0.0; 3],
            action_rms: 0.0,
            foot_positions: [[0.0; 3]; FOOT_COUNT],
            foot_heights: [0.0; FOOT_COUNT],
            foot_contacts: [false; FOOT_COUNT],
            foot_cycle_periods: [0.0; FOOT_COUNT],
            contact_impulses: [0.0; FOOT_COUNT],
            touchdown_impulses: [0.0; FOOT_COUNT],
            motor_torques: [0.0; ACTION_DIM],
            base_yaw: 0.0,
            impulse_history: std::array::from_fn(|_| VecDeque::with_capacity(GRAPH_HISTORY_LEN)),
            torque_history: std::array::from_fn(|_| VecDeque::with_capacity(GRAPH_HISTORY_LEN)),
            trail_mode: TrailMode::BodyRelative,
            trail_length: 60,
            episode_step: 0,
            episode_generation: 0,
            fall_count: 0,
            reset_requested: false,
            external_force_magnitude: 80.0,
            external_force_duration: 0.20,
            external_force_request: [0.0; 3],
            external_force_request_version: 0,
            external_force_active: [0.0; 3],
            external_force_steps_remaining: 0,
            shutdown_requested: false,
        }
    }
}

pub type SharedSimState = Arc<(Mutex<SimState>, Condvar)>;
