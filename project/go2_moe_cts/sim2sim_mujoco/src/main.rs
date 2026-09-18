// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

mod server;
mod simulation;
mod state;

use std::collections::VecDeque;
use std::env;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use mujoco_rs::prelude::{MjtCamera, MjtGeom, MjtObj};
use mujoco_rs::viewer::MjViewer;

use simulation::{FOOT_NAMES, JOINT_NAMES, POLICY_DT, QuadrupedSimulation, load_model};
use state::{
    ACTION_DIM, FOOT_COUNT, GRAPH_HISTORY_LEN, OBSERVATION_DIM, SharedSimState, SimState, TrailMode,
};

const DEFAULT_TCP_ADDR: &str = "127.0.0.1:7001";
const FOOT_TRAIL_LENGTH: usize = 250;
const FOOT_TRAIL_LINE_WIDTH: f64 = 0.006;
const FOOT_TRAIL_COLORS: [[f32; 4]; FOOT_COUNT] = [
    [1.0, 0.15, 0.15, 1.0],
    [0.15, 1.0, 0.15, 1.0],
    [0.15, 0.45, 1.0, 1.0],
    [1.0, 0.85, 0.1, 1.0],
];

#[derive(Clone, Copy, Debug)]
enum CameraAction {
    TrackRobot,
    Free,
    SetFollowRotation(bool),
    PresetIsometric,
    PresetSide,
    PresetRightSide,
    PresetRear,
    PresetFront,
    PresetTop,
    RigidChase,
    RigidSide,
}

#[derive(Debug)]
struct Arguments {
    model: PathBuf,
    address: String,
    command: [f64; 3],
}

fn default_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("model")
        .join("MJCF")
        .join("yuil_dog.xml")
}

fn parse_arguments() -> Arguments {
    let mut result = Arguments {
        model: default_model_path(),
        address: env::var("MUJOCO_SERVER_ADDR").unwrap_or_else(|_| DEFAULT_TCP_ADDR.to_owned()),
        command: [0.5, 0.0, 0.0],
    };
    let values: Vec<String> = env::args().skip(1).collect();
    let mut index = 0;
    while index < values.len() {
        match values[index].as_str() {
            "--model" => {
                index += 1;
                result.model = values
                    .get(index)
                    .map(PathBuf::from)
                    .expect("--model requires a path");
            }
            "--address" => {
                index += 1;
                result.address = values
                    .get(index)
                    .expect("--address requires HOST:PORT")
                    .clone();
            }
            "--command" => {
                for component in &mut result.command {
                    index += 1;
                    *component = values
                        .get(index)
                        .expect("--command requires VX VY WZ")
                        .parse()
                        .expect("command components must be finite numbers");
                    assert!(component.is_finite(), "command components must be finite");
                }
            }
            "--help" | "-h" => {
                println!(
                    "Yuil Dog Rust MuJoCo task\n\n  --model PATH\n  --address HOST:PORT\n  --command VX VY WZ"
                );
                std::process::exit(0);
            }
            unknown => panic!("unknown argument '{unknown}'; use --help"),
        }
        index += 1;
    }
    result.command[0] = result.command[0].clamp(-1.0, 1.0);
    result.command[1] = result.command[1].clamp(-0.5, 0.5);
    result.command[2] = result.command[2].clamp(-1.0, 1.0);
    result
}

fn draw_graph(
    ui: &mut egui::Ui,
    title: &str,
    series: &[(&str, egui::Color32, &VecDeque<f32>)],
    unit: &str,
    symmetric_zero: bool,
) {
    ui.horizontal_wrapped(|ui| {
        ui.strong(title);
        for &(name, color, data) in series {
            let latest = data.back().copied().unwrap_or(0.0);
            ui.colored_label(color, format!("{name}: {latest:+.2} {unit}"));
        }
    });

    let desired_size = egui::Vec2::new(ui.available_width().max(200.0), 90.0);
    let (response, painter) = ui.allocate_painter(desired_size, egui::Sense::hover());
    let rect = response.rect;

    // Background & border
    painter.rect_filled(rect, 4.0_f32, egui::Color32::from_rgb(18, 22, 28));
    painter.rect_stroke(
        rect,
        4.0_f32,
        egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(45, 55, 70)),
        egui::StrokeKind::Inside,
    );

    // Dynamic scale across all series
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for &(_, _, data) in series {
        for &val in data {
            if val.is_finite() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }
    }
    if min_val > max_val {
        min_val = 0.0;
        max_val = 1.0;
    }

    let (plot_min, plot_max) = if symmetric_zero {
        let max_abs = min_val.abs().max(max_val.abs()).max(10.0) * 1.15;
        (-max_abs, max_abs)
    } else {
        let span = (max_val - min_val).max(0.1);
        (0.0f32.min(min_val), max_val + span * 0.15)
    };

    let range = (plot_max - plot_min).max(1e-4);

    // Zero reference line
    if plot_min < 0.0 && plot_max > 0.0 {
        let zero_norm = (0.0 - plot_min) / range;
        let zero_y = rect.bottom() - zero_norm * rect.height();
        painter.line_segment(
            [
                egui::pos2(rect.left(), zero_y),
                egui::pos2(rect.right(), zero_y),
            ],
            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(55, 65, 80)),
        );
    }

    // Min / max labels
    painter.text(
        egui::pos2(rect.left() + 4.0, rect.top() + 2.0),
        egui::Align2::LEFT_TOP,
        format!("{plot_max:.1} {unit}"),
        egui::FontId::monospace(10.0),
        egui::Color32::from_rgb(130, 140, 155),
    );
    painter.text(
        egui::pos2(rect.left() + 4.0, rect.bottom() - 2.0),
        egui::Align2::LEFT_BOTTOM,
        format!("{plot_min:.1} {unit}"),
        egui::FontId::monospace(10.0),
        egui::Color32::from_rgb(130, 140, 155),
    );

    // Plot curves
    let max_len = series
        .iter()
        .map(|(_, _, d)| d.len())
        .max()
        .unwrap_or(0)
        .max(2);
    for &(_, color, data) in series {
        if data.len() < 2 {
            continue;
        }
        let offset = max_len.saturating_sub(data.len());
        let points: Vec<egui::Pos2> = data
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let x = rect.left() + ((i + offset) as f32 / (max_len - 1) as f32) * rect.width();
                let norm_y = ((v - plot_min) / range).clamp(0.0, 1.0);
                let y = rect.bottom() - norm_y * rect.height();
                egui::pos2(x, y)
            })
            .collect();
        for window in points.windows(2) {
            painter.line_segment([window[0], window[1]], egui::Stroke::new(1.5_f32, color));
        }
    }
}

fn update_shared_result(shared: &SharedSimState, result: &simulation::StepResult) {
    let (mutex, updated) = &**shared;
    let mut state = mutex.lock().unwrap();
    state.observation = result.observation;
    state.base_position = result.base_position;
    state.base_height = result.base_height;
    state.base_velocity = result.base_velocity;
    state.action_rms = result.action_rms;
    state.foot_positions = result.foot_positions;
    state.foot_heights = result.foot_heights;
    state.foot_contacts = result.foot_contacts;
    state.foot_cycle_periods = result.foot_cycle_periods;
    state.contact_impulses = result.contact_impulses;
    state.touchdown_impulses = result.touchdown_impulses;
    state.motor_torques = result.motor_torques;
    state.base_yaw = result.base_yaw;
    state.external_force_active = result.external_force;
    state.episode_step = result.episode_step;
    if result.automatic_reset {
        state.episode_generation += 1;
        for hist in &mut state.impulse_history {
            hist.clear();
        }
        for hist in &mut state.torque_history {
            hist.clear();
        }
    } else {
        for foot_index in 0..FOOT_COUNT {
            let hist = &mut state.impulse_history[foot_index];
            if hist.len() >= GRAPH_HISTORY_LEN {
                hist.pop_front();
            }
            hist.push_back(result.contact_impulses[foot_index] as f32);
        }
        for joint_index in 0..ACTION_DIM {
            let hist = &mut state.torque_history[joint_index];
            if hist.len() >= GRAPH_HISTORY_LEN {
                hist.pop_front();
            }
            hist.push_back(result.motor_torques[joint_index] as f32);
        }
    }
    if result.fell {
        state.fall_count += 1;
    }
    updated.notify_all();
}

fn main() {
    let arguments = parse_arguments();
    let model_file = arguments.model.canonicalize().unwrap_or_else(|error| {
        panic!(
            "failed to resolve model {}; run scripts/convert_urdf_to_mjcf.py first: {error}",
            arguments.model.display()
        )
    });
    let model = Arc::new(load_model(&model_file));
    let simulation = QuadrupedSimulation::new(model.clone(), arguments.command);
    let initial_result = simulation.current_result();
    let shared: SharedSimState =
        Arc::new((Mutex::new(SimState::new(arguments.command)), Condvar::new()));
    update_shared_result(&shared, &initial_result);

    println!("Yuil Dog Rust MuJoCo contract validated");
    println!("model={}", model_file.display());
    println!("joint_order={JOINT_NAMES:?}");
    println!(
        "command={:?} policy={OBSERVATION_DIM}-D -> {ACTION_DIM}-D dt={POLICY_DT:.3} s",
        arguments.command
    );

    std::thread::spawn({
        let address = arguments.address.clone();
        let shared = shared.clone();
        move || server::run_server(&address, shared)
    });

    let base_link_id = model
        .name_to_id(MjtObj::mjOBJ_BODY, "base_link")
        .or_else(|| model.name_to_id(MjtObj::mjOBJ_BODY, "base"))
        .expect("could not find base body for camera tracking");

    let camera_action = Arc::new(Mutex::new(None));
    let camera_follow_rotation = Arc::new(AtomicBool::new(true));

    let mut viewer = MjViewer::builder()
        .window_name("Yuil Dog MoE-CTS 450→12 Sim2Sim")
        .vsync(true)
        .warn_non_realtime(false)
        .track_body_id(base_link_id)
        .camera_distance(2.2)
        .camera_azimuth(135.0)
        .camera_elevation(-20.0)
        .max_user_geoms(FOOT_COUNT * (FOOT_TRAIL_LENGTH - 1))
        .build_passive(model.clone())
        .expect("failed to launch the Rust MuJoCo viewer");

    let mut selected_torque_tab = 0_usize;
    viewer.add_ui_callback_detached({
        let shared = shared.clone();
        let camera_action = camera_action.clone();
        let camera_follow_rotation = camera_follow_rotation.clone();
        move |context: &egui::Context| {
            egui::Window::new("Yuil Dog locomotion task")
                .vscroll(true)
                .show(context, |ui| {
                let (
                    position,
                    base_height,
                    velocity,
                    action_rms,
                    foot_positions,
                    foot_heights,
                    foot_contacts,
                    foot_cycle_periods,
                    contact_impulses,
                    touchdown_impulses,
                    motor_torques,
                    impulse_history,
                    torque_history,
                    trail_mode,
                    trail_length,
                    external_force_active,
                    external_force_steps_remaining,
                    mut force_magnitude,
                    mut force_duration,
                    episode_step,
                    generation,
                    falls,
                    mut command,
                ) = {
                    let (mutex, _) = &*shared;
                    let state = mutex.lock().unwrap();
                    (
                        state.base_position,
                        state.base_height,
                        state.base_velocity,
                        state.action_rms,
                        state.foot_positions,
                        state.foot_heights,
                        state.foot_contacts,
                        state.foot_cycle_periods,
                        state.contact_impulses,
                        state.touchdown_impulses,
                        state.motor_torques,
                        state.impulse_history.clone(),
                        state.torque_history.clone(),
                        state.trail_mode,
                        state.trail_length,
                        state.external_force_active,
                        state.external_force_steps_remaining,
                        state.external_force_magnitude,
                        state.external_force_duration,
                        state.episode_step,
                        state.episode_generation,
                        state.fall_count,
                        state.command,
                    )
                };

                egui::Grid::new("locomotion_status")
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label("Base position");
                        ui.label(format!(
                            "{:+.3}, {:+.3}, {:+.3} m",
                            position[0], position[1], position[2]
                        ));
                        ui.end_row();
                        ui.label("Base height (Isaac)");
                        ui.label(format!("{base_height:.3} m"));
                        ui.end_row();
                        ui.label("Body velocity");
                        ui.label(format!(
                            "{:+.3}, {:+.3}, {:+.3} m/s",
                            velocity[0], velocity[1], velocity[2]
                        ));
                        ui.end_row();
                        ui.label("Action RMS");
                        ui.label(format!("{action_rms:.4}"));
                        ui.end_row();
                        ui.label("Stride cycle");
                        let active_periods: Vec<f64> = foot_cycle_periods
                            .iter()
                            .copied()
                            .filter(|&period| period > 0.0)
                            .collect();
                        if !active_periods.is_empty() {
                            let mean_period =
                                active_periods.iter().sum::<f64>() / active_periods.len() as f64;
                            ui.label(format!("{mean_period:.2} s ({:.2} Hz)", 1.0 / mean_period));
                        } else {
                            ui.label("-");
                        }
                        ui.end_row();
                        ui.label("Episode");
                        ui.label(format!("{generation}: {episode_step} steps (no time limit)"));
                        ui.end_row();
                        ui.label("Fall resets");
                        ui.label(falls.to_string());
                        ui.end_row();
                    });

                ui.separator();
                ui.label("Velocity command [vx, vy, wz]");
                let mut changed = false;
                changed |= ui
                    .add(egui::Slider::new(&mut command[0], -1.0..=1.0).text("vx [m/s]"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut command[1], -0.5..=0.5).text("vy [m/s]"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut command[2], -1.0..=1.0).text("wz [rad/s]"))
                    .changed();
                let (mutex, updated) = &*shared;
                if changed {
                    mutex.lock().unwrap().command = command;
                }
                if ui.button("Reset robot").clicked() {
                    mutex.lock().unwrap().reset_requested = true;
                    updated.notify_all();
                }
                ui.separator();
                egui::CollapsingHeader::new("Foot telemetry & impulse")
                    .default_open(true)
                    .show(ui, |ui| {
                        let impulse_series = [
                            ("FL", egui::Color32::from_rgb(255, 60, 60), &impulse_history[0]),
                            ("FR", egui::Color32::from_rgb(60, 230, 60), &impulse_history[1]),
                            ("RL", egui::Color32::from_rgb(60, 140, 255), &impulse_history[2]),
                            ("RR", egui::Color32::from_rgb(255, 200, 30), &impulse_history[3]),
                        ];
                        draw_graph(ui, "Contact Impulse", &impulse_series, "N·s", false);
                        ui.small("Real-time ground normal impulse per step (last 3 s). Diagonal trotting pairs peak together.");
                        ui.add_space(4.0);
                        egui::Grid::new("foot_telemetry")
                            .striped(true)
                            .show(ui, |ui| {
                                ui.strong("Foot");
                                ui.strong("Height [m]");
                                ui.strong("center Z [m]");
                                ui.strong("Ground");
                                ui.strong("Cycle [s]");
                                ui.strong("Impulse [N·s]");
                                ui.strong("Last touchdown [N·s]");
                                ui.end_row();
                                for foot_index in 0..FOOT_COUNT {
                                    ui.label(FOOT_NAMES[foot_index]);
                                    ui.label(format!("{:+.4}", foot_heights[foot_index]));
                                    ui.label(format!("{:+.4}", foot_positions[foot_index][2]));
                                    ui.label(if foot_contacts[foot_index] { "CONTACT" } else { "air" });
                                    if foot_cycle_periods[foot_index] > 0.0 {
                                        let period = foot_cycle_periods[foot_index];
                                        ui.label(format!("{period:.2} s ({:.2} Hz)", 1.0 / period));
                                    } else {
                                        ui.label("-");
                                    }
                                    ui.label(format!("{:.3}", contact_impulses[foot_index]));
                                    ui.label(format!("{:.3}", touchdown_impulses[foot_index]));
                                    ui.end_row();
                                }
                            });
                        ui.small("Height is spherical foot bottom to ground. Cycle [s] is touchdown period (cadence [Hz] in parentheses).");
                    });
                egui::CollapsingHeader::new("Motor torque [N·m]")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.label("Graph view:");
                            ui.selectable_value(&mut selected_torque_tab, 0, "FL Leg");
                            ui.selectable_value(&mut selected_torque_tab, 1, "FR Leg");
                            ui.selectable_value(&mut selected_torque_tab, 2, "RL Leg");
                            ui.selectable_value(&mut selected_torque_tab, 3, "RR Leg");
                            ui.selectable_value(&mut selected_torque_tab, 4, "All Knees");
                            ui.selectable_value(&mut selected_torque_tab, 5, "All Pitches");
                        });
                        match selected_torque_tab {
                            0..=3 => {
                                let leg_idx = selected_torque_tab;
                                let offset = leg_idx * 3;
                                let series = [
                                    ("Roll", egui::Color32::from_rgb(80, 220, 240), &torque_history[offset]),
                                    ("Pitch", egui::Color32::from_rgb(255, 150, 40), &torque_history[offset + 1]),
                                    ("Knee", egui::Color32::from_rgb(240, 80, 180), &torque_history[offset + 2]),
                                ];
                                draw_graph(ui, &format!("{} Leg", &FOOT_NAMES[leg_idx][0..2]), &series, "N·m", true);
                            }
                            4 => {
                                let series = [
                                    ("FL", egui::Color32::from_rgb(255, 60, 60), &torque_history[2]),
                                    ("FR", egui::Color32::from_rgb(60, 230, 60), &torque_history[5]),
                                    ("RL", egui::Color32::from_rgb(60, 140, 255), &torque_history[8]),
                                    ("RR", egui::Color32::from_rgb(255, 200, 30), &torque_history[11]),
                                ];
                                draw_graph(ui, "Knee Torques", &series, "N·m", true);
                            }
                            _ => {
                                let series = [
                                    ("FL", egui::Color32::from_rgb(255, 60, 60), &torque_history[1]),
                                    ("FR", egui::Color32::from_rgb(60, 230, 60), &torque_history[4]),
                                    ("RL", egui::Color32::from_rgb(60, 140, 255), &torque_history[7]),
                                    ("RR", egui::Color32::from_rgb(255, 200, 30), &torque_history[10]),
                                ];
                                draw_graph(ui, "Hip Pitch Torques", &series, "N·m", true);
                            }
                        }
                        ui.add_space(4.0);
                        egui::Grid::new("motor_torque")
                            .striped(true)
                            .show(ui, |ui| {
                                ui.strong("Leg");
                                ui.strong("hip roll");
                                ui.strong("hip pitch");
                                ui.strong("knee");
                                ui.end_row();
                                for leg_index in 0..FOOT_COUNT {
                                    let offset = leg_index * 3;
                                    ui.label(&JOINT_NAMES[offset][0..2]);
                                    for torque in &motor_torques[offset..offset + 3] {
                                        ui.label(format!("{torque:+.2}"));
                                    }
                                    ui.end_row();
                                }
                            });
                    });
                ui.separator();
                ui.label("External force on base (world frame)");
                let mut force_settings_changed = false;
                force_settings_changed |= ui
                    .add(egui::Slider::new(&mut force_magnitude, 0.0..=300.0).text("Force [N]"))
                    .changed();
                force_settings_changed |= ui
                    .add(egui::Slider::new(&mut force_duration, 0.02..=1.0).text("Duration [s]"))
                    .changed();
                let mut requested_direction = None;
                ui.horizontal(|ui| {
                    if ui.button("+X").clicked() {
                        requested_direction = Some([1.0, 0.0, 0.0]);
                    }
                    if ui.button("-X").clicked() {
                        requested_direction = Some([-1.0, 0.0, 0.0]);
                    }
                    if ui.button("+Y").clicked() {
                        requested_direction = Some([0.0, 1.0, 0.0]);
                    }
                    if ui.button("-Y").clicked() {
                        requested_direction = Some([0.0, -1.0, 0.0]);
                    }
                    if ui.button("+Z").clicked() {
                        requested_direction = Some([0.0, 0.0, 1.0]);
                    }
                    if ui.button("-Z").clicked() {
                        requested_direction = Some([0.0, 0.0, -1.0]);
                    }
                });
                if force_settings_changed || requested_direction.is_some() {
                    let (mutex, updated) = &*shared;
                    let mut state = mutex.lock().unwrap();
                    state.external_force_magnitude = force_magnitude;
                    state.external_force_duration = force_duration;
                    if let Some(direction) = requested_direction {
                        state.external_force_request = direction.map(|value| value * force_magnitude);
                        state.external_force_request_version += 1;
                        updated.notify_all();
                    }
                }
                if external_force_steps_remaining > 0 {
                    ui.label(format!(
                        "Applied: [{:+.1}, {:+.1}, {:+.1}] N, {:.2} s remaining",
                        external_force_active[0],
                        external_force_active[1],
                        external_force_active[2],
                        external_force_steps_remaining as f64 * POLICY_DT
                    ));
                } else {
                    ui.label("Applied: none (buttons take effect while the policy is stepping)");
                }
                ui.separator();
                ui.label("Camera Controls (시점 제어)");
                ui.horizontal(|ui| {
                    if ui.button("Track Robot (F)").clicked() {
                        *camera_action.lock().unwrap() = Some(CameraAction::TrackRobot);
                    }
                    if ui.button("Free Cam (Esc)").clicked() {
                        *camera_action.lock().unwrap() = Some(CameraAction::Free);
                    }
                });
                ui.horizontal(|ui| {
                    let mut follow = camera_follow_rotation.load(Ordering::Relaxed);
                    if ui.checkbox(&mut follow, "Follow base_link rotation (회전 추종)").changed() {
                        camera_follow_rotation.store(follow, Ordering::Relaxed);
                        *camera_action.lock().unwrap() = Some(CameraAction::SetFollowRotation(follow));
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("3/4 View").clicked() {
                        *camera_action.lock().unwrap() = Some(CameraAction::PresetIsometric);
                    }
                    if ui.button("Side (L)").clicked() {
                        *camera_action.lock().unwrap() = Some(CameraAction::PresetSide);
                    }
                    if ui.button("Side (R)").clicked() {
                        *camera_action.lock().unwrap() = Some(CameraAction::PresetRightSide);
                    }
                    if ui.button("Rear (후방)").clicked() {
                        *camera_action.lock().unwrap() = Some(CameraAction::PresetRear);
                    }
                    if ui.button("Front (전방)").clicked() {
                        *camera_action.lock().unwrap() = Some(CameraAction::PresetFront);
                    }
                    if ui.button("Top (상단)").clicked() {
                        *camera_action.lock().unwrap() = Some(CameraAction::PresetTop);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Rigid Mount (차체 완전 고정):");
                    if ui.button("Rigid Chase").clicked() {
                        *camera_action.lock().unwrap() = Some(CameraAction::RigidChase);
                    }
                    if ui.button("Rigid Side").clicked() {
                        *camera_action.lock().unwrap() = Some(CameraAction::RigidSide);
                    }
                });
                ui.separator();
                ui.label(format!(
                    "TCP lock-step: {ACTION_DIM}-D action → {OBSERVATION_DIM}-D observation ({:.0} Hz)",
                    1.0 / POLICY_DT
                ));
                egui::CollapsingHeader::new("Foot trajectory trails")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Mode:");
                            let mut mode = trail_mode;
                            if ui.radio_value(&mut mode, TrailMode::BodyRelative, "Superimposed shape (중첩)").clicked() {
                                shared.0.lock().unwrap().trail_mode = TrailMode::BodyRelative;
                            }
                            if ui.radio_value(&mut mode, TrailMode::WorldFrame, "World trail (지면)").clicked() {
                                shared.0.lock().unwrap().trail_mode = TrailMode::WorldFrame;
                            }
                        });
                        let mut len = trail_length;
                        let len_label = format!("{len} steps ({:.2} s)", len as f64 * POLICY_DT);
                        if ui.add(egui::Slider::new(&mut len, 15..=200).text(len_label)).changed() {
                            shared.0.lock().unwrap().trail_length = len;
                        }
                        ui.horizontal(|ui| {
                            ui.colored_label(egui::Color32::from_rgb(255, 38, 38), "FL");
                            ui.colored_label(egui::Color32::from_rgb(38, 255, 38), "FR");
                            ui.colored_label(egui::Color32::from_rgb(38, 115, 255), "RL");
                            ui.colored_label(egui::Color32::from_rgb(255, 217, 26), "RR");
                        });
                        ui.small("Superimposed shape mode anchors foot motions to the robot chassis frame. Stride cycles overlap into a closed-loop trajectory without trailing through empty space.");
                    });
                ui.label("Mouse: Left-drag to orbit/rotate; Right-drag/Scroll to zoom");
                ui.label("Keys: F to re-center track; Esc for free camera");
            });
        }
    });

    let viewer_state = viewer.state().clone();
    let physics_thread = std::thread::spawn({
        let shared = shared.clone();
        move || {
            let mut simulation = simulation;
            let mut viewer_sync_counter = 0_u64;
            let mut foot_trails: [VecDeque<[f64; 3]>; FOOT_COUNT] =
                std::array::from_fn(|_| VecDeque::with_capacity(FOOT_TRAIL_LENGTH));
            let mut active_trail_mode = TrailMode::BodyRelative;
            let mut handled_force_request_version = 0_u64;
            let mut active_external_force = [0.0; 3];
            let mut external_force_steps_remaining = 0_usize;
            loop {
                let (
                    action,
                    command,
                    action_version,
                    should_step,
                    reset,
                    force_request,
                    force_duration,
                    force_request_version,
                    target_trail_mode,
                    target_trail_length,
                    shutdown,
                ) = {
                    let (mutex, updated) = &*shared;
                    let mut state = mutex.lock().unwrap();
                    while state.action_version <= state.completed_action_version
                        && !state.reset_requested
                        && !state.shutdown_requested
                    {
                        state = updated.wait(state).unwrap();
                    }
                    let values = (
                        state.action,
                        state.command,
                        state.action_version,
                        state.action_version > state.completed_action_version,
                        state.reset_requested,
                        state.external_force_request,
                        state.external_force_duration,
                        state.external_force_request_version,
                        state.trail_mode,
                        state.trail_length,
                        state.shutdown_requested,
                    );
                    state.reset_requested = false;
                    values
                };
                if shutdown {
                    break;
                }
                if target_trail_mode != active_trail_mode {
                    active_trail_mode = target_trail_mode;
                    for trail in &mut foot_trails {
                        trail.clear();
                    }
                }
                simulation.set_command(command);
                if reset {
                    simulation.reset_episode();
                    active_external_force = [0.0; 3];
                    external_force_steps_remaining = 0;
                    handled_force_request_version = force_request_version;
                }
                if should_step && force_request_version > handled_force_request_version {
                    active_external_force = force_request;
                    external_force_steps_remaining =
                        (force_duration / POLICY_DT).ceil().max(1.0) as usize;
                    handled_force_request_version = force_request_version;
                }
                simulation.set_external_force(if external_force_steps_remaining > 0 {
                    active_external_force
                } else {
                    [0.0; 3]
                });
                let result = if should_step {
                    simulation.policy_step(action)
                } else {
                    simulation.current_result()
                };
                if should_step && external_force_steps_remaining > 0 {
                    external_force_steps_remaining -= 1;
                    if external_force_steps_remaining == 0 {
                        active_external_force = [0.0; 3];
                    }
                }
                if result.automatic_reset {
                    active_external_force = [0.0; 3];
                    external_force_steps_remaining = 0;
                }
                if reset || result.automatic_reset {
                    for trail in &mut foot_trails {
                        trail.clear();
                    }
                }
                if should_step || reset {
                    let positions = match active_trail_mode {
                        TrailMode::BodyRelative => simulation.foot_relative_positions(),
                        TrailMode::WorldFrame => simulation.foot_positions(),
                    };
                    for (trail, position) in foot_trails.iter_mut().zip(positions) {
                        while trail.len() >= target_trail_length {
                            trail.pop_front();
                        }
                        trail.push_back(position);
                    }
                }
                update_shared_result(&shared, &result);
                {
                    let (mutex, _) = &*shared;
                    mutex.lock().unwrap().external_force_steps_remaining =
                        external_force_steps_remaining;
                }
                if should_step {
                    let (mutex, updated) = &*shared;
                    mutex.lock().unwrap().completed_action_version = action_version;
                    updated.notify_all();
                    viewer_sync_counter += 1;
                }

                if (!should_step || viewer_sync_counter.is_multiple_of(2))
                    && let Ok(mut state) = viewer_state.try_lock()
                {
                    state.sync_data(&mut simulation.data);
                    let scene = state.user_scene_mut();
                    scene.clear_geom();
                    let curr_base = simulation.data.xpos()[simulation.base_body_id];
                    let curr_mat = simulation.data.xmat()[simulation.base_body_id];
                    for (foot_index, trail) in foot_trails.iter().enumerate() {
                        let mut points = trail.iter().copied();
                        if let Some(from_raw) = points.next() {
                            let mut from = match active_trail_mode {
                                TrailMode::BodyRelative => [
                                    curr_base[0]
                                        + curr_mat[0] * from_raw[0]
                                        + curr_mat[1] * from_raw[1]
                                        + curr_mat[2] * from_raw[2],
                                    curr_base[1]
                                        + curr_mat[3] * from_raw[0]
                                        + curr_mat[4] * from_raw[1]
                                        + curr_mat[5] * from_raw[2],
                                    curr_base[2]
                                        + curr_mat[6] * from_raw[0]
                                        + curr_mat[7] * from_raw[1]
                                        + curr_mat[8] * from_raw[2],
                                ],
                                TrailMode::WorldFrame => from_raw,
                            };
                            for to_raw in points {
                                let to = match active_trail_mode {
                                    TrailMode::BodyRelative => [
                                        curr_base[0]
                                            + curr_mat[0] * to_raw[0]
                                            + curr_mat[1] * to_raw[1]
                                            + curr_mat[2] * to_raw[2],
                                        curr_base[1]
                                            + curr_mat[3] * to_raw[0]
                                            + curr_mat[4] * to_raw[1]
                                            + curr_mat[5] * to_raw[2],
                                        curr_base[2]
                                            + curr_mat[6] * to_raw[0]
                                            + curr_mat[7] * to_raw[1]
                                            + curr_mat[8] * to_raw[2],
                                    ],
                                    TrailMode::WorldFrame => to_raw,
                                };
                                scene
                                    .create_geom(
                                        MjtGeom::mjGEOM_LINE,
                                        None,
                                        None,
                                        None,
                                        Some(FOOT_TRAIL_COLORS[foot_index]),
                                    )
                                    .connect(FOOT_TRAIL_LINE_WIDTH, from, to);
                                from = to;
                            }
                        }
                    }
                    if !state.running() {
                        break;
                    }
                }
            }
        }
    });

    let mut follow_rotation = true;
    let mut rel_azimuth: f64 = 135.0;
    let mut rel_elevation: f64 = -20.0;
    let mut rel_distance: f64 = 2.2;
    let mut last_cam_azimuth: f64 = 135.0;
    let mut last_base_yaw_deg: f64 = 0.0;
    let mut was_tracking = true;

    while viewer.running() {
        let base_yaw_deg = shared.0.lock().unwrap().base_yaw.to_degrees();

        if let Some(action) = camera_action.lock().unwrap().take() {
            match action {
                CameraAction::TrackRobot => {
                    viewer.track_body(base_link_id);
                    let cam = viewer.camera_mut();
                    cam.distance = rel_distance;
                    cam.elevation = rel_elevation;
                    if follow_rotation {
                        cam.azimuth = base_yaw_deg + rel_azimuth;
                    }
                    last_cam_azimuth = cam.azimuth;
                    last_base_yaw_deg = base_yaw_deg;
                    was_tracking = true;
                }
                CameraAction::Free => {
                    viewer.free_camera();
                    was_tracking = false;
                }
                CameraAction::SetFollowRotation(enabled) => {
                    follow_rotation = enabled;
                    let cam = viewer.camera_mut();
                    if cam.type_ == MjtCamera::mjCAMERA_TRACKING as i32 {
                        if follow_rotation {
                            rel_azimuth = cam.azimuth - base_yaw_deg;
                            cam.azimuth = base_yaw_deg + rel_azimuth;
                        }
                        last_cam_azimuth = cam.azimuth;
                        last_base_yaw_deg = base_yaw_deg;
                    }
                }
                CameraAction::PresetIsometric => {
                    viewer.track_body(base_link_id);
                    rel_azimuth = 135.0;
                    rel_elevation = -20.0;
                    rel_distance = 2.2;
                    let cam = viewer.camera_mut();
                    cam.distance = rel_distance;
                    cam.elevation = rel_elevation;
                    cam.azimuth = if follow_rotation {
                        base_yaw_deg + rel_azimuth
                    } else {
                        rel_azimuth
                    };
                    last_cam_azimuth = cam.azimuth;
                    last_base_yaw_deg = base_yaw_deg;
                    was_tracking = true;
                }
                CameraAction::PresetSide => {
                    viewer.track_body(base_link_id);
                    rel_azimuth = 90.0;
                    rel_elevation = -10.0;
                    rel_distance = 2.0;
                    let cam = viewer.camera_mut();
                    cam.distance = rel_distance;
                    cam.elevation = rel_elevation;
                    cam.azimuth = if follow_rotation {
                        base_yaw_deg + rel_azimuth
                    } else {
                        rel_azimuth
                    };
                    last_cam_azimuth = cam.azimuth;
                    last_base_yaw_deg = base_yaw_deg;
                    was_tracking = true;
                }
                CameraAction::PresetRightSide => {
                    viewer.track_body(base_link_id);
                    rel_azimuth = -90.0;
                    rel_elevation = -10.0;
                    rel_distance = 2.0;
                    let cam = viewer.camera_mut();
                    cam.distance = rel_distance;
                    cam.elevation = rel_elevation;
                    cam.azimuth = if follow_rotation {
                        base_yaw_deg + rel_azimuth
                    } else {
                        rel_azimuth
                    };
                    last_cam_azimuth = cam.azimuth;
                    last_base_yaw_deg = base_yaw_deg;
                    was_tracking = true;
                }
                CameraAction::PresetRear => {
                    viewer.track_body(base_link_id);
                    rel_azimuth = 180.0;
                    rel_elevation = -15.0;
                    rel_distance = 2.0;
                    let cam = viewer.camera_mut();
                    cam.distance = rel_distance;
                    cam.elevation = rel_elevation;
                    cam.azimuth = if follow_rotation {
                        base_yaw_deg + rel_azimuth
                    } else {
                        rel_azimuth
                    };
                    last_cam_azimuth = cam.azimuth;
                    last_base_yaw_deg = base_yaw_deg;
                    was_tracking = true;
                }
                CameraAction::PresetFront => {
                    viewer.track_body(base_link_id);
                    rel_azimuth = 0.0;
                    rel_elevation = -15.0;
                    rel_distance = 2.0;
                    let cam = viewer.camera_mut();
                    cam.distance = rel_distance;
                    cam.elevation = rel_elevation;
                    cam.azimuth = if follow_rotation {
                        base_yaw_deg + rel_azimuth
                    } else {
                        rel_azimuth
                    };
                    last_cam_azimuth = cam.azimuth;
                    last_base_yaw_deg = base_yaw_deg;
                    was_tracking = true;
                }
                CameraAction::PresetTop => {
                    viewer.track_body(base_link_id);
                    rel_azimuth = 90.0;
                    rel_elevation = -89.0;
                    rel_distance = 3.0;
                    let cam = viewer.camera_mut();
                    cam.distance = rel_distance;
                    cam.elevation = rel_elevation;
                    cam.azimuth = if follow_rotation {
                        base_yaw_deg + rel_azimuth
                    } else {
                        rel_azimuth
                    };
                    last_cam_azimuth = cam.azimuth;
                    last_base_yaw_deg = base_yaw_deg;
                    was_tracking = true;
                }
                CameraAction::RigidChase => {
                    if let Some(cam_id) = model.name_to_id(MjtObj::mjOBJ_CAMERA, "base_chase") {
                        viewer.camera_mut().fix(cam_id);
                    }
                    was_tracking = false;
                }
                CameraAction::RigidSide => {
                    if let Some(cam_id) = model.name_to_id(MjtObj::mjOBJ_CAMERA, "base_side") {
                        viewer.camera_mut().fix(cam_id);
                    }
                    was_tracking = false;
                }
            }
        }

        // Real-time camera rotation follow update
        {
            let cam = viewer.camera_mut();
            if cam.type_ == MjtCamera::mjCAMERA_TRACKING as i32 {
                if !was_tracking {
                    was_tracking = true;
                    rel_azimuth = cam.azimuth - base_yaw_deg;
                } else if follow_rotation {
                    let mouse_delta = cam.azimuth - last_cam_azimuth;
                    if mouse_delta.abs() > 1e-4 {
                        rel_azimuth = cam.azimuth - last_base_yaw_deg;
                    }
                    cam.azimuth = base_yaw_deg + rel_azimuth;
                }
                last_cam_azimuth = cam.azimuth;
                last_base_yaw_deg = base_yaw_deg;
            } else {
                was_tracking = false;
            }
        }

        viewer.render().expect("MuJoCo viewer rendering failed");
    }
    {
        let (mutex, updated) = &*shared;
        mutex.lock().unwrap().shutdown_requested = true;
        updated.notify_all();
    }
    physics_thread.join().unwrap();
}
