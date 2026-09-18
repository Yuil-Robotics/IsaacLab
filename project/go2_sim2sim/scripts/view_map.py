#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive WebGL 3D Point Cloud & Live Autonomous Navigation Synchronized Viewer."""

from __future__ import annotations

import argparse
import http.server
import json
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAPS_DIR = PROJECT_ROOT / "logs" / "maps"


def get_latest_ply() -> Path | None:
    """Find the most recently created PLY map in logs/maps/."""
    if not MAPS_DIR.exists():
        return None
    ply_files = sorted(MAPS_DIR.glob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ply_files[0] if ply_files else None


def parse_ply(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse ASCII PLY point cloud file returning positions (N, 3) and colors (N, 3)."""
    positions = []
    colors = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        header = True
        for line in f:
            line_str = line.strip()
            if header:
                if line_str == "end_header":
                    header = False
                continue
            if not line_str:
                continue
            parts = line_str.split()
            if len(parts) >= 3:
                positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if len(parts) >= 6:
                    colors.append([float(parts[3]) / 255.0, float(parts[4]) / 255.0, float(parts[5]) / 255.0])
                else:
                    colors.append([0.2, 0.8, 0.4])

    pos_arr = np.array(positions, dtype=np.float32)
    col_arr = np.array(colors, dtype=np.float32)
    return pos_arr, col_arr


def generate_viewer_html(ply_path: Path, positions: np.ndarray, colors: np.ndarray, out_html: Path) -> None:
    """Generate a high-performance Three.js dual-mode interactive 3D WebGL viewer."""
    flat_positions = positions.flatten().tolist()
    flat_colors = colors.flatten().tolist()

    mins = positions.min(axis=0) if len(positions) > 0 else np.zeros(3)
    maxs = positions.max(axis=0) if len(positions) > 0 else np.zeros(3)
    center = ((mins + maxs) / 2.0).tolist()
    size = (maxs - mins).tolist()

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Go2 3D Map & Live Navigation Viewer - {ply_path.name}</title>
    <style>
        :root {{
            --bg-dark: #0f1117;
            --panel-bg: rgba(20, 24, 35, 0.88);
            --accent-cyan: #00f2fe;
            --accent-green: #10b981;
            --accent-yellow: #f59e0b;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --border-color: rgba(255, 255, 255, 0.12);
        }}
        body {{
            margin: 0;
            overflow: hidden;
            background-color: var(--bg-dark);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: var(--text-primary);
            user-select: none;
        }}
        #canvas-container {{
            width: 100vw;
            height: 100vh;
            display: block;
        }}
        .hud-panel {{
            position: absolute;
            background: var(--panel-bg);
            backdrop-filter: blur(14px);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
            z-index: 10;
        }}
        #main-hud {{
            top: 16px;
            left: 16px;
            padding: 16px 20px;
            min-width: 320px;
            max-width: 360px;
        }}
        .mode-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin-bottom: 12px;
            transition: all 0.3s ease;
        }}
        .mode-standalone {{
            background: rgba(156, 163, 175, 0.2);
            color: #d1d5db;
            border: 1px solid rgba(156, 163, 175, 0.3);
        }}
        .mode-live {{
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.4);
            box-shadow: 0 0 16px rgba(16, 185, 129, 0.3);
        }}
        .pulse-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #34d399;
            animation: pulse 1.5s infinite;
        }}
        .gray-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #9ca3af;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(0.95); opacity: 0.8; box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }}
            70% {{ transform: scale(1.15); opacity: 1; box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }}
            100% {{ transform: scale(0.95); opacity: 0.8; box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }}
        }}
        .title {{
            font-size: 17px;
            font-weight: 700;
            margin: 0 0 6px 0;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 10px;
        }}
        .metric-card {{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 8px;
            padding: 8px 10px;
        }}
        .metric-label {{
            font-size: 11px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-val {{
            font-size: 14px;
            font-weight: 600;
            color: #fff;
            margin-top: 2px;
            font-family: monospace;
        }}
        .live-status-box {{
            background: rgba(0, 242, 254, 0.08);
            border: 1px solid rgba(0, 242, 254, 0.25);
            border-radius: 8px;
            padding: 8px 12px;
            margin-top: 10px;
            font-size: 13px;
            color: #e0f2fe;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        #controls-panel {{
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            padding: 8px 14px;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .btn {{
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 8px;
            color: #fff;
            padding: 8px 14px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .btn:hover {{
            background: rgba(255, 255, 255, 0.18);
            border-color: rgba(255, 255, 255, 0.3);
            transform: translateY(-1px);
        }}
        .btn.active {{
            background: rgba(0, 242, 254, 0.25);
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }}
        #legend {{
            top: 16px;
            right: 16px;
            padding: 14px 18px;
            font-size: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 6px 0;
            color: #d1d5db;
        }}
        .legend-color {{
            width: 14px;
            height: 14px;
            border-radius: 4px;
        }}
    </style>
    <!-- Three.js and OrbitControls CDN -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="canvas-container"></div>

    <!-- Main HUD Overlay -->
    <div id="main-hud" class="hud-panel">
        <div id="mode-badge" class="mode-badge mode-standalone">
            <span id="mode-dot" class="gray-dot"></span>
            <span id="mode-text">Standalone 3D Map Mode</span>
        </div>

        <div class="title">Go2 3D Autonomous Arena</div>

        <div id="live-status-box" class="live-status-box" style="display: none;">
            <span id="live-nav-status">IDLE</span>
        </div>

        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Robot Position</div>
                <div class="metric-val" id="metric-pos">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Robot Heading</div>
                <div class="metric-val" id="metric-yaw">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Target Destination</div>
                <div class="metric-val" id="metric-goal">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Rem. Distance</div>
                <div class="metric-val" id="metric-dist">--</div>
            </div>
        </div>

        <div class="metric-grid" style="margin-top: 6px;">
            <div class="metric-card">
                <div class="metric-label">Linear Speed</div>
                <div class="metric-val" id="metric-vx">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Angular Speed</div>
                <div class="metric-val" id="metric-wz">--</div>
            </div>
        </div>
    </div>

    <!-- Bottom Quick Controls -->
    <div id="controls-panel" class="hud-panel">
        <button id="btn-follow" class="btn" onclick="toggleFollowCamera()">
            Follow Robot
        </button>
        <button class="btn" onclick="setTopDownView()">
            Top-Down
        </button>
        <button class="btn" onclick="resetIsometricView()">
            Isometric
        </button>
        <button class="btn" onclick="togglePointColors()">
            Point Size
        </button>
    </div>

    <!-- Legend Panel -->
    <div id="legend" class="hud-panel">
        <div style="font-weight: 700; margin-bottom: 8px; color: #fff;">Legend</div>
        <div class="legend-item">
            <div class="legend-color" style="background: #00f2fe; box-shadow: 0 0 8px #00f2fe;"></div>
            <span>Go2 Robot</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #10b981; box-shadow: 0 0 8px #10b981;"></div>
            <span>Planned A* Path</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #f59e0b; box-shadow: 0 0 8px #f59e0b;"></div>
            <span>Target Goal Beacon</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #3b82f6;"></div>
            <span>Scanned 3D Points ({len(positions):,})</span>
        </div>
    </div>

    <script>
        // Raw Map Point Cloud Data
        const rawPositions = new Float32Array({flat_positions});
        const rawColors = new Float32Array({flat_colors});
        const mapCenter = {center};
        const mapSize = {size};

        let scene, camera, renderer, controls;
        let pointCloud, pointsMaterial;
        let robotGroup, goalMarker, pathLine, trailLine;
        let trailPoints = [];
        let isFollowCamera = false;
        let colorMode = 'elevation'; // 'original' or 'elevation'

        function init() {{
            const container = document.getElementById('canvas-container');

            // 1. Scene Setup
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0f1117);

            // 2. Camera Setup
            camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 500);
            camera.position.set(mapCenter[0] - 8, mapCenter[1] - 12, mapCenter[2] + 10);
            camera.up.set(0, 0, 1); // Z-Up Coordinate System

            // 3. Renderer Setup
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.shadowMap.enabled = true;
            container.appendChild(renderer.domElement);

            // 4. Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.target.set(mapCenter[0], mapCenter[1], mapCenter[2]);
            controls.enableDamping = true;
            controls.dampingFactor = 0.08;
            controls.maxDistance = 150;

            // 5. Lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
            scene.add(ambientLight);
            const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
            dirLight.position.set(10, -10, 20);
            scene.add(dirLight);

            // 6. Ground Grid
            const gridHelper = new THREE.GridHelper(30, 30, 0x00f2fe, 0x222736);
            gridHelper.rotation.x = Math.PI / 2;
            gridHelper.position.set(0, 0, -0.01);
            scene.add(gridHelper);

            // 7. Load Point Cloud
            loadPointCloud();

            // 8. Create Robot 3D Avatar
            createRobotAvatar();

            // 9. Create Goal Marker & Path Visualizers
            createNavVisualizers();

            // Window resize handler
            window.addEventListener('resize', onWindowResize, false);

            // Start Live State Poller
            startLiveSync();

            // Render loop
            animate();
        }}

        function loadPointCloud() {{
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.BufferAttribute(rawPositions, 3));
            
            // Elevation Colormap by default
            const elevationColors = new Float32Array(rawPositions.length);
            for (let i = 0; i < rawPositions.length; i += 3) {{
                const z = rawPositions[i + 2];
                // Color ramp: Purple/Blue (Ground) -> Green -> Yellow/Red (Obstacles)
                const normZ = Math.min(Math.max((z - mins[2]) / Math.max(0.1, maxs[2] - mins[2]), 0), 1);
                let r, g, b;
                if (normZ < 0.25) {{
                    r = 0.1; g = 0.4 + normZ * 2; b = 0.9;
                }} else if (normZ < 0.6) {{
                    r = 0.2 + (normZ - 0.25) * 1.5; g = 0.9; b = 0.3;
                }} else {{
                    r = 0.95; g = 0.8 - (normZ - 0.6) * 1.2; b = 0.2;
                }}
                elevationColors[i] = r;
                elevationColors[i + 1] = g;
                elevationColors[i + 2] = b;
            }}
            geometry.setAttribute('color', new THREE.BufferAttribute(elevationColors, 3));

            pointsMaterial = new THREE.PointsMaterial({{
                size: 0.045,
                vertexColors: true,
                transparent: true,
                opacity: 0.92,
            }});

            pointCloud = new THREE.Points(geometry, pointsMaterial);
            scene.add(pointCloud);
        }}

        function createRobotAvatar() {{
            robotGroup = new THREE.Group();

            // Main Body Chassis
            const bodyGeo = new THREE.BoxGeometry(0.55, 0.28, 0.16);
            const bodyMat = new THREE.MeshStandardMaterial({{
                color: 0x00f2fe,
                metalness: 0.7,
                roughness: 0.3,
                emissive: 0x003344,
            }});
            const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
            bodyMesh.position.z = 0.18;
            robotGroup.add(bodyMesh);

            // Forward Heading Arrow
            const arrowGeo = new THREE.ConeGeometry(0.12, 0.35, 16);
            const arrowMat = new THREE.MeshStandardMaterial({{
                color: 0x38bdf8,
                emissive: 0x0284c7,
            }});
            const arrowMesh = new THREE.Mesh(arrowGeo, arrowMat);
            arrowMesh.rotation.x = Math.PI / 2;
            arrowMesh.position.set(0.35, 0, 0.18);
            robotGroup.add(arrowMesh);

            // Robot 4 Legs (Stylized cylinders)
            const legGeo = new THREE.CylinderGeometry(0.025, 0.02, 0.25);
            const legMat = new THREE.MeshStandardMaterial({{ color: 0x64748b }});
            const legOffsets = [
                [0.2, 0.12], [0.2, -0.12], [-0.2, 0.12], [-0.2, -0.12]
            ];
            legOffsets.forEach(([lx, ly]) => {{
                const leg = new THREE.Mesh(legGeo, legMat);
                leg.position.set(lx, ly, 0.08);
                robotGroup.add(leg);
            }});

            robotGroup.visible = false; // Hidden until live connection
            scene.add(robotGroup);
        }}

        function createNavVisualizers() {{
            // Target Goal Beacon (Yellow Cylinder)
            const goalGeo = new THREE.CylinderGeometry(0.25, 0.25, 0.8, 24);
            const goalMat = new THREE.MeshStandardMaterial({{
                color: 0xf59e0b,
                emissive: 0xd97706,
                transparent: true,
                opacity: 0.85,
                metalness: 0.5,
            }});
            goalMarker = new THREE.Mesh(goalGeo, goalMat);
            goalMarker.position.z = 0.4;
            goalMarker.visible = false;
            scene.add(goalMarker);

            // A* Planned Path Ribbon
            const pathGeo = new THREE.BufferGeometry();
            const pathMat = new THREE.LineBasicMaterial({{
                color: 0x10b981,
                linewidth: 4,
                transparent: true,
                opacity: 0.95,
            }});
            pathLine = new THREE.Line(pathGeo, pathMat);
            scene.add(pathLine);

            // Traversed Motion Trail
            const trailGeo = new THREE.BufferGeometry();
            const trailMat = new THREE.LineBasicMaterial({{
                color: 0x00f2fe,
                linewidth: 2,
                transparent: true,
                opacity: 0.6,
            }});
            trailLine = new THREE.Line(trailGeo, trailMat);
            scene.add(trailLine);
        }}

        // Real-Time Live State Sync
        let lastLiveTime = 0;
        function startLiveSync() {{
            setInterval(() => {{
                fetch('/nav_live_state.json?t=' + Date.now())
                    .then(res => res.json())
                    .then(data => {{
                        updateLiveState(data);
                    }})
                    .catch(() => {{
                        setModeOffline();
                    }});
            }}, 50); // 20 Hz polling
        }}

        function updateLiveState(data) {{
            const now = Date.now() / 1000.0;
            const isFresh = data.connected && (now - (data.timestamp || 0) < 3.0);

            if (isFresh && data.robot) {{
                setModeLive();

                // 1. Update Robot Avatar Position & Heading
                const rx = data.robot.x;
                const ry = data.robot.y;
                const rz = data.robot.z || 0.2;
                const ryaw = data.robot.yaw || 0.0;

                robotGroup.position.set(rx, ry, rz);
                robotGroup.rotation.z = ryaw;
                robotGroup.visible = true;

                // Motion Trail update
                if (trailPoints.length === 0 || Math.hypot(rx - trailPoints[trailPoints.length-3], ry - trailPoints[trailPoints.length-2]) > 0.08) {{
                    trailPoints.push(rx, ry, rz + 0.05);
                    if (trailPoints.length > 900) trailPoints.splice(0, 3); // keep last 300 points
                    trailLine.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(trailPoints), 3));
                    trailLine.geometry.attributes.position.needsUpdate = true;
                }}

                // 2. Update Goal Marker (3D Z height supported)
                if (data.goal && data.goal.x !== null) {{
                    const gz = (data.goal.z !== null && data.goal.z !== undefined) ? data.goal.z + 0.35 : 0.4;
                    goalMarker.position.set(data.goal.x, data.goal.y, gz);
                    goalMarker.visible = true;
                    document.getElementById('metric-goal').innerText = `${{data.goal.x.toFixed(1)}}, ${{data.goal.y.toFixed(1)}} (Z=${{gz.toFixed(2)}}m)`;
                }} else {{
                    goalMarker.visible = false;
                    document.getElementById('metric-goal').innerText = '--';
                }}

                // 3. Update Planned A* Path (True 3D Z stairs & mezzanine)
                if (data.path && data.path.length > 0) {{
                    const flatPath = [];
                    data.path.forEach(pt => {{
                        const pz = (pt[2] !== undefined && pt[2] !== null) ? pt[2] + 0.08 : 0.12;
                        flatPath.push(pt[0], pt[1], pz);
                    }});
                    pathLine.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(flatPath), 3));
                    pathLine.geometry.attributes.position.needsUpdate = true;
                    pathLine.visible = true;
                }} else {{
                    pathLine.visible = false;
                }}

                // 4. Update HUD Telemetry
                document.getElementById('metric-pos').innerText = '(' + rx.toFixed(1) + ', ' + ry.toFixed(1) + ')';
                document.getElementById('metric-yaw').innerText = (ryaw * 180 / Math.PI).toFixed(1) + ' deg';
                document.getElementById('metric-dist').innerText = data.stats ? data.stats.rem_dist.toFixed(2) + 'm' : '--';
                document.getElementById('metric-vx').innerText = data.stats ? data.stats.vx.toFixed(2) + ' m/s' : '--';
                document.getElementById('metric-wz').innerText = data.stats ? data.stats.wz.toFixed(2) + ' rad/s' : '--';
                
                if (data.status) {{
                    document.getElementById('live-nav-status').innerText = data.status;
                    document.getElementById('live-status-box').style.display = 'flex';
                }}

                // Follow Camera Update
                if (isFollowCamera) {{
                    controls.target.lerp(new THREE.Vector3(rx, ry, rz), 0.1);
                    camera.position.set(
                        rx - 3.5 * Math.cos(ryaw),
                        ry - 3.5 * Math.sin(ryaw),
                        rz + 2.5
                    );
                }}
            }} else {{
                setModeOffline();
            }}
        }}

        function setModeLive() {{
            const badge = document.getElementById('mode-badge');
            badge.className = 'mode-badge mode-live';
            document.getElementById('mode-dot').className = 'pulse-dot';
            document.getElementById('mode-text').innerText = 'LIVE SIM NAVIGATION MODE';
        }}

        function setModeOffline() {{
            const badge = document.getElementById('mode-badge');
            badge.className = 'mode-badge mode-standalone';
            document.getElementById('mode-dot').className = 'gray-dot';
            document.getElementById('mode-text').innerText = 'Standalone 3D Map Mode';
            document.getElementById('live-status-box').style.display = 'none';
            robotGroup.visible = false;
            goalMarker.visible = false;
            pathLine.visible = false;
        }}

        function toggleFollowCamera() {{
            isFollowCamera = !isFollowCamera;
            const btn = document.getElementById('btn-follow');
            btn.className = isFollowCamera ? 'btn active' : 'btn';
        }}

        function setTopDownView() {{
            isFollowCamera = false;
            document.getElementById('btn-follow').className = 'btn';
            controls.target.set(mapCenter[0], mapCenter[1], 0);
            camera.position.set(mapCenter[0], mapCenter[1], 24);
        }}

        function resetIsometricView() {{
            isFollowCamera = false;
            document.getElementById('btn-follow').className = 'btn';
            controls.target.set(mapCenter[0], mapCenter[1], mapCenter[2]);
            camera.position.set(mapCenter[0] - 8, mapCenter[1] - 12, mapCenter[2] + 10);
        }}

        function togglePointColors() {{
            // Toggle point size/visibility mode
            pointsMaterial.size = (pointsMaterial.size === 0.045) ? 0.08 : 0.045;
            pointsMaterial.needsUpdate = true;
        }}

        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}

        function animate() {{
            requestAnimationFrame(animate);
            if (!isFollowCamera) {{
                controls.update();
            }}
            // Animate target goal pulse
            if (goalMarker && goalMarker.visible) {{
                goalMarker.rotation.z += 0.02;
            }}
            renderer.render(scene, camera);
        }}

        const mins = {mins.tolist()};
        const maxs = {maxs.tolist()};

        window.onload = init;
    </script>
</body>
</html>
"""
    out_html.write_text(html_content, encoding="utf-8")
    print(f"[INFO] Generated Dual-Mode WebGL Viewer: {out_html}")


class MapHttpServer:
    """Non-blocking local HTTP server for serving 3D WebGL viewer and live JSON."""

    def __init__(self, directory: Path, port: int = 8088):
        self.directory = directory
        self.port = port
        self.httpd = None
        self.thread = None

    def start(self) -> int:
        handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(
            *args, directory=str(self.directory), **kwargs
        )
        for p in range(self.port, self.port + 10):
            try:
                self.httpd = socketserver.TCPServer(("", p), handler)
                self.port = p
                break
            except OSError:
                continue

        if self.httpd is None:
            raise RuntimeError("Could not find free port for HTTP server.")

        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self.port

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Go2 3D Map & Live Autonomous Navigation Synchronized Viewer.")
    parser.add_argument("map_file", nargs="?", type=Path, default=None, help="Path to .ply map file.")
    parser.add_argument("--port", type=int, default=8088, help="Local HTTP server port (default: 8088)")
    parser.add_argument("--no_browser", action="store_true", help="Do not open browser automatically.")
    args = parser.parse_args()

    ply_path = args.map_file
    if ply_path is None:
        ply_path = get_latest_ply()
        if ply_path is None:
            print(f"[ERROR] No .ply map files found in {MAPS_DIR}")
            return
        print(f"[INFO] Auto-selected latest map: {ply_path.name}")

    positions, colors = parse_ply(ply_path)
    print(f"[INFO] Loaded {len(positions):,} 3D points from {ply_path.name}")

    out_html = MAPS_DIR / "index.html"
    generate_viewer_html(ply_path, positions, colors, out_html)

    # Start local HTTP server
    server = MapHttpServer(directory=MAPS_DIR, port=args.port)
    actual_port = server.start()
    url = f"http://localhost:{actual_port}/index.html"

    print("\n" + "=" * 70)
    print("Go2 3D Map & Live Navigation Viewer Running")
    print("=" * 70)
    print(f"  URL: {url}")
    print("  Modes:")
    print("    1. Standalone Mode : Pure 3D Map Inspection (Sim closed)")
    print("    2. Live Sync Mode   : Real-Time Go2 Pose, Goal & Path (Sim active)")
    print("=" * 70)
    print("Press Ctrl+C to exit server.\n")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[INFO] Viewer server stopped.")
        server.stop()


if __name__ == "__main__":
    main()
