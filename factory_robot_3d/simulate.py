#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import subprocess
import sys
import time

OUT = pathlib.Path("/kaggle/working/factory_robot_3d_output")
OUT.mkdir(parents=True, exist_ok=True)

def ensure_deps():
    needed = []
    try:
        import pybullet  # noqa: F401
    except Exception:
        needed.append("pybullet")
    try:
        import imageio  # noqa: F401
        import imageio_ffmpeg  # noqa: F401
    except Exception:
        needed += ["imageio", "imageio-ffmpeg"]
    try:
        from PIL import Image  # noqa: F401
    except Exception:
        needed.append("pillow")
    if needed:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed])

ensure_deps()

import imageio.v2 as imageio
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

VIDEO = OUT / "factory_robot_3d.mp4"
CSV = OUT / "telemetry.csv"
SUMMARY = OUT / "summary.json"
FINAL = OUT / "final_frame.png"

def gpu_info():
    try:
        return subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"nvidia-smi unavailable: {e}"

print("GPU:", gpu_info(), flush=True)

cid = p.connect(p.DIRECT)
if cid < 0:
    raise RuntimeError("PyBullet DIRECT connection failed")
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0 / 240.0)
p.setPhysicsEngineParameter(numSolverIterations=120, deterministicOverlappingPairs=1)

renderer = p.ER_TINY_RENDERER
renderer_name = "TinyRenderer"
egl_plugin = -1
try:
    egl_plugin = p.loadPlugin("eglRendererPlugin")
    if egl_plugin >= 0:
        renderer = p.ER_BULLET_HARDWARE_OPENGL
        renderer_name = "EGL/OpenGL"
except Exception as e:
    print("EGL renderer not available, falling back:", e, flush=True)

def rgba(hex_rgb, alpha=1.0):
    h = hex_rgb.lstrip("#")
    return [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)] + [alpha]

def box_body(half, pos, color, mass=0.0):
    c = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    v = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=c, baseVisualShapeIndex=v, basePosition=pos)

# Factory cell
plane = p.loadURDF("plane.urdf")
p.changeVisualShape(plane, -1, rgbaColor=rgba("#4b5563"))

# Robot pedestal
box_body([0.32, 0.32, 0.18], [0, 0, 0.18], rgba("#374151"))

# Conveyor deck + legs
box_body([0.75, 0.22, 0.06], [0.55, -0.55, 0.28], rgba("#1f2937"))
for x in (-0.05, 1.15):
    for y in (-0.70, -0.40):
        box_body([0.04, 0.04, 0.28], [x, y, 0.0 + 0.14], rgba("#6b7280"))

# Conveyor rollers (visual)
for x in np.linspace(-0.12, 1.22, 10):
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.035, length=0.38, rgbaColor=rgba("#9ca3af"))
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[float(x), -0.55, 0.35],
                      baseOrientation=p.getQuaternionFromEuler([math.pi/2, 0, 0]))

# Destination bin: floor + four walls
bin_center = np.array([-0.58, 0.55, 0.22], dtype=float)
box_body([0.25, 0.25, 0.035], [bin_center[0], bin_center[1], 0.07], rgba("#2563eb", 0.8))
wall_h = 0.22
box_body([0.27, 0.025, wall_h], [bin_center[0], bin_center[1]-0.25, 0.07+wall_h], rgba("#1d4ed8", 0.55))
box_body([0.27, 0.025, wall_h], [bin_center[0], bin_center[1]+0.25, 0.07+wall_h], rgba("#1d4ed8", 0.55))
box_body([0.025, 0.27, wall_h], [bin_center[0]-0.25, bin_center[1], 0.07+wall_h], rgba("#1d4ed8", 0.55))
box_body([0.025, 0.27, wall_h], [bin_center[0]+0.25, bin_center[1], 0.07+wall_h], rgba("#1d4ed8", 0.55))

# Safety fence posts
for x in (-1.0, 1.35):
    for y in (-1.0, 1.0):
        box_body([0.025, 0.025, 0.75], [x, y, 0.75], rgba("#f59e0b", 0.85))

# KUKA iiwa: 7-DOF industrial arm
robot = p.loadURDF(
    "kuka_iiwa/model.urdf",
    basePosition=[0, 0, 0.36],
    baseOrientation=[0, 0, 0, 1],
    useFixedBase=True,
)
for j in range(p.getNumJoints(robot)):
    p.changeDynamics(robot, j, linearDamping=0.04, angularDamping=0.04)
    p.changeVisualShape(robot, j, rgbaColor=rgba("#f97316"))
ee_link = 6
joint_ids = list(range(7))

# Workpiece
part_half = [0.055, 0.055, 0.055]
part_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=part_half)
part_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=part_half, rgbaColor=rgba("#ef4444"))
part = p.createMultiBody(baseMass=0.25, baseCollisionShapeIndex=part_col, baseVisualShapeIndex=part_vis,
                         basePosition=[1.05, -0.55, 0.42])
p.changeDynamics(part, -1, lateralFriction=0.9, restitution=0.05)

# Camera
WIDTH, HEIGHT, FPS = 640, 480, 24
view = p.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=[0.05, 0.0, 0.55],
    distance=2.65,
    yaw=138,
    pitch=-28,
    roll=0,
    upAxisIndex=2,
)
proj = p.computeProjectionMatrixFOV(fov=52, aspect=WIDTH/HEIGHT, nearVal=0.05, farVal=6.0)

def capture():
    _, _, rgba_img, _, _ = p.getCameraImage(
        WIDTH, HEIGHT,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=renderer,
        shadow=1,
        lightDirection=[1, -1, 2],
    )
    return np.asarray(rgba_img, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)[:, :, :3]

down_q = p.getQuaternionFromEuler([0, math.pi, 0])

# Phase endpoints
pickup = np.array([0.55, -0.55, 0.46])
above_pick = pickup + np.array([0.0, 0.0, 0.28])
place = np.array([bin_center[0], bin_center[1], 0.48])
above_place = place + np.array([0.0, 0.0, 0.28])
home = np.array([0.45, 0.05, 0.95])

phases = [
    ("conveyor", home, 2.0),
    ("approach_pick", above_pick, 1.5),
    ("lower_pick", pickup, 1.0),
    ("grip", pickup, 0.7),
    ("lift", above_pick, 1.2),
    ("transfer", above_place, 2.0),
    ("lower_place", place, 1.1),
    ("release", place, 0.8),
    ("retreat", home, 1.6),
]

sim_hz = 240
capture_every = sim_hz // FPS
telemetry = []
constraint_id = None
part_attached = False
frame_count = 0
sim_step = 0

# Initialize robot close to home.
q0 = p.calculateInverseKinematics(robot, ee_link, home.tolist(), down_q, maxNumIterations=200, residualThreshold=1e-5)
for j, q in zip(joint_ids, q0[:7]):
    p.resetJointState(robot, j, q)

video = imageio.get_writer(str(VIDEO), fps=FPS, codec="libx264", quality=8, macro_block_size=None)

def set_target(target):
    q = p.calculateInverseKinematics(
        robot, ee_link, target.tolist(), down_q,
        maxNumIterations=120, residualThreshold=1e-5,
    )
    for j, angle in zip(joint_ids, q[:7]):
        p.setJointMotorControl2(
            robot, j, p.POSITION_CONTROL,
            targetPosition=float(angle),
            force=500,
            positionGain=0.16,
            velocityGain=0.9,
        )

def smooth(a, b, u):
    u = min(1.0, max(0.0, u))
    s = u*u*(3.0 - 2.0*u)
    return a*(1.0-s) + b*s

prev = home.copy()
start_time = time.time()

for phase, target, duration in phases:
    steps = max(1, int(duration * sim_hz))
    for local_step in range(steps):
        u = local_step / max(1, steps - 1)

        # Conveyor moves the part into pickup position during first phase.
        if phase == "conveyor" and not part_attached:
            x = float(1.05*(1-u) + pickup[0]*u)
            p.resetBasePositionAndOrientation(part, [x, -0.55, 0.42], [0, 0, 0, 1])
            p.resetBaseVelocity(part, [0, 0, 0], [0, 0, 0])

        current_target = smooth(prev, target, u)
        set_target(current_target)

        if phase == "grip" and constraint_id is None and u > 0.35:
            ee_state = p.getLinkState(robot, ee_link, computeForwardKinematics=True)
            part_pos, _ = p.getBasePositionAndOrientation(part)
            dist = np.linalg.norm(np.array(ee_state[4]) - np.array(part_pos))
            # Attach the workpiece beneath the tool. The distance gate still checks that
            # the arm genuinely reached the pickup area first.
            if dist < 0.22:
                constraint_id = p.createConstraint(
                    parentBodyUniqueId=robot,
                    parentLinkIndex=ee_link,
                    childBodyUniqueId=part,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, 0.10],
                    childFramePosition=[0, 0, 0],
                )
                part_attached = True

        if phase == "release" and constraint_id is not None and u > 0.25:
            p.removeConstraint(constraint_id)
            constraint_id = None
            part_attached = False

        p.stepSimulation()
        sim_step += 1

        if sim_step % capture_every == 0:
            img = capture()
            video.append_data(img)
            frame_count += 1

        if sim_step % 4 == 0:
            ee = p.getLinkState(robot, ee_link, computeForwardKinematics=True)[4]
            pp, _ = p.getBasePositionAndOrientation(part)
            qs = [p.getJointState(robot, j)[0] for j in joint_ids]
            telemetry.append([
                sim_step / sim_hz, phase,
                *ee, *pp, *qs,
                int(part_attached),
            ])

    prev = target.copy()

# Let released part settle physically in the bin.
for _ in range(sim_hz * 2):
    set_target(home)
    p.stepSimulation()
    sim_step += 1
    if sim_step % capture_every == 0:
        img = capture()
        video.append_data(img)
        frame_count += 1

video.close()

final_img = capture()
Image.fromarray(final_img).save(FINAL)

with CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "time_s", "phase",
        "ee_x", "ee_y", "ee_z",
        "part_x", "part_y", "part_z",
        "q0", "q1", "q2", "q3", "q4", "q5", "q6",
        "attached",
    ])
    w.writerows(telemetry)

part_pos, _ = p.getBasePositionAndOrientation(part)
part_pos = np.array(part_pos)
xy_error = float(np.linalg.norm(part_pos[:2] - bin_center[:2]))
inside_xy = abs(part_pos[0]-bin_center[0]) < 0.20 and abs(part_pos[1]-bin_center[1]) < 0.20
settled_z = 0.08 <= part_pos[2] <= 0.38
success = bool(inside_xy and settled_z)

summary = {
    "success": success,
    "renderer": renderer_name,
    "egl_plugin_id": int(egl_plugin),
    "gpu": gpu_info(),
    "frames": frame_count,
    "video": VIDEO.name,
    "telemetry": CSV.name,
    "final_frame": FINAL.name,
    "final_part_position_m": [float(x) for x in part_pos],
    "bin_center_m": [float(x) for x in bin_center],
    "final_xy_error_m": xy_error,
    "runtime_seconds": round(time.time() - start_time, 3),
}
SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)

p.disconnect()

if not success:
    raise SystemExit("Simulation completed but final placement validation failed")
