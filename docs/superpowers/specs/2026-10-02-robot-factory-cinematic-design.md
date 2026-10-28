# Cinematic Multi-Robot Factory Simulation Design

**Date:** 2026-10-02  
**Status:** Approved  
**Branch:** `factory-robot-3d`

## Goal

Replace the existing single-cell PyBullet demo with a cinematic digital-twin-style robot factory simulation in which 20 industrial robot arms work as one coordinated production system. The final deliverable is a high-quality 1920×1080 video, backed by deterministic simulation telemetry and automated validation.

## Approved Scope

- 20 industrial robot arms.
- 5 production cells with 4 robots per cell.
- 2 main conveyors and 1 central transfer line.
- 1 quality-control area and 2 packaging areas.
- Robot roles:
  - 8 pick-and-place robots.
  - 4 assembly robots.
  - 4 welding/joining robots.
  - 2 quality-control routing robots.
  - 2 packaging robots.
- Coordinated part ownership, transfer-zone locks, and collision-aware scheduling.
- Industrial factory environment with safety fencing, floor markings, control panels, warning lights, conveyors, bins, pallets, structural columns, and ceiling fixtures.
- Cinematic materials, lighting, motion blur, reflections, and camera choreography.
- No human characters, PLC integration, live ROS network, or screw-level manufacturing physics in this version.

## Final Presentation Target

- Resolution: 1920×1080.
- Frame rate: 24 FPS.
- Duration: 12 seconds.
- Total frames: 288.
- Renderer: Blender 5.2.2 LTS Cycles.
- Preferred device: NVIDIA OptiX on Tesla T4.
- Fallback device: NVIDIA CUDA on Tesla T4.
- No CPU final-render fallback: GPU unavailability is a failed render.
- Cycles samples: 64 with adaptive sampling and denoising.
- Color management: AgX.
- Motion blur enabled for moving robots and conveyors.
- Six camera beats:
  1. Factory-wide establishing shot.
  2. Conveyor tracking shot.
  3. Close multi-arm synchronized pick-and-place shot.
  4. Overhead coordination shot.
  5. Assembly/welding and packaging detail shot.
  6. Final wide hero shot with all cells active.

## Architecture

The system is split into three independently testable subsystems.

### 1. Multi-Robot Simulation Core

A Python/PyBullet simulation owns factory time, robot state, workpiece state, transfer-zone locks, and deterministic task scheduling. It runs 20 fixed-base 7-DOF industrial arms in one coordinated world, records robot joint positions and workpiece transforms, and exports animation-ready data at 24 FPS.

The scheduler prevents two robots from owning the same workpiece or transfer zone simultaneously. Shared transfer points are treated as explicit resources. The simulation is deterministic under a fixed seed so regressions can be reproduced.

### 2. Blender Cinematic Scene and Render

A Blender Python pipeline consumes the simulation export and procedurally builds the factory scene. Robot rigs are instantiated from reusable geometry, then driven from exported joint transforms. Factory structures, conveyors, safety fencing, floor markings, lights, warning beacons, panels, bins, pallets, and workpieces are generated procedurally so the pipeline has no licensing dependency on external assets.

Cycles handles the final image. Materials use Principled BSDF with painted metal, brushed metal, rubber, glass, plastic, emissive displays, and epoxy floor surfaces. The scene uses physically plausible area lights and emissive fixtures, controlled volumetrics, reflections, depth of field on selected shots, and motion blur.

### 3. Kaggle T4 / GitHub Actions Pipeline

GitHub Actions validates the source, builds a private Kaggle kernel bundle, requests Tesla T4 acceleration, waits for the Kaggle job to finish, downloads outputs, validates the MP4/JSON/CSV artifacts, and uploads one final GitHub Actions artifact.

The Kaggle job:
1. verifies the assigned NVIDIA GPU,
2. installs/starts Blender 5.2.2 LTS headlessly,
3. runs the simulation,
4. builds the Blender scene,
5. renders all 288 frames with Cycles on OptiX or CUDA,
6. encodes H.264 MP4 with ffmpeg,
7. writes telemetry and summary files,
8. runs artifact validation.

## Data Contracts

### `factory_config.json`

Contains immutable run-level settings including robot count, cell count, FPS, duration, resolution, render samples, random seed, and camera shot ranges.

### `animation.json`

Contains one record per rendered frame:
- frame index,
- simulation time,
- each robot's seven joint angles,
- each workpiece transform and state,
- conveyor offsets,
- warning-light states,
- active task IDs.

### `telemetry.csv`

Contains event-oriented records:
- time,
- robot ID,
- cell ID,
- task ID,
- workpiece ID,
- phase,
- transfer-zone ID,
- attached/released state,
- task completion result.

### `summary.json`

Must include:
- success,
- robot_count,
- cell_count,
- completed_parts,
- lost_parts,
- transfer_conflicts,
- simulation_frames,
- rendered_frames,
- width,
- height,
- fps,
- render_engine,
- render_device,
- gpu,
- render_seconds,
- video duration,
- output filenames.

## Simulation Success Criteria

A simulation run passes only when:
- exactly 20 robots are instantiated;
- exactly 5 cells are active;
- no workpiece has two owners;
- no transfer zone has two simultaneous owners;
- no workpiece is lost outside the modeled production flow;
- every active robot produces telemetry;
- at least 6 workpieces finish the complete production route during the 12-second showcase;
- the animation export contains exactly 288 frame records.

## Render Success Criteria

A render run passes only when:
- Blender reports Cycles;
- the selected final render device contains `OPTIX` or `CUDA`;
- `nvidia-smi` reports at least one Tesla T4;
- the MP4 is 1920×1080;
- the MP4 is 24 FPS;
- the MP4 contains 288 frames;
- all six camera ranges are represented;
- final frame and preview stills exist;
- `summary.json` reports `success=true`.

## Error Handling

Critical errors fail the run rather than silently degrading:
- no T4 assigned;
- Blender cannot enable OptiX or CUDA;
- missing animation frames;
- invalid robot/workpiece ownership;
- transfer-zone conflict;
- incomplete video;
- missing telemetry or summary.

OptiX may fall back to CUDA. Final rendering may not fall back to CPU.

Noncritical visual extras such as one missing decorative prop may be logged without invalidating the production simulation, provided the render contract remains satisfied.

## Testing Strategy

- Pure Python unit tests for configuration, layout, scheduler locks, workpiece ownership, export schemas, and deterministic replay.
- Integration test for the complete 20-robot / 5-cell simulation.
- Blender headless smoke tests for scene creation, robot count, material count, camera cuts, and keyframe import.
- One low-resolution preview render before full-quality rendering.
- ffprobe validation for resolution, FPS, codec, frame count, and duration.
- Artifact checks for MP4, JSON, CSV, final frame, preview frame, and render log.

## Deliverables

The final artifact directory must contain:

- `factory_robot_cinematic.mp4`
- `factory_config.json`
- `animation.json`
- `telemetry.csv`
- `summary.json`
- `final_frame.png`
- `preview_wide.png`
- `preview_close.png`
- `render.log`

## Implementation Decomposition

Because the approved design contains three substantial subsystems, implementation is intentionally split into three plans:

1. multi-robot simulation core;
2. Blender cinematic scene and render;
3. Kaggle/GitHub orchestration and final validation.

Each plan ends in a working, independently testable deliverable and provides a stable interface to the next plan.
