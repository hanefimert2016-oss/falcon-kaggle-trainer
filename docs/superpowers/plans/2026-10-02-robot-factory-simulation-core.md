# Multi-Robot Simulation Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic 20-robot, 5-cell factory simulation that exports exactly 288 animation frames plus telemetry for the cinematic renderer.

**Architecture:** Split the current monolithic `factory_robot_3d/simulate.py` into focused configuration, domain-model, layout, scheduling, simulation, and export modules. PyBullet owns joint motion and workpiece state; a deterministic scheduler owns part assignment and shared transfer-zone locks.

**Tech Stack:** Python 3.11+, PyBullet, NumPy, dataclasses, pytest, CSV/JSON.

**Spec:** `docs/superpowers/specs/2026-10-02-robot-factory-cinematic-design.md`

## Global Constraints

- Exactly 20 industrial robot arms.
- Exactly 5 production cells with 4 robots per cell.
- Role split: 8 pick-and-place, 4 assembly, 4 welding/joining, 2 quality-control, 2 packaging.
- Showcase duration: 12 seconds.
- Export frame rate: 24 FPS.
- Export frame count: 288.
- At least 6 workpieces must complete the production route.
- Zero double-owned workpieces.
- Zero simultaneous owners for a transfer zone.
- Fixed random seed for deterministic replay.
- Existing FLM training code outside `factory_robot_3d/` must remain untouched.

## Review Focus

- A workpiece requested by two robots in the same tick must have exactly one owner; covered in Task 3 ownership tests.
- Two robots requesting the same transfer zone must serialize rather than overlap; covered in Task 3 lock tests.
- An unreachable robot target must fail the task without corrupting global state; covered in Task 4 failure-path tests.
- Export must contain 288 contiguous frame indices even when a task fails; covered in Task 5 schema tests.
- Re-running with the same seed must produce byte-equivalent animation metadata excluding timestamps; covered in Task 5 deterministic replay test.

---

### Task 1: Configuration and Domain Contracts

**Files:**
- Create: `factory_robot_3d/config.py`
- Create: `factory_robot_3d/models.py`
- Create: `tests/factory_robot_3d/test_config_models.py`

**Interfaces:**
- Produces: `FactoryConfig`, `RobotSpec`, `WorkpieceState`, `TaskState`, `TransferZoneState`.
- Consumes: nothing from later tasks.

- [ ] **Step 1: Write the failing configuration/model tests**

Assert that `FactoryConfig.cinematic_default()` returns robot_count=20, cell_count=5, fps=24, duration_s=12.0, frame_count=288, seed=20261002 and the exact approved role counts.

- [ ] **Step 2: Run the focused tests**

Run: `pytest tests/factory_robot_3d/test_config_models.py -v`  
Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement the contracts**

Implement:
- `FactoryConfig.cinematic_default() -> FactoryConfig`
- `FactoryConfig.validate() -> None`
- immutable `RobotSpec` with `robot_id`, `cell_id`, `role`, `base_position`, `base_yaw`
- mutable workpiece/task/zone state dataclasses with explicit owner IDs.

- [ ] **Step 4: Run tests**

Run: `pytest tests/factory_robot_3d/test_config_models.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

`git add factory_robot_3d/config.py factory_robot_3d/models.py tests/factory_robot_3d/test_config_models.py && git commit -m "feat: add robot factory simulation contracts"`

### Task 2: Factory Layout and Robot Assignment

**Files:**
- Create: `factory_robot_3d/layout.py`
- Create: `tests/factory_robot_3d/test_layout.py`

**Interfaces:**
- Consumes: `FactoryConfig`, `RobotSpec`.
- Produces: `build_factory_layout(config: FactoryConfig) -> FactoryLayout`, where `FactoryLayout` contains cells, robot specs, conveyor paths, transfer zones, QC area, and packaging areas.

- [ ] **Step 1: Write failing layout tests**

Tests must assert:
- 20 unique robot IDs;
- 5 cell IDs;
- 4 robots per cell;
- role totals 8/4/4/2/2;
- no duplicate base positions;
- two main conveyors and one central transfer path;
- at least one transfer zone connecting every adjacent production stage.

- [ ] **Step 2: Run tests**

Run: `pytest tests/factory_robot_3d/test_layout.py -v`  
Expected: FAIL.

- [ ] **Step 3: Implement `FactoryLayout` and `build_factory_layout`**

Use a fixed metric factory coordinate system and deterministic robot placement. Keep all layout constants in this module.

- [ ] **Step 4: Run tests**

Run: `pytest tests/factory_robot_3d/test_layout.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

`git add factory_robot_3d/layout.py tests/factory_robot_3d/test_layout.py && git commit -m "feat: add five-cell factory layout"`

### Task 3: Deterministic Scheduler, Part Ownership, and Zone Locks

**Files:**
- Create: `factory_robot_3d/scheduler.py`
- Create: `tests/factory_robot_3d/test_scheduler.py`

**Interfaces:**
- Consumes: `FactoryLayout`, `RobotSpec`, workpiece/task/zone states.
- Produces: `FactoryScheduler.step(sim_time: float, world: FactoryWorldState) -> list[RobotCommand]`.
- Produces: `acquire_workpiece(workpiece_id, robot_id) -> bool`, `release_workpiece(...)`, `acquire_zone(zone_id, robot_id) -> bool`, `release_zone(...)`.

- [ ] **Step 1: Write failing ownership and lock tests**

Include:
- two robots request one workpiece -> one succeeds;
- two robots request one transfer zone -> one succeeds, second waits;
- releasing ownership makes the resource available on the next step;
- scheduler ordering is stable under seed 20261002;
- no task can skip a production stage.

- [ ] **Step 2: Run tests**

Run: `pytest tests/factory_robot_3d/test_scheduler.py -v`  
Expected: FAIL.

- [ ] **Step 3: Implement scheduler and resource locks**

Represent the route as ordered stages: input -> pick/place -> assembly -> joining -> QC -> packaging -> output. Use deterministic priority: ready time, then stage priority, then robot ID.

- [ ] **Step 4: Run tests**

Run: `pytest tests/factory_robot_3d/test_scheduler.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

`git add factory_robot_3d/scheduler.py tests/factory_robot_3d/test_scheduler.py && git commit -m "feat: add deterministic factory scheduler"`

### Task 4: PyBullet Multi-Robot Simulation Engine

**Files:**
- Create: `factory_robot_3d/simulation.py`
- Create: `tests/factory_robot_3d/test_simulation.py`
- Modify: `factory_robot_3d/simulate.py` to become a thin compatibility entry point.

**Interfaces:**
- Consumes: `FactoryConfig`, `FactoryLayout`, `FactoryScheduler`.
- Produces: `run_simulation(config: FactoryConfig) -> SimulationResult`.
- `SimulationResult` exposes frame samples, telemetry events, completed/lost part counts, conflict count, and per-robot activity.

- [ ] **Step 1: Write failing engine tests**

Cover:
- exactly 20 fixed-base robots created;
- every robot emits telemetry;
- unreachable target returns a failed task event without changing another robot's state;
- one 1-second smoke run produces 24 sampled frame states.

- [ ] **Step 2: Run tests**

Run: `pytest tests/factory_robot_3d/test_simulation.py -v`  
Expected: FAIL.

- [ ] **Step 3: Implement `FactorySimulation` and `run_simulation`**

Run PyBullet at 240 Hz, sample animation state at 24 Hz, and use per-role motion primitives. Shared-zone safety stays scheduler-owned; physics is responsible for arm motion and workpiece transforms.

- [ ] **Step 4: Run tests**

Run: `pytest tests/factory_robot_3d/test_simulation.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

`git add factory_robot_3d/simulation.py factory_robot_3d/simulate.py tests/factory_robot_3d/test_simulation.py && git commit -m "feat: add twenty-robot PyBullet simulation"`

### Task 5: Animation Export, Telemetry, and Full Showcase Regression

**Files:**
- Create: `factory_robot_3d/export.py`
- Create: `factory_robot_3d/run_simulation.py`
- Create: `tests/factory_robot_3d/test_export.py`
- Create: `tests/factory_robot_3d/test_showcase.py`

**Interfaces:**
- Consumes: `SimulationResult`.
- Produces: `write_simulation_artifacts(result, output_dir: Path) -> ArtifactManifest`.
- Output files: `factory_config.json`, `animation.json`, `telemetry.csv`, `summary.json`.

- [ ] **Step 1: Write failing export and end-to-end tests**

Assert:
- frame indices are exactly 1..288;
- every animation frame contains all 20 robots;
- telemetry contains all 20 robot IDs;
- same seed yields the same canonicalized `animation.json`;
- full 12-second run has completed_parts >= 6, lost_parts == 0, transfer_conflicts == 0.

- [ ] **Step 2: Run focused tests**

Run: `pytest tests/factory_robot_3d/test_export.py tests/factory_robot_3d/test_showcase.py -v`  
Expected: FAIL.

- [ ] **Step 3: Implement exporters and CLI**

CLI: `python -m factory_robot_3d.run_simulation --output <dir>`. It must exit non-zero when showcase success criteria fail.

- [ ] **Step 4: Run all robot-factory tests**

Run: `pytest tests/factory_robot_3d -v`  
Expected: PASS.

- [ ] **Step 5: Run the deterministic showcase**

Run: `python -m factory_robot_3d.run_simulation --output /tmp/factory-showcase`  
Expected: summary reports 20 robots, 5 cells, 288 frames, >=6 completed parts, zero lost parts, zero transfer conflicts.

- [ ] **Step 6: Commit**

`git add factory_robot_3d tests/factory_robot_3d && git commit -m "feat: export deterministic multi-robot showcase"`
