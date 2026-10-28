# Blender Cinematic Factory Scene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the 20-robot simulation export into a high-quality 1920×1080, 24 FPS, 12-second cinematic factory sequence rendered in Blender Cycles on NVIDIA GPU.

**Architecture:** A pure-Python schema layer validates simulation artifacts outside Blender. Blender-specific modules then procedurally build the factory, instantiate reusable robot geometry, import 288 frames of joint/workpiece animation, configure materials/lights/camera cuts, and prepare a Cycles scene for headless rendering.

**Tech Stack:** Blender 5.2.2 LTS, Blender Python API, Cycles, OptiX/CUDA, Python 3.11+, pytest, ffmpeg/ffprobe.

**Spec:** `docs/superpowers/specs/2026-10-02-robot-factory-cinematic-design.md`

## Global Constraints

- Exactly 20 visible robot arms.
- Exactly 5 production cells.
- 1920×1080 final output.
- 24 FPS, 12 seconds, 288 frames.
- Cycles final renderer.
- Prefer OptiX; allow CUDA fallback; never CPU for the final render.
- 64 Cycles samples, adaptive sampling enabled, denoising enabled.
- AgX color management.
- Motion blur enabled.
- Six camera beats matching the approved spec.
- Factory assets must be procedurally generated or project-owned; no untracked external asset dependency.
- Existing FLM project visuals and training code remain untouched.

## Review Focus

- Missing or malformed robot frame data must fail import before Blender keyframes are created; covered in Task 1 schema tests.
- A robot rig count other than 20 must fail scene validation; covered in Task 3 scene tests.
- A camera cut range with gaps or overlaps must fail validation; covered in Task 5 camera tests.
- If neither OptiX nor CUDA is enabled, final render setup must fail rather than use CPU; covered in Task 6 device tests.
- Materials or lights missing after scene rebuild must be detected by scene-manifest validation; covered in Tasks 2 and 4.

---

### Task 1: Animation and Render Schema Validation

**Files:**
- Create: `factory_robot_3d/blender/__init__.py`
- Create: `factory_robot_3d/blender/schema.py`
- Create: `tests/factory_robot_3d/test_blender_schema.py`

**Interfaces:**
- Consumes: `factory_config.json`, `animation.json`, `summary.json` from the simulation plan.
- Produces: `load_animation_bundle(input_dir: Path) -> AnimationBundle`.
- `AnimationBundle` contains config, 288 frame records, robot IDs, workpieces, and camera timing metadata.

- [ ] **Step 1: Write failing schema tests**

Assert rejection of:
- 287 or 289 frames;
- duplicate frame indices;
- missing robot IDs in any frame;
- robot count != 20;
- non-finite joint transforms;
- missing required simulation summary fields.

- [ ] **Step 2: Run tests**

Run: `pytest tests/factory_robot_3d/test_blender_schema.py -v`  
Expected: FAIL.

- [ ] **Step 3: Implement `AnimationBundle` and `load_animation_bundle`**

Use JSON only; do not import `bpy` in this module so tests run outside Blender.

- [ ] **Step 4: Run tests**

Run: `pytest tests/factory_robot_3d/test_blender_schema.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

`git add factory_robot_3d/blender/schema.py factory_robot_3d/blender/__init__.py tests/factory_robot_3d/test_blender_schema.py && git commit -m "feat: validate cinematic animation bundle"`

### Task 2: Procedural Factory Geometry and Materials

**Files:**
- Create: `factory_robot_3d/blender/materials.py`
- Create: `factory_robot_3d/blender/factory_scene.py`
- Create: `tests/factory_robot_3d/test_scene_manifest.py`

**Interfaces:**
- Consumes: factory layout constants from simulation artifacts.
- Produces: `build_factory_shell(scene, bundle) -> SceneManifest`.
- `SceneManifest` records named object groups and material slots for validation.

- [ ] **Step 1: Write failing scene-manifest tests**

Validate the required manifest groups:
- epoxy floor;
- 2 main conveyors;
- 1 transfer conveyor;
- 5 cell bases;
- safety fencing;
- structural columns;
- control panels;
- packaging bins;
- warning lights;
- ceiling light fixtures.

Require named materials for painted metal, brushed metal, rubber, safety yellow, glass, emissive display, red warning beacon, and epoxy floor.

- [ ] **Step 2: Run tests**

Run: `pytest tests/factory_robot_3d/test_scene_manifest.py -v`  
Expected: FAIL.

- [ ] **Step 3: Implement material recipes and factory shell builder**

Use Principled BSDF. Keep material construction idempotent by material name. Use beveled procedural meshes and realistic roughness/metallic ranges; no external textures are required for the first production version.

- [ ] **Step 4: Run tests**

Run: `pytest tests/factory_robot_3d/test_scene_manifest.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

`git add factory_robot_3d/blender/materials.py factory_robot_3d/blender/factory_scene.py tests/factory_robot_3d/test_scene_manifest.py && git commit -m "feat: add cinematic procedural factory shell"`

### Task 3: Reusable Industrial Robot Rig and 20-Robot Instancing

**Files:**
- Create: `factory_robot_3d/blender/robot_rig.py`
- Create: `factory_robot_3d/blender/build_scene.py`
- Create: `tests/factory_robot_3d/test_robot_manifest.py`

**Interfaces:**
- Consumes: `AnimationBundle.robot_ids` and robot base transforms.
- Produces: `create_robot_rig(robot_id: str, base_transform) -> RobotRig`.
- Produces: `instantiate_factory_robots(bundle) -> dict[str, RobotRig]`.

- [ ] **Step 1: Write failing robot-manifest tests**

Assert:
- exactly 20 robot IDs;
- every rig exposes joints `j0` through `j6`;
- every rig has one tool mount and one warning/status light;
- base transforms are unique;
- object naming is stable: `Robot_<id>_J0` etc.

- [ ] **Step 2: Run tests**

Run: `pytest tests/factory_robot_3d/test_robot_manifest.py -v`  
Expected: FAIL.

- [ ] **Step 3: Implement a reusable 7-DOF industrial-arm rig**

Build one visually polished procedural arm with beveled housings, joint cylinders, cable guards, painted-metal materials, dark rubber joints, and a compact end effector. Instantiate linked geometry where safe to reduce memory, while keeping transforms independent.

- [ ] **Step 4: Add Blender headless smoke check**

Run: `blender -b --python factory_robot_3d/blender/build_scene.py -- --input /tmp/factory-showcase --output /tmp/factory.blend --smoke`  
Expected: exit 0 and manifest reports 20 rigs.

- [ ] **Step 5: Commit**

`git add factory_robot_3d/blender/robot_rig.py factory_robot_3d/blender/build_scene.py tests/factory_robot_3d/test_robot_manifest.py && git commit -m "feat: add twenty cinematic robot rigs"`

### Task 4: Industrial Lighting and Environment Look

**Files:**
- Create: `factory_robot_3d/blender/lighting.py`
- Create: `tests/factory_robot_3d/test_lighting_manifest.py`

**Interfaces:**
- Consumes: built factory scene.
- Produces: `configure_factory_lighting(scene) -> LightingManifest`.

- [ ] **Step 1: Write failing lighting-manifest tests**

Require:
- ceiling area-light rows;
- practical emissive fixtures;
- a soft key/fill balance for hero areas;
- red/amber warning beacons;
- low-density world/factory atmosphere;
- exposure and AgX settings;
- bounded volumetric density so the scene is not obscured.

- [ ] **Step 2: Run tests**

Run: `pytest tests/factory_robot_3d/test_lighting_manifest.py -v`  
Expected: FAIL.

- [ ] **Step 3: Implement lighting setup**

Keep practical fixtures visible in frame. Use physically plausible large sources rather than many tiny point lights. Configure AgX and a controlled contrast look.

- [ ] **Step 4: Render one low-resolution preview still**

Run Blender at 640×360, 16 samples, one frame.  
Expected: image exists and render completes without scene or shader errors.

- [ ] **Step 5: Commit**

`git add factory_robot_3d/blender/lighting.py tests/factory_robot_3d/test_lighting_manifest.py && git commit -m "feat: add cinematic factory lighting"`

### Task 5: Animation Import and Six-Shot Camera Choreography

**Files:**
- Create: `factory_robot_3d/blender/animation.py`
- Create: `factory_robot_3d/blender/cameras.py`
- Create: `tests/factory_robot_3d/test_camera_plan.py`

**Interfaces:**
- Consumes: `AnimationBundle`, robot rigs, workpiece objects.
- Produces: `apply_animation(bundle, rigs, workpieces) -> AnimationManifest`.
- Produces: `build_camera_plan(scene) -> CameraPlan`.

- [ ] **Step 1: Write failing camera and animation tests**

Pin six contiguous ranges covering frames 1..288:
- 1–48 establishing;
- 49–96 conveyor tracking;
- 97–144 synchronized close shot;
- 145–192 overhead coordination;
- 193–240 assembly/welding/packaging detail;
- 241–288 final hero shot.

Assert no gaps, no overlaps, and all 20 robots receive joint keyframes.

- [ ] **Step 2: Run tests**

Run: `pytest tests/factory_robot_3d/test_camera_plan.py -v`  
Expected: FAIL.

- [ ] **Step 3: Implement animation importer**

Convert exported joint angles and workpiece transforms to Blender keyframes. Use linear interpolation for conveyor motion and Bezier/eased interpolation for cinematic camera rigs only; robot joint data must follow simulation values.

- [ ] **Step 4: Implement camera choreography**

Use a master camera with markers or active-camera switching. Configure selective depth of field only on the close/detail shots.

- [ ] **Step 5: Run a 6-frame camera-cut smoke render**

Render frames 1, 49, 97, 145, 193, 241 at 640×360.  
Expected: six distinct images and no missing-camera errors.

- [ ] **Step 6: Commit**

`git add factory_robot_3d/blender/animation.py factory_robot_3d/blender/cameras.py tests/factory_robot_3d/test_camera_plan.py && git commit -m "feat: animate factory and add cinematic camera plan"`

### Task 6: Production Cycles Configuration and Scene Validation

**Files:**
- Create: `factory_robot_3d/blender/render_config.py`
- Create: `factory_robot_3d/blender/validate_scene.py`
- Create: `tests/factory_robot_3d/test_render_config.py`

**Interfaces:**
- Produces: `configure_cycles(scene, preferred=("OPTIX", "CUDA")) -> RenderDeviceInfo`.
- Produces: `validate_scene(scene, bundle) -> ValidationReport`.

- [ ] **Step 1: Write failing render-config tests**

Assert target settings:
- engine CYCLES;
- 1920×1080;
- 24 FPS;
- frames 1..288;
- 64 samples;
- adaptive sampling on;
- denoising on;
- motion blur on;
- CPU-only device selection raises `RuntimeError`.

- [ ] **Step 2: Run tests**

Run: `pytest tests/factory_robot_3d/test_render_config.py -v`  
Expected: FAIL.

- [ ] **Step 3: Implement GPU-device selection and production settings**

Select OptiX when available, otherwise CUDA. Record the selected backend and device names for `summary.json`.

- [ ] **Step 4: Run a Blender device probe**

Run: `blender -b /tmp/factory.blend -f 1 -- --cycles-device OPTIX` when OptiX is available; otherwise use CUDA.  
Expected: GPU backend is reported; CPU-only final render is rejected.

- [ ] **Step 5: Run full scene validation**

Expected: 20 rigs, 5 cells, all material groups, all lighting groups, six camera ranges, 288 animation frames.

- [ ] **Step 6: Commit**

`git add factory_robot_3d/blender/render_config.py factory_robot_3d/blender/validate_scene.py tests/factory_robot_3d/test_render_config.py && git commit -m "feat: configure production Cycles render"`
