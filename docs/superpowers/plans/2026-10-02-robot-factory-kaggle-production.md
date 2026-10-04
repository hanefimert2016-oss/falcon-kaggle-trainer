# Kaggle T4 and GitHub Actions Production Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the approved 20-robot cinematic simulation and Blender render reproducibly on Kaggle Tesla T4, validate the 1080p/24 FPS/288-frame MP4, and publish one verified GitHub Actions artifact.

**Architecture:** GitHub Actions performs source validation and creates a private Kaggle kernel bundle. The Kaggle job verifies T4 availability, runs the multi-robot simulation, downloads/starts Blender 5.2.2 LTS headlessly, renders with OptiX or CUDA, encodes H.264, emits JSON/CSV/log artifacts, and exits non-zero on any contract violation. GitHub then downloads and re-validates the outputs before uploading the final artifact.

**Tech Stack:** GitHub Actions, Kaggle CLI, Kaggle Tesla T4, Blender 5.2.2 LTS, Cycles, OptiX/CUDA, ffmpeg/ffprobe, Python 3.11+.

**Spec:** `docs/superpowers/specs/2026-10-02-robot-factory-cinematic-design.md`

## Global Constraints

- Private Kaggle kernel.
- Tesla T4 required; no CPU final-render fallback.
- Blender 5.2.2 LTS Linux build.
- Final video: 1920×1080, H.264, 24 FPS, exactly 288 frames.
- Simulation summary must pass before Blender rendering starts.
- OptiX preferred; CUDA allowed fallback.
- Final artifact must contain MP4, config JSON, animation JSON, telemetry CSV, summary JSON, final frame, two preview stills, and render log.
- Existing FLM workflows must not be changed except where explicitly required to avoid accidental cross-triggering.
- GPU quota/session failures must be reported clearly, not treated as simulation failures.

## Review Focus

- Kaggle GPU quota/session saturation must stop before uploading a misleading failed render; covered in Task 2 preflight tests.
- Blender download/version mismatch must fail the environment setup; covered in Task 3 version verification.
- OptiX failure must fall back to CUDA exactly once; CPU must never be selected for production; covered in Task 4.
- Partial frame sequences must not be encoded as a successful final video; covered in Task 5 frame validation.
- GitHub must reject a Kaggle output whose MP4 metadata disagrees with summary.json; covered in Task 6 artifact validation.

---

### Task 1: Production Kernel Entrypoint and Artifact Contract

**Files:**
- Create: `factory_robot_3d/pipeline/run_production.py`
- Create: `factory_robot_3d/pipeline/validate_artifacts.py`
- Create: `factory_robot_3d/pipeline/__init__.py`
- Create: `tests/factory_robot_3d/test_pipeline_contract.py`

**Interfaces:**
- Consumes: simulation CLI and Blender scene builder/render modules from Plans 1 and 2.
- Produces: `run_production(output_dir: Path) -> ProductionSummary`.
- Produces: `validate_artifacts(output_dir: Path) -> ValidationReport`.

- [ ] **Step 1: Write failing artifact-contract tests**

Require these exact filenames:
- `factory_robot_cinematic.mp4`
- `factory_config.json`
- `animation.json`
- `telemetry.csv`
- `summary.json`
- `final_frame.png`
- `preview_wide.png`
- `preview_close.png`
- `render.log`

Reject missing, empty, or duplicate-named outputs.

- [ ] **Step 2: Run tests**

Run: `pytest tests/factory_robot_3d/test_pipeline_contract.py -v`  
Expected: FAIL.

- [ ] **Step 3: Implement orchestration and artifact validator**

The production entrypoint must execute simulation -> scene build -> preview validation -> final render -> encode -> final validation in that order. It must stop immediately if simulation success criteria fail.

- [ ] **Step 4: Run tests**

Run: `pytest tests/factory_robot_3d/test_pipeline_contract.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

`git add factory_robot_3d/pipeline tests/factory_robot_3d/test_pipeline_contract.py && git commit -m "feat: add cinematic production pipeline contract"`

### Task 2: Kaggle GPU Preflight and Session Failure Classification

**Files:**
- Create: `factory_robot_3d/pipeline/kaggle_preflight.py`
- Create: `tests/factory_robot_3d/test_kaggle_preflight.py`

**Interfaces:**
- Produces: `classify_kaggle_status(text: str) -> KaggleStatus`.
- Produces: `verify_gpu_identity(nvidia_smi_text: str) -> list[str]`.
- `KaggleStatus` distinguishes queued, running, complete, kernel_error, quota_error, cancelled, unknown.

- [ ] **Step 1: Write failing status parsing tests**

Include the real failure text previously observed:
- `Maximum batch GPU session count of 2 reached` -> quota_error;
- `KernelWorkerStatus.RUNNING` -> running;
- `KernelWorkerStatus.COMPLETE` -> complete;
- `KernelWorkerStatus.ERROR` -> kernel_error.

Require `verify_gpu_identity` to accept one or more `Tesla T4` lines and reject no-GPU or non-NVIDIA output.

- [ ] **Step 2: Run tests**

Run: `pytest tests/factory_robot_3d/test_kaggle_preflight.py -v`  
Expected: FAIL.

- [ ] **Step 3: Implement status and GPU parsers**

Keep parsing pure and testable; shell execution stays in workflow/kernel wrappers.

- [ ] **Step 4: Run tests**

Run: `pytest tests/factory_robot_3d/test_kaggle_preflight.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

`git add factory_robot_3d/pipeline/kaggle_preflight.py tests/factory_robot_3d/test_kaggle_preflight.py && git commit -m "feat: classify Kaggle GPU preflight failures"`

### Task 3: Reproducible Blender 5.2.2 LTS Bootstrap

**Files:**
- Create: `factory_robot_3d/pipeline/install_blender.sh`
- Create: `factory_robot_3d/pipeline/check_blender.py`
- Create: `tests/factory_robot_3d/test_blender_bootstrap.py`

**Interfaces:**
- Produces executable `$BLENDER_HOME/blender`.
- `check_blender.py` exits 0 only when the reported version begins with `Blender 5.2.2`.

- [ ] **Step 1: Write failing bootstrap metadata tests**

Test pinned version, archive name, extraction directory naming, and checksum field presence.

- [ ] **Step 2: Run tests**

Run: `pytest tests/factory_robot_3d/test_blender_bootstrap.py -v`  
Expected: FAIL.

- [ ] **Step 3: Implement installer**

Download the official Linux Blender 5.2.2 LTS archive, verify its configured SHA-256, extract under the Kaggle working/cache directory, and avoid redownloading a verified existing installation.

- [ ] **Step 4: Verify version in a Linux smoke environment**

Run: `$BLENDER_HOME/blender --version`  
Expected: first line contains `Blender 5.2.2`.

- [ ] **Step 5: Commit**

`git add factory_robot_3d/pipeline/install_blender.sh factory_robot_3d/pipeline/check_blender.py tests/factory_robot_3d/test_blender_bootstrap.py && git commit -m "build: pin Blender 5.2.2 LTS"`

### Task 4: T4 Cycles Device Probe with OptiX-to-CUDA Fallback

**Files:**
- Create: `factory_robot_3d/blender/probe_gpu.py`
- Create: `tests/factory_robot_3d/test_gpu_probe.py`

**Interfaces:**
- Produces: `select_cycles_backend(devices: list[DeviceInfo]) -> DeviceSelection`.
- Selection order: OPTIX Tesla T4 -> CUDA Tesla T4 -> raise `RuntimeError`.

- [ ] **Step 1: Write failing device-selection tests**

Assert:
- OptiX T4 wins over CUDA T4;
- CUDA T4 is selected when OptiX unavailable;
- CPU-only raises;
- unrelated GPU names without T4 fail production policy.

- [ ] **Step 2: Run tests**

Run: `pytest tests/factory_robot_3d/test_gpu_probe.py -v`  
Expected: FAIL.

- [ ] **Step 3: Implement Blender device probe**

Run inside Blender Python and write `render_device.json` containing backend, device names, Blender version, and GPU identity.

- [ ] **Step 4: Run Kaggle smoke probe**

Expected: Tesla T4 reported and backend is OPTIX or CUDA.

- [ ] **Step 5: Commit**

`git add factory_robot_3d/blender/probe_gpu.py tests/factory_robot_3d/test_gpu_probe.py && git commit -m "feat: select T4 Cycles backend"`

### Task 5: Frame Rendering, Encoding, and Media Validation

**Files:**
- Create: `factory_robot_3d/pipeline/render_frames.sh`
- Create: `factory_robot_3d/pipeline/encode_video.sh`
- Create: `factory_robot_3d/pipeline/validate_video.py`
- Create: `tests/factory_robot_3d/test_video_validation.py`

**Interfaces:**
- Final frame directory contains exactly `000001.png` through `000288.png`.
- Produces: `factory_robot_cinematic.mp4`.
- `validate_video(path: Path) -> VideoInfo`.

- [ ] **Step 1: Write failing video-validation tests**

Using mocked ffprobe JSON, assert acceptance only for:
- width=1920;
- height=1080;
- avg_frame_rate=24/1;
- codec_name=h264;
- frame count=288;
- duration within 11.9..12.1 seconds.

Reject partial sequences before ffmpeg runs.

- [ ] **Step 2: Run tests**

Run: `pytest tests/factory_robot_3d/test_video_validation.py -v`  
Expected: FAIL.

- [ ] **Step 3: Implement frame render wrapper**

Invoke Blender in background mode with the saved production `.blend` and selected Cycles backend. Write stdout/stderr to `render.log`.

- [ ] **Step 4: Implement H.264 encoding**

Use ffmpeg with input 24 FPS, H.264, yuv420p, and a high-quality constant-rate-factor setting suitable for archival preview.

- [ ] **Step 5: Run validator tests and one 6-frame preview encode**

Expected: tests PASS and preview MP4 metadata matches configured dimensions/FPS for its preview contract.

- [ ] **Step 6: Commit**

`git add factory_robot_3d/pipeline/render_frames.sh factory_robot_3d/pipeline/encode_video.sh factory_robot_3d/pipeline/validate_video.py tests/factory_robot_3d/test_video_validation.py && git commit -m "feat: render and validate cinematic video"`

### Task 6: Consolidated GitHub Actions -> Kaggle T4 Workflow

**Files:**
- Modify: `.github/workflows/factory-robot-3d.yml`
- Keep manual-only: `.github/workflows/factory-robot-3d-kaggle.yml`
- Modify: `.github/workflows/factory-robot-3d-status.yml`
- Create: `factory_robot_3d/kernel-metadata.template.json`
- Create: `tests/factory_robot_3d/test_workflow_contract.py`

**Interfaces:**
- The primary workflow owns the production kernel slug `factory-robot-cinematic-t4`.
- GitHub artifact name: `factory-robot-cinematic-t4-output`.

- [ ] **Step 1: Write failing workflow-contract tests**

Parse the YAML as text/YAML and assert:
- one automatic production workflow for `factory_robot_3d/**`;
- duplicate legacy Kaggle workflow remains workflow_dispatch-only;
- T4 accelerator requested;
- production slug is consistent in push, status, output-download, and status-probe steps;
- wait loop distinguishes quota errors from kernel errors;
- artifact retention is at least 7 days.

- [ ] **Step 2: Run tests**

Run: `pytest tests/factory_robot_3d/test_workflow_contract.py -v`  
Expected: FAIL.

- [ ] **Step 3: Replace the primary workflow with production stages**

Stages:
1. checkout and Python validation;
2. run unit tests;
3. build Kaggle kernel bundle;
4. push T4 kernel;
5. poll status;
6. on ERROR fetch Kaggle logs;
7. on COMPLETE download outputs;
8. run GitHub-side artifact/media validation;
9. upload verified artifact.

- [ ] **Step 4: Update the status probe**

Probe the same production slug and show recent Kaggle logs without consuming another GPU session.

- [ ] **Step 5: Run workflow tests**

Run: `pytest tests/factory_robot_3d/test_workflow_contract.py -v`  
Expected: PASS.

- [ ] **Step 6: Commit**

`git add .github/workflows/factory-robot-3d.yml .github/workflows/factory-robot-3d-kaggle.yml .github/workflows/factory-robot-3d-status.yml factory_robot_3d/kernel-metadata.template.json tests/factory_robot_3d/test_workflow_contract.py && git commit -m "ci: add cinematic robot factory T4 pipeline"`

### Task 7: Production Run and Final Regression Gate

**Files:**
- No new product code unless a failure requires a fix.
- Generated outputs are downloaded from the GitHub Actions artifact, not committed.

**Interfaces:**
- Consumes the final production workflow.
- Produces the verified user deliverables.

- [ ] **Step 1: Run all robot-factory tests**

Run: `pytest tests/factory_robot_3d -v`  
Expected: PASS.

- [ ] **Step 2: Trigger the production workflow**

Use one Kaggle T4 session. Do not trigger the manual duplicate workflow.

- [ ] **Step 3: Verify Kaggle runtime**

Require:
- `nvidia-smi` contains Tesla T4;
- Blender reports 5.2.2;
- render backend is OPTIX or CUDA;
- simulation summary passes before frame 1 is rendered.

- [ ] **Step 4: Verify the final artifact**

Require all nine deliverables and MP4 properties: 1920×1080, H.264, 24 FPS, 288 frames, approximately 12 seconds.

- [ ] **Step 5: Inspect representative frames**

Inspect at least frames corresponding to the six camera beats. Reject obvious missing robots, black renders, broken materials, camera clipping, or incomplete scene geometry.

- [ ] **Step 6: Run one final whole-branch regression**

Re-run Python tests and static workflow validation after any production-run fix.

- [ ] **Step 7: Commit only source fixes**

Do not commit generated video frames or MP4 to the repository. The GitHub Actions artifact is the canonical rendered deliverable.
