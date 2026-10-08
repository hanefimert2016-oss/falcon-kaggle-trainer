# Iron Man Systems Lab v3

Fan-made, non-canon 3D engineering visualization built with Blender on Kaggle T4 and orchestrated from GitHub Actions.

## Deliverables

The render workflow validates all of these before publishing the release:

- `ironman_systems_v3_editable.blend` — editable master scene.
- `ironman_systems_v3_interactive.glb` — portable animated 3D scene.
- `ironman_systems_v3_presentation.mp4` — cinematic systems tour.
- `ironman_systems_v3_poster.png` — high-resolution verification/poster frame.
- `ironman_systems_v3_manifest.json` — systems/asset/source manifest.
- `ironman_systems_v3_gpu_report.json` — Kaggle GPU and output-size report.

## External 3D asset set

The armor gallery uses external fan-made models credited by the upstream portfolio project as CC BY 4.0 assets:

- Mark 1 — CAPTAAINR source model.
- Mark 7 — CHANG747 source model.
- Mark 50 / Infinity armor — CAPTAAINR source model.
- Mark 85 — Vfx Boy source model.
- War Machine — CAPTAAINR source model.
- Hulkbuster presentation variant based on the credited external armor set.
- Arc Reactor — Ludus101 source model.

The project keeps source attribution in the generated manifest. Iron Man and related characters are Marvel intellectual property; this project is a non-commercial fan visualization.

## Systems visualization

The master scene combines the external armor meshes with a separate engineering-visualization layer for:

- Arc-reactor core and power-distribution visualization.
- Internal frame / servo and joint visualization.
- Cooling / thermal path visualization.
- Repulsor and flight/thruster modules.
- Sensor / control-core visualization.
- Nanotech / nanobot reservoir and assembly animation.
- Exploded-view armor inspection.
- Hall-of-Armor multi-suit presentation.

These internals are illustrative and non-canon, not real weapon or propulsion engineering specifications.

## Pipeline

GitHub Actions builds a private Kaggle kernel, submits it to a Tesla T4 session, waits for Blender to finish, downloads the validated outputs, uploads an Actions artifact, then publishes the same files as the `ironman-systems-v3` GitHub Release.

The workflow also handles Meshopt-compressed source GLBs by decoding `EXT_meshopt_compression` before Blender import without intentionally simplifying the geometry.
