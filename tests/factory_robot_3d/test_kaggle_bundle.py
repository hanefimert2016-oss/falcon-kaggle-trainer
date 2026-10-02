from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import re
import zipfile


def _payload(script: str) -> bytes:
    match = re.search(
        r'^EMBEDDED_PROJECT_ZIP_B64 = "([A-Za-z0-9+/=]+)"$',
        script,
        flags=re.MULTILINE,
    )
    assert match, "embedded payload assignment missing"
    return base64.b64decode(match.group(1))


def test_kaggle_bundle_embeds_required_project_files_deterministically():
    from factory_robot_3d.pipeline.build_kaggle_bundle import build_embedded_script

    package = Path("factory_robot_3d")
    template = package / "kaggle_entrypoint.py"

    first = build_embedded_script(template, package)
    second = build_embedded_script(template, package)

    assert first == second
    assert "__EMBEDDED_PROJECT_ZIP_B64__" not in first

    with zipfile.ZipFile(BytesIO(_payload(first))) as archive:
        names = set(archive.namelist())

    assert "factory_robot_3d/pipeline/run_production.py" in names
    assert "factory_robot_3d/pipeline/install_blender.sh" in names
    assert "factory_robot_3d/pipeline/render_frames.sh" in names
    assert "factory_robot_3d/pipeline/encode_video.sh" in names
    assert "factory_robot_3d/blender/build_scene.py" in names
    assert "factory_robot_3d/blender/probe_gpu.py" in names


def test_kaggle_bundle_excludes_transient_python_cache_files(tmp_path):
    from factory_robot_3d.pipeline.build_kaggle_bundle import build_project_archive

    package = tmp_path / "factory_robot_3d"
    (package / "__pycache__").mkdir(parents=True)
    (package / "ok.py").write_text("x = 1\n")
    (package / "__pycache__" / "bad.pyc").write_bytes(b"bad")

    payload = build_project_archive(package)
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        names = set(archive.namelist())

    assert "factory_robot_3d/ok.py" in names
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
