from __future__ import annotations

from pathlib import Path


def test_blender_bootstrap_metadata_is_pinned_to_522_lts():
    from factory_robot_3d.pipeline.check_blender import (
        BLENDER_ARCHIVE,
        BLENDER_SHA256,
        BLENDER_URL,
        BLENDER_VERSION,
    )

    assert BLENDER_VERSION == "5.2.2"
    assert BLENDER_ARCHIVE == "blender-5.2.2-linux-x64.tar.xz"
    assert BLENDER_URL.endswith("/Blender5.2/blender-5.2.2-linux-x64.tar.xz")
    assert BLENDER_SHA256 == "84098912789dc450e95697c4184fb8a90acbe5111c2ba4aede3fecb57806a168"
    assert len(BLENDER_SHA256) == 64


def test_blender_installer_contains_checksum_and_official_manifest_verification():
    script = Path("factory_robot_3d/pipeline/install_blender.sh").read_text()

    assert "blender-5.2.2.sha256" in script
    assert "84098912789dc450e95697c4184fb8a90acbe5111c2ba4aede3fecb57806a168" in script
    assert "sha256sum -c" in script
    assert 'grep -F "$ARCHIVE"' in script
    assert "blender-5.2.2-linux-x64.tar.xz" in script


def test_version_parser_accepts_only_blender_522():
    from factory_robot_3d.pipeline.check_blender import verify_version_output

    assert verify_version_output("Blender 5.2.2\n") is True
    assert verify_version_output("Blender 5.2.2 LTS\n") is True
    assert verify_version_output("Blender 5.2.1\n") is False
    assert verify_version_output("not blender") is False
