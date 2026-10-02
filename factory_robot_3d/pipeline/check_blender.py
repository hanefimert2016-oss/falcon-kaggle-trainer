from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


BLENDER_VERSION = "5.2.2"
BLENDER_ARCHIVE = "blender-5.2.2-linux-x64.tar.xz"
BLENDER_URL = (
    "https://download.blender.org/release/Blender5.2/"
    + BLENDER_ARCHIVE
)
BLENDER_SHA256 = "84098912789dc450e95697c4184fb8a90acbe5111c2ba4aede3fecb57806a168"


def verify_version_output(text: str) -> bool:
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first.strip() == f"Blender {BLENDER_VERSION}"


def check_blender(binary: Path) -> bool:
    result = subprocess.run(
        [str(binary), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return verify_version_output(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    if not check_blender(args.binary):
        raise SystemExit(f"Expected Blender {BLENDER_VERSION}")
    print(f"BLENDER_VERSION_OK {BLENDER_VERSION} {args.binary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
