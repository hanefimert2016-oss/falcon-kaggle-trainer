from __future__ import annotations

import argparse
import base64
from io import BytesIO
from pathlib import Path
import zipfile

from .build_kaggle_bundle import build_project_archive

PROJECT_PLACEHOLDER = "__EMBEDDED_PROJECT_ZIP_B64__"
SIMULATION_PLACEHOLDER = "__EMBEDDED_SIMULATION_ZIP_B64__"


def build_simulation_archive(simulation_dir: Path) -> bytes:
    simulation_dir = Path(simulation_dir)
    required = (
        "factory_config.json",
        "animation.json",
        "telemetry.csv",
        "summary.json",
    )
    missing = [name for name in required if not (simulation_dir / name).is_file()]
    if missing:
        raise ValueError("missing precomputed simulation files: " + ", ".join(missing))

    buffer = BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(required):
            source = simulation_dir / name
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o100644 << 16)
            archive.writestr(info, source.read_bytes())
    return buffer.getvalue()


def build_precomputed_script(
    template_path: Path,
    package_root: Path,
    simulation_dir: Path,
) -> str:
    template = Path(template_path).read_text(encoding="utf-8")
    if PROJECT_PLACEHOLDER not in template:
        raise ValueError(f"template missing {PROJECT_PLACEHOLDER}")
    if SIMULATION_PLACEHOLDER not in template:
        raise ValueError(f"template missing {SIMULATION_PLACEHOLDER}")

    project_payload = base64.b64encode(
        build_project_archive(Path(package_root))
    ).decode("ascii")
    simulation_payload = base64.b64encode(
        build_simulation_archive(Path(simulation_dir))
    ).decode("ascii")
    return (
        template
        .replace(PROJECT_PLACEHOLDER, project_payload)
        .replace(SIMULATION_PLACEHOLDER, simulation_payload)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--simulation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    script = build_precomputed_script(
        args.template,
        args.package_root,
        args.simulation_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(script, encoding="utf-8")
    print(
        "KAGGLE_PRECOMPUTED_BUNDLE_OK",
        f"output={args.output}",
        f"bytes={args.output.stat().st_size}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
