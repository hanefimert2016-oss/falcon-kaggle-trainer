from __future__ import annotations

import argparse
import base64
from io import BytesIO
from pathlib import Path
import stat
import zipfile


PLACEHOLDER = "__EMBEDDED_PROJECT_ZIP_B64__"


def _include(path: Path, package_root: Path) -> bool:
    relative = path.relative_to(package_root)
    if "__pycache__" in relative.parts:
        return False
    if path.suffix in {".pyc", ".pyo"}:
        return False
    if path.name in {"kernel-metadata.json"}:
        return False
    return path.is_file()


def build_project_archive(package_root: Path) -> bytes:
    package_root = Path(package_root)
    if not package_root.is_dir():
        raise ValueError(f"package root does not exist: {package_root}")

    buffer = BytesIO()
    prefix = package_root.name
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for source in sorted(package_root.rglob("*"), key=lambda p: p.as_posix()):
            if not _include(source, package_root):
                continue
            relative = source.relative_to(package_root).as_posix()
            arcname = f"{prefix}/{relative}"
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            executable = source.suffix == ".sh"
            mode = stat.S_IFREG | (0o755 if executable else 0o644)
            info.external_attr = mode << 16
            archive.writestr(info, source.read_bytes())

    return buffer.getvalue()


def build_embedded_script(template_path: Path, package_root: Path) -> str:
    template_path = Path(template_path)
    template = template_path.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise ValueError(f"template is missing {PLACEHOLDER}")

    payload = base64.b64encode(build_project_archive(package_root)).decode("ascii")
    return template.replace(PLACEHOLDER, payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a self-contained Kaggle entrypoint with embedded project sources."
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    script = build_embedded_script(args.template, args.package_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(script, encoding="utf-8")
    print(
        "KAGGLE_BUNDLE_OK",
        f"output={args.output}",
        f"bytes={args.output.stat().st_size}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
