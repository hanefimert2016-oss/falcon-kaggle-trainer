from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REQUIRED_ARTIFACTS = (
    "factory_robot_cinematic.mp4",
    "factory_config.json",
    "animation.json",
    "telemetry.csv",
    "summary.json",
    "final_frame.png",
    "preview_wide.png",
    "preview_close.png",
    "render.log",
)


@dataclass(frozen=True)
class ArtifactValidationReport:
    ok: bool
    errors: tuple[str, ...]
    files: tuple[Path, ...]


def validate_artifacts(output_dir: Path) -> ArtifactValidationReport:
    output_dir = Path(output_dir)
    errors: list[str] = []
    files: list[Path] = []
    for name in REQUIRED_ARTIFACTS:
        path = output_dir / name
        files.append(path)
        if not path.exists():
            errors.append(f"missing required artifact: {name}")
            continue
        if not path.is_file():
            errors.append(f"required artifact is not a file: {name}")
            continue
        if path.stat().st_size <= 0:
            errors.append(f"required artifact is empty: {name}")
    return ArtifactValidationReport(
        ok=not errors,
        errors=tuple(errors),
        files=tuple(files),
    )
