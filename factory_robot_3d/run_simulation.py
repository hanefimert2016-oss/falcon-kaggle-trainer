from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import FactoryConfig
from .export import write_simulation_artifacts
from .simulation import run_simulation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic cinematic robot-factory simulation.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    config = FactoryConfig.cinematic_default()
    result = run_simulation(config)
    manifest = write_simulation_artifacts(result, args.output)
    summary = json.loads(manifest.summary_json.read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
