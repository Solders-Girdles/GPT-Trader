#!/usr/bin/env python3
"""Generate agent-health schema and example outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "var" / "agents" / "health"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from scripts.agents.health_report import build_example, build_schema  # noqa: E402


def generate(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / "agent_health_schema.json"
    example_path = output_dir / "agent_health_example.json"

    schema_path.write_text(json.dumps(build_schema(), indent=2))
    example_path.write_text(json.dumps(build_example(), indent=2))

    return {
        "agent_health_schema.json": schema_path,
        "agent_health_example.json": example_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate agent-health schema/example files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: var/agents/health)",
    )
    args = parser.parse_args()

    outputs = generate(args.output_dir)
    print("Generated agent-health artifacts:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
