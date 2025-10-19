#!/usr/bin/env python3
"""Developer script to run pytest with coverage reporting."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Run pytest with comprehensive coverage reporting."""
    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent
    if not (project_root / "pyproject.toml").exists():
        print("Error: Must run from project root", file=sys.stderr)
        return 1

    # Run pytest with coverage
    cmd = [
        "poetry",
        "run",
        "pytest",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=json:coverage.json",
        "--cov-report=html:htmlcov",
        "--cov-report=xml",
        "-v",
    ]

    # Add any additional args passed to this script
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    print(f"Running: {' '.join(cmd)}")
    print("=" * 80)

    try:
        result = subprocess.run(cmd, cwd=project_root, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error running coverage: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
