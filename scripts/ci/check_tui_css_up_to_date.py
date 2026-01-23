#!/usr/bin/env python3
"""Verify generated TUI CSS artifacts are up to date.

This is used by CI/pre-commit to prevent editing TCSS modules without
regenerating the compiled output files.

Usage:
    python scripts/ci/check_tui_css_up_to_date.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

GENERATED_FILES = (
    Path("src/gpt_trader/tui/styles/main.tcss"),
    Path("src/gpt_trader/tui/styles/main_light.tcss"),
    Path("src/gpt_trader/tui/styles/main_high_contrast.tcss"),
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _run_build(project_root: Path) -> None:
    build_script = project_root / "scripts" / "build_tui_css.py"
    subprocess.run([sys.executable, str(build_script)], cwd=project_root, check=True)


def _diff_names(project_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", *(str(p) for p in GENERATED_FILES)],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _print_diff(project_root: Path) -> None:
    subprocess.run(
        ["git", "diff", "--", *(str(p) for p in GENERATED_FILES)],
        cwd=project_root,
        check=False,
    )


def main() -> int:
    project_root = _project_root()
    _run_build(project_root)

    changed = _diff_names(project_root)
    if not changed:
        return 0

    print("Error: TUI CSS is out of date.")
    print("Run `python scripts/build_tui_css.py` and commit the updated files:")
    for name in changed:
        print(f"  - {name}")
    _print_diff(project_root)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
