#!/usr/bin/env python3
"""Guard against re-introducing the removed ``gpt_trader.orchestration`` package.

The ``gpt_trader.orchestration`` package was removed in v3.0. This check fails if
any Python file under ``src``, ``tests`` or ``scripts`` references it again.
Canonical replacements: ``app.*``, ``features.live_trade.*``,
``features.brokerages.*`` (see ``docs/DEPRECATIONS.md``).

Single source of truth: invoked identically from CI (``.github/workflows/ci.yml``),
``make ci-required``, and ``uv run local-ci``.

Usage:
    python scripts/ci/check_orchestration_imports.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEARCH_ROOTS: tuple[str, ...] = ("src", "tests", "scripts")
# Matches "from"/"import" statements that reference the removed subpackage.
IMPORT_PATTERN = re.compile(r"(?:from|import)\s+gpt_trader\.orchestration")


def find_violations(project_root: Path) -> list[str]:
    """Return ``path:line: text`` for every orchestration import found."""
    violations: list[str] = []
    for root in SEARCH_ROOTS:
        base = project_root / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if IMPORT_PATTERN.search(line):
                    relative = path.relative_to(project_root).as_posix()
                    violations.append(f"{relative}:{lineno}: {line.strip()}")
    return violations


def main() -> int:
    violations = find_violations(PROJECT_ROOT)
    if violations:
        for violation in violations:
            print(violation)
        print("::error::gpt_trader.orchestration was removed in v3.0", file=sys.stderr)
        print(
            "Use canonical paths: app.*, features.live_trade.*, features.brokerages.*",
            file=sys.stderr,
        )
        print("See docs/DEPRECATIONS.md for migration guidance.", file=sys.stderr)
        return 1
    print("No orchestration imports found - package was removed in v3.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
