#!/usr/bin/env python3
"""CLI entry points for agent tools.

Wraps scripts/agents/*.py to expose as uv/poetry commands.
Each function corresponds to an entry point in pyproject.toml.

Usage:
    uv run agent-check --format text
    uv run agent-impact --from-git
    uv run agent-map --component-summary
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _get_scripts_dir() -> Path:
    """Get the scripts/agents directory path."""
    # Navigate from src/gpt_trader/agents/cli.py to project root
    return Path(__file__).parent.parent.parent.parent / "scripts" / "agents"


def _run_script(script_name: str) -> int:
    """Run a script from scripts/agents/ with forwarded args.

    Args:
        script_name: Name of the script file (e.g., "quality_gate.py")

    Returns:
        Exit code from the script
    """
    scripts_dir = _get_scripts_dir()
    script = scripts_dir / script_name

    if not script.exists():
        print(f"Error: Script not found: {script}", file=sys.stderr)
        return 1

    # Run from project root for consistent path resolution
    project_root = scripts_dir.parent.parent
    result = subprocess.run(
        [sys.executable, str(script)] + sys.argv[1:],
        cwd=project_root,
    )
    return result.returncode


def check() -> int:
    """Run quality gate checks.

    Entry point: agent-check

    Runs lint, format, types, and tests with machine-readable output.

    Examples:
        uv run agent-check                    # All checks, JSON output
        uv run agent-check --format text      # Human-readable output
        uv run agent-check --check lint,types # Specific checks
        uv run agent-check --files src/path/  # Check specific paths
        uv run agent-check --full             # Include slow tests
    """
    return _run_script("quality_gate.py")


def impact() -> int:
    """Analyze change impact and suggest tests.

    Entry point: agent-impact

    Analyzes changed files and suggests relevant tests to run.

    Examples:
        uv run agent-impact --from-git        # Analyze git changes
        uv run agent-impact --from-git --base main
        uv run agent-impact --files src/path/file.py
        uv run agent-impact --format text
    """
    return _run_script("change_impact.py")


def map_deps() -> int:
    """Generate dependency graph.

    Entry point: agent-map

    Builds and queries module dependency relationships.

    Examples:
        uv run agent-map                      # Full graph JSON
        uv run agent-map --format text        # Summary view
        uv run agent-map --format dot         # GraphViz output
        uv run agent-map --check-circular     # Find circular imports
        uv run agent-map --depends-on gpt_trader.errors
    """
    return _run_script("dependency_graph.py")


def tests() -> int:
    """Generate test inventory.

    Entry point: agent-tests

    Generates comprehensive test inventory with marker and path filtering.

    Examples:
        uv run agent-tests                    # Generate full inventory
        uv run agent-tests --by-marker risk   # Tests with risk marker
        uv run agent-tests --by-path tests/unit/gpt_trader/cli
        uv run agent-tests --stdout           # Output to stdout
    """
    return _run_script("generate_test_inventory.py")


def risk() -> int:
    """Query risk configuration.

    Entry point: agent-risk

    Queries risk configuration values with documentation.

    Examples:
        uv run agent-risk                     # Full config JSON
        uv run agent-risk --with-docs         # Include field docs
        uv run agent-risk --field max_leverage
        uv run agent-risk --generate-schema
    """
    return _run_script("query_risk_config.py")


def naming() -> int:
    """Check naming standards.

    Entry point: agent-naming

    Scans for naming convention violations.

    Examples:
        uv run agent-naming                   # Full scan
        uv run agent-naming --strict          # Fail on violations
        uv run agent-naming --quiet           # Suppress stdout
    """
    return _run_script("naming_inventory.py")


def health() -> int:
    """Aggregate health checks (lint/types/tests/preflight/config).

    Entry point: agent-health

    Examples:
        uv run agent-health
        uv run agent-health --format json --output var/agents/health/health_report.json
        uv run agent-health --pytest-args -q tests/unit
    """
    return _run_script("health_report.py")


def regenerate() -> int:
    """Regenerate all static context files.

    Entry point: agent-regenerate

    Regenerates all files in var/agents/ by running all generator scripts.

    Examples:
        uv run agent-regenerate               # Regenerate all
        uv run agent-regenerate --verify      # Check freshness only
    """
    return _run_script("regenerate_all.py")
