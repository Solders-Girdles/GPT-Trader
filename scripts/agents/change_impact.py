#!/usr/bin/env python3
"""Analyze change impact and suggest relevant tests.

Given a set of changed files, this tool:
- Identifies affected modules and components
- Suggests relevant tests to run
- Estimates impact scope (low/medium/high)

Usage:
    python scripts/agents/change_impact.py [--files FILE...] [--from-git]
    python scripts/agents/change_impact.py --from-git --base main
    python scripts/agents/change_impact.py --files src/gpt_trader/cli/commands/orders.py

Output:
    JSON report with suggested tests and impact analysis.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
SOURCE_TEST_MAP_PATH = PROJECT_ROOT / "var" / "agents" / "testing" / "source_test_map.json"

# Mapping from source paths to test paths
PATH_TO_TEST_MAPPING = {
    "src/gpt_trader/cli/": "tests/unit/gpt_trader/cli/",
    "src/gpt_trader/features/brokerages/": "tests/unit/gpt_trader/features/brokerages/",
    "src/gpt_trader/features/backtesting/": "tests/unit/gpt_trader/features/backtesting/",
    "src/gpt_trader/features/optimize/": "tests/unit/gpt_trader/features/optimize/",
    "src/gpt_trader/features/live_trade/": "tests/unit/gpt_trader/features/live_trade/",
    "src/gpt_trader/app/": "tests/unit/gpt_trader/app/",
    "src/gpt_trader/validation/": "tests/unit/gpt_trader/validation/",
    "src/gpt_trader/config/": "tests/unit/gpt_trader/config/",
    "src/gpt_trader/errors/": "tests/unit/gpt_trader/errors/",
    "src/gpt_trader/monitoring/": "tests/unit/gpt_trader/monitoring/",
    "src/gpt_trader/logging/": "tests/unit/gpt_trader/logging/",
    "src/gpt_trader/security/": "tests/unit/gpt_trader/security/",
    "src/gpt_trader/utilities/": "tests/unit/gpt_trader/utilities/",
}

# Component markers for test selection
COMPONENT_MARKERS = {
    "cli": ["cli"],
    "brokerages": ["api", "endpoints"],
    "backtesting": ["backtesting"],
    "optimize": ["backtesting"],
    "live_trade": ["execution", "perps", "spot"],
    "app": ["config"],
    "validation": [],
    "config": [],
    "errors": [],
    "monitoring": ["monitoring"],
    "security": ["security"],
    "risk": ["risk"],
}

# High-impact files that warrant broader testing
HIGH_IMPACT_PATTERNS = [
    r"src/gpt_trader/errors/",
    r"src/gpt_trader/config/",
    r"src/gpt_trader/app/config/",
    r"src/gpt_trader/features/brokerages/core/",
    r"pyproject\.toml$",
    r"pytest\.ini$",
]

# Critical files that should trigger full test suite
CRITICAL_PATTERNS = [
    r"src/gpt_trader/__init__\.py$",
    r"src/gpt_trader/cli/__init__\.py$",
]


def get_changed_files_from_git(base: str = "main") -> list[str]:
    """Get list of changed files from git diff."""
    try:
        # Get staged changes
        result = subprocess.run(
            ["git", "diff", "--name-only", "--cached"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        staged = set(result.stdout.strip().split("\n")) if result.stdout.strip() else set()

        # Get unstaged changes
        result = subprocess.run(
            ["git", "diff", "--name-only"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        unstaged = set(result.stdout.strip().split("\n")) if result.stdout.strip() else set()

        # Get changes compared to base branch
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{base}...HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        branch_changes = set(result.stdout.strip().split("\n")) if result.stdout.strip() else set()

        all_changes = staged | unstaged | branch_changes
        return [f for f in all_changes if f and f.endswith(".py")]

    except Exception:
        return []


def module_from_path(source_file: str) -> str | None:
    """Convert a source path to a module name."""
    if not source_file.startswith("src/"):
        return None
    rel_path = source_file[len("src/") :]
    if not rel_path.endswith(".py"):
        return None
    module = rel_path[:-3].replace("/", ".")
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    return module


def iter_module_prefixes(module: str) -> list[str]:
    """Return module and its parent prefixes for lookup."""
    parts = module.split(".")
    return [".".join(parts[:i]) for i in range(len(parts), 0, -1)]


def build_pytest_command(paths: list[str], markers: list[str]) -> str:
    """Build a pytest command from paths and optional markers."""
    if not paths:
        marker_str = " or ".join(markers) if markers else ""
        if marker_str:
            return f"uv run pytest tests/unit -m '{marker_str}'"
        return "uv run pytest tests/unit -m 'not slow'"

    return f"uv run pytest {' '.join(paths[:5])}"


def load_source_test_map() -> dict[str, Any] | None:
    """Load the source/test map if available."""
    if not SOURCE_TEST_MAP_PATH.exists():
        return None
    try:
        return json.loads(SOURCE_TEST_MAP_PATH.read_text())
    except Exception:
        return None


def find_test_file(source_file: str) -> str | None:
    """Find the corresponding test file for a source file."""
    # Direct mapping based on path prefixes
    for src_prefix, test_prefix in PATH_TO_TEST_MAPPING.items():
        if source_file.startswith(src_prefix):
            relative = source_file[len(src_prefix) :]
            # Convert source.py to test_source.py
            if relative.endswith(".py"):
                parts = Path(relative).parts
                if parts:
                    test_name = f"test_{parts[-1]}"
                    test_path = Path(test_prefix) / Path(*parts[:-1]) / test_name
                    if (PROJECT_ROOT / test_path).exists():
                        return str(test_path)

                    # Try without nested structure
                    test_path = Path(test_prefix) / test_name
                    if (PROJECT_ROOT / test_path).exists():
                        return str(test_path)

    return None


def find_test_directory(source_file: str) -> str | None:
    """Find the test directory for a source path."""
    for src_prefix, test_prefix in PATH_TO_TEST_MAPPING.items():
        if source_file.startswith(src_prefix):
            test_dir = PROJECT_ROOT / test_prefix
            if test_dir.exists():
                return test_prefix.rstrip("/")
    return None


def get_component(source_file: str) -> str | None:
    """Extract component name from file path."""
    # Match patterns like src/gpt_trader/features/X/ or src/gpt_trader/X/
    match = re.search(r"src/gpt_trader/(?:features/)?(\w+)/", source_file)
    if match:
        return match.group(1)
    return None


def analyze_impact(
    changed_files: list[str],
    importer_files: list[str] | None = None,
) -> dict[str, Any]:
    """Analyze the impact of changed files."""
    result: dict[str, Any] = {
        "changed_files": changed_files,
        "impact_level": "low",
        "suggested_tests": [],
        "suggested_markers": [],
        "test_directories": [],
        "affected_components": [],
        "reasons": [],
    }

    if not changed_files:
        result["reasons"].append("No changed files detected")
        return result

    test_files: set[str] = set()
    test_dirs: set[str] = set()
    markers: set[str] = set()
    components: set[str] = set()
    is_high_impact = False
    is_critical = False
    source_test_map = load_source_test_map()
    source_test_map_used = False

    for file in changed_files:
        # Skip test files
        if file.startswith("tests/"):
            test_files.add(file)
            continue

        # Check for critical files
        for pattern in CRITICAL_PATTERNS:
            if re.search(pattern, file):
                is_critical = True
                result["reasons"].append(f"Critical file changed: {file}")
                break

        # Check for high-impact files
        for pattern in HIGH_IMPACT_PATTERNS:
            if re.search(pattern, file):
                is_high_impact = True
                result["reasons"].append(f"High-impact file changed: {file}")
                break

        # Find corresponding tests
        test_file = find_test_file(file)
        if test_file:
            test_files.add(test_file)

        test_dir = find_test_directory(file)
        if test_dir:
            test_dirs.add(test_dir)

        # Get component and markers
        component = get_component(file)
        if component:
            components.add(component)
            if component in COMPONENT_MARKERS:
                markers.update(COMPONENT_MARKERS[component])

        module = module_from_path(file)
        if module and source_test_map:
            for prefix in iter_module_prefixes(module):
                tests = source_test_map.get("source_to_tests", {}).get(prefix)
                if tests:
                    test_files.update(tests)
                    source_test_map_used = True

    if importer_files:
        for importer in importer_files:
            if importer.startswith("tests/"):
                test_files.add(importer)
                continue

            importer_test = find_test_file(importer)
            if importer_test:
                test_files.add(importer_test)

            importer_dir = find_test_directory(importer)
            if importer_dir:
                test_dirs.add(importer_dir)

            importer_module = module_from_path(importer)
            if importer_module and source_test_map:
                for prefix in iter_module_prefixes(importer_module):
                    tests = source_test_map.get("source_to_tests", {}).get(prefix)
                    if tests:
                        test_files.update(tests)
                        source_test_map_used = True
        result["reasons"].append("Added tests for importer modules")
        result["importer_tests_added"] = True

    # Determine impact level
    if is_critical:
        result["impact_level"] = "critical"
        result["reasons"].append("Recommend running full test suite")
        test_dirs.add("tests/unit")
    elif is_high_impact or len(components) > 2:
        result["impact_level"] = "high"
        result["reasons"].append("Changes affect core components")
    elif len(changed_files) > 5:
        result["impact_level"] = "medium"
    else:
        result["impact_level"] = "low"

    # Build test suggestions (filter out pytest conftest helpers).
    test_files = {path for path in test_files if not path.endswith("/conftest.py")}
    result["suggested_tests"] = sorted(test_files)
    result["test_directories"] = sorted(test_dirs)
    result["suggested_markers"] = sorted(markers)
    result["affected_components"] = sorted(components)
    if source_test_map_used:
        result["reasons"].append("Added tests from source/test import map")
        result["source_test_map_used"] = True

    # Generate pytest commands
    paths = list(test_files) + list(test_dirs)
    result["pytest_command"] = build_pytest_command(paths, sorted(markers))

    if test_files:
        result["pytest_command_files_only"] = build_pytest_command(
            list(test_files), sorted(markers)
        )

    return result


def find_importers(module_path: str, source_dir: Path) -> list[str]:
    """Find files that import the given module."""
    importers = []
    module_name = module_path.replace("/", ".").replace(".py", "")
    module_name = module_name.replace("src.", "")

    # Simple patterns to find imports
    patterns = [
        f"from {module_name} import",
        f"import {module_name}",
        f'from {module_name.rsplit(".", 1)[0]} import',
    ]

    for py_file in source_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        try:
            content = py_file.read_text()
            for pattern in patterns:
                if pattern in content:
                    importers.append(str(py_file.relative_to(PROJECT_ROOT)))
                    break
        except Exception:
            continue

    return importers


def format_text_report(analysis: dict[str, Any]) -> str:
    """Format analysis as human-readable text."""
    lines = [
        "Change Impact Analysis",
        "=" * 50,
        f"Impact Level: {analysis['impact_level'].upper()}",
        f"Changed Files: {len(analysis['changed_files'])}",
        f"Affected Components: {', '.join(analysis['affected_components']) or 'None'}",
        "",
    ]

    if analysis["reasons"]:
        lines.append("Reasons:")
        for reason in analysis["reasons"]:
            lines.append(f"  - {reason}")
        lines.append("")

    if analysis["suggested_tests"]:
        lines.append(f"Suggested Test Files ({len(analysis['suggested_tests'])}):")
        for test in analysis["suggested_tests"][:10]:
            lines.append(f"  - {test}")
        if len(analysis["suggested_tests"]) > 10:
            lines.append(f"  ... and {len(analysis['suggested_tests']) - 10} more")
        lines.append("")

    if analysis["test_directories"]:
        lines.append("Test Directories:")
        for dir in analysis["test_directories"]:
            lines.append(f"  - {dir}")
        lines.append("")

    if analysis["suggested_markers"]:
        lines.append(f"Suggested Markers: {', '.join(analysis['suggested_markers'])}")
        lines.append("")

    lines.append("Recommended Command:")
    lines.append(f"  {analysis.get('pytest_command', 'uv run pytest tests/unit')}")

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze change impact and suggest tests")
    parser.add_argument(
        "--files",
        nargs="+",
        help="Changed files to analyze",
    )
    parser.add_argument(
        "--from-git",
        action="store_true",
        help="Get changed files from git",
    )
    parser.add_argument(
        "--base",
        type=str,
        default="main",
        help="Base branch for git diff (default: main)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format",
    )
    parser.add_argument(
        "--include-importers",
        action="store_true",
        help="Include files that import changed modules",
    )
    parser.add_argument(
        "--source-files",
        action="store_true",
        help="Prefer test file-only suggestions in the recommended command",
    )
    parser.add_argument(
        "--exclude-integration",
        action="store_true",
        help="Exclude integration tests from suggestions",
    )

    args = parser.parse_args()

    # Get changed files
    if args.from_git:
        changed_files = get_changed_files_from_git(args.base)
        print(f"Found {len(changed_files)} changed Python files from git", file=sys.stderr)
    elif args.files:
        changed_files = args.files
    else:
        # Default to git changes
        changed_files = get_changed_files_from_git(args.base)
        if not changed_files:
            print("No changed files specified. Use --files or --from-git", file=sys.stderr)
            return 1

    importer_files: list[str] = []
    if args.include_importers:
        all_importers: set[str] = set()
        for file in changed_files:
            if file.startswith("src/"):
                importers = find_importers(file, PROJECT_ROOT / "src")
                all_importers.update(importers)
        importer_files = sorted(all_importers)

    # Analyze impact
    analysis = analyze_impact(changed_files, importer_files=importer_files)

    if args.exclude_integration:
        integration_prefix = "tests/integration"
        analysis["suggested_tests"] = [
            test
            for test in analysis["suggested_tests"]
            if not test.startswith(f"{integration_prefix}/")
        ]
        analysis["test_directories"] = [
            test_dir for test_dir in analysis["test_directories"] if test_dir != integration_prefix
        ]
        analysis["reasons"].append("Excluded integration tests")
        analysis["excluded_integration"] = True
        analysis["pytest_command"] = build_pytest_command(
            analysis.get("suggested_tests", []) + analysis.get("test_directories", []),
            analysis.get("suggested_markers", []),
        )
        if analysis.get("suggested_tests"):
            analysis["pytest_command_files_only"] = build_pytest_command(
                analysis.get("suggested_tests", []),
                analysis.get("suggested_markers", []),
            )
            if analysis.get("files_only"):
                analysis["pytest_command"] = analysis["pytest_command_files_only"]

    if args.source_files:
        files_only_command = analysis.get("pytest_command_files_only")
        if files_only_command:
            analysis["pytest_command"] = files_only_command
            analysis["test_directories"] = []
            analysis["reasons"].append("File-only test suggestions requested")
            analysis["files_only"] = True

    if args.include_importers:
        analysis["importing_files"] = importer_files

    # Output
    if args.format == "text":
        print(format_text_report(analysis))
    else:
        print(json.dumps(analysis, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
