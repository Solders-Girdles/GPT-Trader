#!/usr/bin/env python3
"""Check per-package coverage thresholds.

This script parses pytest-cov output and validates per-package coverage meets
the defined thresholds for each phase of the coverage improvement plan.

Usage:
    # Check Phase 0 modules (security, config, cli)
    pytest --cov=src/bot_v2 --cov-report=json -q
    python scripts/check_package_coverage.py phase0

    # Check specific phase from environment variable (CI)
    COVERAGE_PHASE=phase1 python scripts/check_package_coverage.py

    # Just show current coverage for all tracked packages
    python scripts/check_package_coverage.py --show-all
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict

# Phase-specific thresholds
PHASE_THRESHOLDS = {
    "phase0": {
        "src/bot_v2/security": 88,
        "src/bot_v2/config": 86,
        "src/bot_v2/cli": 93,
    },
    "phase1": {
        "src/bot_v2/features/brokerages": 92,
        "src/bot_v2/monitoring": 92,
    },
    "phase2": {
        "src/bot_v2/orchestration": 97,
        "src/bot_v2/state": 97,
        "src/bot_v2/features/live_trade": 97,
    },
    "phase3": {
        # All packages should be at 100%
        "*": 100,
    },
}


def load_coverage_data(coverage_file: str = "coverage.json") -> dict:
    """Load coverage data from JSON file."""
    if not Path(coverage_file).exists():
        print(f"❌ Coverage file '{coverage_file}' not found.")
        print("Run: pytest --cov=src/bot_v2 --cov-report=json")
        sys.exit(1)

    with open(coverage_file) as f:
        return json.load(f)


def calculate_package_coverage(coverage_data: dict, package_prefix: str) -> float:
    """Calculate coverage percentage for a package prefix."""
    total_statements = 0
    covered_statements = 0

    files_data = coverage_data.get("files", {})
    for file_path, file_data in files_data.items():
        if file_path.startswith(package_prefix):
            summary = file_data.get("summary", {})
            total_statements += summary.get("num_statements", 0)
            covered_statements += summary.get("covered_lines", 0)

    if total_statements == 0:
        return 0.0

    return (covered_statements / total_statements) * 100


def check_phase_thresholds(coverage_data: dict, phase: str) -> bool:
    """Check if all packages in a phase meet their thresholds."""
    thresholds = PHASE_THRESHOLDS.get(phase, {})
    if not thresholds:
        print(f"❌ Unknown phase: {phase}")
        print(f"Valid phases: {', '.join(PHASE_THRESHOLDS.keys())}")
        return False

    print(f"\n📊 Phase {phase.upper()} Coverage Check")
    print("=" * 60)

    all_passing = True
    for package, threshold in thresholds.items():
        if package == "*":
            # Global threshold - check overall coverage
            overall = coverage_data.get("totals", {}).get("percent_covered", 0)
            status = "✅" if overall >= threshold else "❌"
            print(f"{status} Overall: {overall:.1f}% (threshold: {threshold}%)")
            if overall < threshold:
                all_passing = False
        else:
            coverage_pct = calculate_package_coverage(coverage_data, package)
            status = "✅" if coverage_pct >= threshold else "❌"
            package_name = package.replace("src/bot_v2/", "")
            print(f"{status} {package_name:30} {coverage_pct:5.1f}% (threshold: {threshold}%)")
            if coverage_pct < threshold:
                all_passing = False

    print("=" * 60)
    if all_passing:
        print("✅ All thresholds met!")
        return True
    else:
        print("❌ Some thresholds not met")
        return False


def show_all_packages(coverage_data: dict) -> None:
    """Show coverage for all tracked packages."""
    print("\n📊 Coverage for All Tracked Packages")
    print("=" * 60)

    all_packages = set()
    for phase_thresholds in PHASE_THRESHOLDS.values():
        all_packages.update(pkg for pkg in phase_thresholds.keys() if pkg != "*")

    for package in sorted(all_packages):
        coverage_pct = calculate_package_coverage(coverage_data, package)
        package_name = package.replace("src/bot_v2/", "")
        print(f"{package_name:30} {coverage_pct:5.1f}%")

    overall = coverage_data.get("totals", {}).get("percent_covered", 0)
    print("=" * 60)
    print(f"{'Overall':30} {overall:5.1f}%")


def main() -> int:
    """Main entry point."""
    # Check for --show-all flag
    if "--show-all" in sys.argv:
        coverage_data = load_coverage_data()
        show_all_packages(coverage_data)
        return 0

    # Determine phase from args or environment
    phase = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        phase = sys.argv[1]
    else:
        phase = os.environ.get("COVERAGE_PHASE", "phase0")

    # Load and check coverage
    coverage_data = load_coverage_data()
    success = check_phase_thresholds(coverage_data, phase)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
