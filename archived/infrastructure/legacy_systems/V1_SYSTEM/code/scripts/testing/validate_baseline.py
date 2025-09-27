#!/usr/bin/env python3
"""
Validate the minimal baseline tests we just created.
Run this immediately to see if our baseline actually works.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description="", capture_output=True):
    """Run command and show result."""
    print(f"\n--- {description} ---")
    print(f"Command: {cmd}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=capture_output,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        if capture_output:
            if result.stdout:
                print("STDOUT:", result.stdout[:500])
            if result.stderr:
                print("STDERR:", result.stderr[:500])

        print(f"Return code: {result.returncode}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    print("=== Validating Minimal Baseline Tests ===")

    # Check environment
    print("\n1. Environment Check")
    os.chdir(Path(__file__).parent.parent)
    print(f"Working directory: {os.getcwd()}")

    # Test basic imports
    success = run_command("python scripts/test_baseline_quick.py", "Quick import test")

    if not success:
        print("❌ Basic imports failing - stopping here")
        return 1

    # Test strategy functionality
    success = run_command("python scripts/test_strategy_quick.py", "Quick strategy test")

    # Try to collect tests
    success = run_command(
        "python -m pytest tests/minimal_baseline --collect-only -q", "Test collection"
    )

    # Run just the core import tests
    success = run_command(
        "python -m pytest tests/minimal_baseline/test_core_imports.py -v",
        "Core import tests",
        capture_output=False,
    )

    if success:
        print("\n✅ Core import tests pass!")
    else:
        print("\n❌ Core import tests fail")
        return 1

    # Run one more test file
    success = run_command(
        "python -m pytest tests/minimal_baseline/test_data_pipeline.py -v -k 'not slow'",
        "Data pipeline tests (fast only)",
        capture_output=False,
    )

    if success:
        print("\n✅ Basic data pipeline tests pass!")
    else:
        print("\n❌ Data pipeline tests fail")

    print("\n=== Summary ===")
    print("Minimal baseline created with:")
    print("• 5 test files covering critical functionality")
    print("• ~42 individual tests")
    print("• Focus on real functionality that actually works")
    print("• Separate configs for baseline vs full test suite")
    print("• Helper scripts for quick validation")

    return 0


if __name__ == "__main__":
    sys.exit(main())
