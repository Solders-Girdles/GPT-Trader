#!/usr/bin/env python3
"""
TEST-004: Minimal Test Baseline Runner

Runs the critical 20-30 tests that must pass for minimal functionality.
Provides clear pass/fail summary and identifies what actually works.

Usage:
    python scripts/run_baseline_tests.py
    python scripts/run_baseline_tests.py --slow  # Include slow tests
    python scripts/run_baseline_tests.py --coverage  # Generate coverage report
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, capture_output=True):
    """Run a command and return result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=capture_output,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        return result
    except Exception as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e}")
        return None


def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_section(title):
    """Print a formatted section."""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")


def main():
    parser = argparse.ArgumentParser(description="Run minimal test baseline")
    parser.add_argument("--slow", action="store_true", help="Include slow tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print_header("GPT-Trader Minimal Test Baseline")
    print("Testing critical functionality that MUST work...")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check if we can import pytest
    print_section("Environment Check")
    result = run_command("python -c 'import pytest; print(pytest.__version__)'")
    if result and result.returncode == 0:
        print(f"✓ pytest available: {result.stdout.strip()}")
    else:
        print("✗ pytest not available")
        sys.exit(1)

    # Check basic imports
    print_section("Basic Import Test")
    import_test = run_command(
        'python -c \'import sys; sys.path.insert(0, "src"); from bot.config import get_config; print("Core imports OK")\''
    )
    if import_test and import_test.returncode == 0:
        print("✓ Core imports working")
    else:
        print("✗ Core imports failing")
        if import_test:
            print(f"Error: {import_test.stderr}")

    # Build pytest command
    pytest_cmd = "python -m pytest tests/minimal_baseline"

    if args.coverage:
        pytest_cmd += " --cov=src/bot --cov-report=term-missing --cov-report=html"

    if args.verbose:
        pytest_cmd += " -v"
    else:
        pytest_cmd += " -q"

    if not args.slow:
        pytest_cmd += " -m 'not slow'"

    # Add configuration
    pytest_cmd += " -c pytest_baseline.ini"

    print_section("Running Baseline Tests")
    print(f"Command: {pytest_cmd}")

    start_time = time.time()
    result = run_command(pytest_cmd, capture_output=False)
    end_time = time.time()

    print_section("Results Summary")
    print(f"Test execution time: {end_time - start_time:.2f} seconds")

    if result:
        if result.returncode == 0:
            print("✓ ALL BASELINE TESTS PASSED")
            print("✓ System has minimal functionality")
        else:
            print("✗ Some baseline tests failed")
            print("✗ System needs fixes before it's minimally functional")

        # Try to extract test counts from output
        print_section("Test Statistics")
        stats_result = run_command(f"{pytest_cmd} --collect-only -q")
        if stats_result and stats_result.returncode == 0:
            lines = stats_result.stdout.split("\n")
            for line in lines:
                if "collected" in line:
                    print(f"Tests collected: {line.strip()}")

        return result.returncode
    else:
        print("✗ Failed to run tests")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
