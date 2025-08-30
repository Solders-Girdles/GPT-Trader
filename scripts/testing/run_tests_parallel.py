#!/usr/bin/env python
"""
Script to run tests with parallel execution and proper configuration.
"""

import argparse
import subprocess
import sys


def run_tests(args):
    """Run pytest with parallel execution and specified configuration."""

    # Base command
    cmd = ["poetry", "run", "pytest"]

    # Add test path
    if args.path:
        cmd.append(args.path)
    else:
        cmd.append("tests/")

    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", str(args.workers)])

    # Add markers
    if args.marker:
        cmd.extend(["-m", args.marker])

    # Add verbose output
    if args.verbose:
        cmd.append("-vv")

    # Add coverage options if not doing performance tests
    if not args.no_coverage:
        cmd.extend(
            [
                "--cov=src",
                "--cov-report=term-missing",
                f"--cov-fail-under={args.coverage_threshold}",
            ]
        )

    # Add other options
    if args.fast:
        cmd.extend(["-x", "--tb=no"])  # Stop on first failure, no traceback

    if args.debug:
        cmd.extend(["-s", "--capture=no"])  # Show print statements

    # Run the command
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run tests with parallel execution")

    parser.add_argument("path", nargs="?", help="Path to test file or directory (default: tests/)")

    parser.add_argument(
        "-p", "--parallel", action="store_true", help="Enable parallel test execution"
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=str,
        default="auto",
        help="Number of parallel workers (default: auto)",
    )

    parser.add_argument(
        "-m", "--marker", help="Run tests with specific marker (e.g., 'unit', 'integration')"
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")

    parser.add_argument(
        "--coverage-threshold",
        type=int,
        default=80,
        help="Coverage threshold percentage (default: 80)",
    )

    parser.add_argument("--fast", action="store_true", help="Fast mode - stop on first failure")

    parser.add_argument("--debug", action="store_true", help="Debug mode - show print statements")

    args = parser.parse_args()

    # Run tests
    exit_code = run_tests(args)

    # Exit with the same code as pytest
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
