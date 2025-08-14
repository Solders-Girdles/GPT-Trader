#!/usr/bin/env python3
"""
Test runner script for Phase 5 Production Integration System.
Provides easy access to run different types of tests with useful output.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Optional


def run_command(command: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(command, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False


def run_unit_tests(verbose: bool = False, coverage: bool = True) -> bool:
    """Run unit tests."""
    command = ["pytest", "tests/unit/"]

    if verbose:
        command.append("-v")

    if coverage:
        command.extend(
            [
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/unit",
                "--cov-fail-under=80",
            ]
        )

    return run_command(command, "Unit Tests")


def run_integration_tests(verbose: bool = False, coverage: bool = True) -> bool:
    """Run integration tests."""
    command = ["pytest", "tests/integration/"]

    if verbose:
        command.append("-v")

    if coverage:
        command.extend(
            [
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/integration",
                "--cov-append",
            ]
        )

    return run_command(command, "Integration Tests")


def run_system_tests(verbose: bool = False, coverage: bool = True) -> bool:
    """Run system tests."""
    command = ["pytest", "tests/system/"]

    if verbose:
        command.append("-v")

    if coverage:
        command.extend(
            [
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/system",
                "--cov-append",
            ]
        )

    return run_command(command, "System Tests")


def run_performance_tests(verbose: bool = False, benchmark: bool = True) -> bool:
    """Run performance tests."""
    command = ["pytest", "tests/performance/"]

    if verbose:
        command.append("-v")

    if benchmark:
        command.extend(["--benchmark-only", "--benchmark-sort=mean", "--benchmark-min-rounds=5"])

    return run_command(command, "Performance Tests")


def run_acceptance_tests(verbose: bool = False, coverage: bool = True) -> bool:
    """Run acceptance tests."""
    command = ["pytest", "tests/acceptance/"]

    if verbose:
        command.append("-v")

    if coverage:
        command.extend(
            [
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/acceptance",
                "--cov-append",
            ]
        )

    return run_command(command, "Acceptance Tests")


def run_production_tests(verbose: bool = False, coverage: bool = True) -> bool:
    """Run production readiness tests."""
    command = ["pytest", "tests/production/"]

    if verbose:
        command.append("-v")

    if coverage:
        command.extend(
            [
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/production",
                "--cov-append",
            ]
        )

    return run_command(command, "Production Readiness Tests")


def run_code_quality_checks() -> bool:
    """Run code quality checks."""
    checks = [
        (["flake8", "src/", "tests/"], "Flake8 Linting"),
        (["black", "--check", "--diff", "src/", "tests/"], "Black Code Formatting"),
        (["isort", "--check-only", "--diff", "src/", "tests/"], "Import Sorting"),
        (["mypy", "src/", "--ignore-missing-imports"], "Type Checking"),
        (["bandit", "-r", "src/", "-f", "json", "-o", "bandit-report.json"], "Security Check"),
    ]

    all_passed = True
    for command, description in checks:
        if not run_command(command, description):
            all_passed = False

    return all_passed


def run_documentation_tests() -> bool:
    """Run documentation tests."""
    command = [
        "python",
        "-m",
        "doctest",
        "docs/NEXT_STEPS_ROADMAP.md",
        "docs/TESTING_ITERATION_ROADMAP.md",
        "examples/phase5_production_integration_example.py",
    ]

    return run_command(command, "Documentation Tests")


def run_all_tests(verbose: bool = False, coverage: bool = True) -> bool:
    """Run all tests in sequence."""
    print("🚀 Starting comprehensive test suite for Phase 5 Production Integration System")

    test_results = []

    # Unit tests
    test_results.append(("Unit Tests", run_unit_tests(verbose, coverage)))

    # Integration tests
    test_results.append(("Integration Tests", run_integration_tests(verbose, coverage)))

    # System tests
    test_results.append(("System Tests", run_system_tests(verbose, coverage)))

    # Performance tests
    test_results.append(("Performance Tests", run_performance_tests(verbose, True)))

    # Code quality checks
    test_results.append(("Code Quality Checks", run_code_quality_checks()))

    # Documentation tests
    test_results.append(("Documentation Tests", run_documentation_tests()))

    # Acceptance tests
    test_results.append(("Acceptance Tests", run_acceptance_tests(verbose, coverage)))

    # Production tests
    test_results.append(("Production Readiness Tests", run_production_tests(verbose, coverage)))

    # Summary
    print(f"\n{'='*80}")
    print("📊 TEST SUMMARY")
    print(f"{'='*80}")

    passed = 0
    failed = 0

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {len(test_results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if coverage:
        print(f"\n📈 Coverage reports available in htmlcov/ directory")

    if failed == 0:
        print(f"\n🎉 All tests passed! The system is ready for production.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review and fix issues before proceeding.")
        return False


def run_smoke_tests() -> bool:
    """Run a quick smoke test suite."""
    print("🔥 Running smoke tests for Phase 5 Production Integration System")

    # Quick unit tests for core components
    command = [
        "pytest",
        "tests/unit/test_strategy_selector.py::TestStrategySelector::test_initialization",
        "tests/unit/test_strategy_selector.py::TestSelectionConfig::test_default_config",
        "-v",
    ]

    return run_command(command, "Smoke Tests")


def run_regression_tests() -> bool:
    """Run regression tests."""
    print("🔄 Running regression tests for Phase 5 Production Integration System")

    command = ["pytest", "-m", "regression", "-v", "--tb=short"]

    return run_command(command, "Regression Tests")


def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(
        description="Test runner for Phase 5 Production Integration System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_tests.py --all                    # Run all tests
  python scripts/run_tests.py --unit --verbose         # Run unit tests with verbose output
  python scripts/run_tests.py --integration            # Run integration tests
  python scripts/run_tests.py --system                 # Run system tests
  python scripts/run_tests.py --performance            # Run performance tests
  python scripts/run_tests.py --acceptance             # Run acceptance tests
  python scripts/run_tests.py --production             # Run production readiness tests
  python scripts/run_tests.py --quality                # Run code quality checks
  python scripts/run_tests.py --docs                   # Run documentation tests
  python scripts/run_tests.py --smoke                  # Run smoke tests
  python scripts/run_tests.py --regression             # Run regression tests
        """,
    )

    # Test type arguments
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--system", action="store_true", help="Run system tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--acceptance", action="store_true", help="Run acceptance tests")
    parser.add_argument("--production", action="store_true", help="Run production readiness tests")
    parser.add_argument("--quality", action="store_true", help="Run code quality checks")
    parser.add_argument("--docs", action="store_true", help="Run documentation tests")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests")
    parser.add_argument("--regression", action="store_true", help="Run regression tests")

    # Output options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--no-benchmark", action="store_true", help="Disable benchmark reporting")

    args = parser.parse_args()

    # Set coverage and benchmark flags
    coverage = not args.no_coverage
    benchmark = not args.no_benchmark

    # Check if any test type is specified
    test_types = [
        args.all,
        args.unit,
        args.integration,
        args.system,
        args.performance,
        args.acceptance,
        args.production,
        args.quality,
        args.docs,
        args.smoke,
        args.regression,
    ]

    if not any(test_types):
        parser.print_help()
        return 1

    # Run tests based on arguments
    success = True

    if args.all:
        success = run_all_tests(args.verbose, coverage)
    else:
        if args.unit:
            success &= run_unit_tests(args.verbose, coverage)

        if args.integration:
            success &= run_integration_tests(args.verbose, coverage)

        if args.system:
            success &= run_system_tests(args.verbose, coverage)

        if args.performance:
            success &= run_performance_tests(args.verbose, benchmark)

        if args.acceptance:
            success &= run_acceptance_tests(args.verbose, coverage)

        if args.production:
            success &= run_production_tests(args.verbose, coverage)

        if args.quality:
            success &= run_code_quality_checks()

        if args.docs:
            success &= run_documentation_tests()

        if args.smoke:
            success &= run_smoke_tests()

        if args.regression:
            success &= run_regression_tests()

    # Exit with appropriate code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
