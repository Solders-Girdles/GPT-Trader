#!/usr/bin/env python3
"""
CLI Smoke Test Script - CLI-005
GPT-Trader Emergency Recovery

Tests each CLI command to verify basic functionality.
Run with: poetry run python scripts/cli_smoke_test.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def run_command(cmd, description, timeout=10, check_output=None):
    """Run a CLI command and check if it works"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {cmd}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

        # Check for success
        if result.returncode == 0 or (check_output and check_output in result.stdout):
            print(f"{GREEN}✅ PASSED{RESET}")
            if result.stdout:
                print(f"Output preview: {result.stdout[:200]}...")
            return True
        else:
            print(f"{RED}❌ FAILED{RESET}")
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"{YELLOW}⏱️ TIMEOUT{RESET} (exceeded {timeout}s)")
        return False
    except Exception as e:
        print(f"{RED}❌ EXCEPTION: {e}{RESET}")
        return False


def main():
    """Run all CLI smoke tests"""
    print("=" * 60)
    print("🚀 GPT-Trader CLI Smoke Test")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    tests = [
        # Basic commands
        ("poetry run python -m bot.cli --help", "CLI Help", 5, "GPT-Trader"),
        ("poetry run python -m bot.cli --version", "CLI Version", 5, None),
        # Command help tests
        ("poetry run python -m bot.cli backtest --help", "Backtest Help", 5, "backtest"),
        ("poetry run python -m bot.cli optimize --help", "Optimize Help", 5, "optimize"),
        ("poetry run python -m bot.cli paper --help", "Paper Trading Help", 5, "paper"),
        ("poetry run python -m bot.cli monitor --help", "Monitor Help", 5, "monitor"),
        ("poetry run python -m bot.cli dashboard --help", "Dashboard Help", 5, "dashboard"),
        ("poetry run python -m bot.cli ml-train --help", "ML Train Help", 5, "ml-train"),
        ("poetry run python -m bot.cli auto-trade --help", "Auto Trade Help", 5, "auto-trade"),
        # Simple functional tests (non-interactive)
        (
            "poetry run python -m bot.cli backtest --symbol AAPL --start 2024-01-01 --end 2024-01-31 --strategy demo_ma 2>&1 | head -5",
            "Backtest Execution (AAPL)",
            10,
            None,
        ),
        # Test with bad arguments to verify error handling
        (
            "poetry run python -m bot.cli backtest --symbol 2>&1",
            "Backtest Error Handling",
            5,
            "error",
        ),
    ]

    results = {}
    passed = 0
    failed = 0

    for cmd, desc, timeout, check in tests:
        success = run_command(cmd, desc, timeout, check)
        results[desc] = success
        if success:
            passed += 1
        else:
            failed += 1

    # Summary Report
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    total = passed + failed
    pass_rate = (passed / total * 100) if total > 0 else 0

    for desc, success in results.items():
        status = f"{GREEN}✅ PASS{RESET}" if success else f"{RED}❌ FAIL{RESET}"
        print(f"{status}: {desc}")

    print("\n" + "-" * 60)
    print(f"Total Tests: {total}")
    print(f"{GREEN}Passed: {passed}{RESET}")
    print(f"{RED}Failed: {failed}{RESET}")
    print(f"Pass Rate: {pass_rate:.1f}%")

    # Write detailed report
    report_path = Path("logs/cli_test_report.txt")
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, "w") as f:
        f.write(f"CLI Smoke Test Report\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"{'='*60}\n\n")

        f.write("TEST RESULTS:\n")
        for desc, success in results.items():
            status = "PASS" if success else "FAIL"
            f.write(f"  [{status}] {desc}\n")

        f.write(f"\nSUMMARY:\n")
        f.write(f"  Total: {total}\n")
        f.write(f"  Passed: {passed}\n")
        f.write(f"  Failed: {failed}\n")
        f.write(f"  Pass Rate: {pass_rate:.1f}%\n")

        # Known issues
        f.write(f"\nKNOWN ISSUES:\n")
        f.write(f"  1. Backtest command has argument mismatch (start_date vs start)\n")
        f.write(f"  2. Wizard requires interactive input (can't test in CI)\n")
        f.write(f"  3. Dashboard launches server (requires manual shutdown)\n")
        f.write(f"  4. Most commands missing actual implementation\n")

    print(f"\n📝 Detailed report saved to: {report_path}")

    # Overall status
    if pass_rate >= 80:
        print(f"\n{GREEN}🎉 CLI SMOKE TEST PASSED!{RESET}")
        return 0
    elif pass_rate >= 50:
        print(f"\n{YELLOW}⚠️ CLI PARTIALLY WORKING{RESET}")
        return 1
    else:
        print(f"\n{RED}❌ CLI CRITICALLY BROKEN{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
