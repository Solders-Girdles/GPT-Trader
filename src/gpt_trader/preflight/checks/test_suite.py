from __future__ import annotations

import re
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpt_trader.preflight.core import PreflightCheck


def check_test_suite(checker: PreflightCheck) -> bool:
    """Run a targeted subset of the test suite."""
    checker.section_header("7. TEST SUITE VALIDATION")

    print("Running core test suite...")

    runner = "uv" if shutil.which("uv") else "poetry" if shutil.which("poetry") else None
    if not runner:
        checker.log_warning("No test runner found (uv/poetry); skipping test suite")
        return True

    command = [
        runner,
        "run",
        "pytest",
        "tests/unit/gpt_trader/app",
        "tests/unit/gpt_trader/features/brokerages/coinbase/test_coinbase_auth.py",
        "tests/unit/gpt_trader/features/brokerages/coinbase/test_coinbase_models.py",
        "-q",
        "--tb=no",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = result.stdout + result.stderr
        if "passed" in output:
            match = re.search(r"(\d+) passed", output)
            if match:
                passed = int(match.group(1))
                checker.log_success(f"{passed} core tests passed")

            if "failed" in output:
                match = re.search(r"(\d+) failed", output)
                if match:
                    failed = int(match.group(1))
                    checker.log_warning(f"{failed} tests failed")
                    return False
            return True

        checker.log_error("Test suite failed")
        if checker.verbose:
            print(output)
        return False

    except subprocess.TimeoutExpired:
        checker.log_error("Test suite timed out")
        return False
    except Exception as exc:
        checker.log_error(f"Failed to run tests: {exc}")
        return False
