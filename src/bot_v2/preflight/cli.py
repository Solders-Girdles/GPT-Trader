from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Callable, Sequence

from .context import Colors
from .core import PreflightCheck


def _header(profile: str) -> None:
    print(f"{Colors.BOLD}{Colors.MAGENTA}")
    print("=" * 70)
    print("GPT-TRADER PRODUCTION PREFLIGHT CHECK")
    print(f"Profile: {profile}")
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)
    print(f"{Colors.RESET}")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry-point for production preflight command."""
    parser = argparse.ArgumentParser(description="Production preflight check for GPT-Trader")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--profile",
        "-p",
        default="canary",
        choices=["dev", "canary", "prod"],
        help="Trading profile to validate (default: canary)",
    )

    args = parser.parse_args(argv)
    _header(args.profile)

    checker = PreflightCheck(verbose=args.verbose, profile=args.profile)
    check_functions: Sequence[Callable[[], bool]] = [
        checker.check_python_version,
        checker.check_dependencies,
        checker.check_environment_variables,
        checker.check_api_connectivity,
        checker.check_key_permissions,
        checker.check_risk_configuration,
        checker.check_test_suite,
        checker.check_profile_configuration,
        checker.check_system_time,
        checker.check_disk_space,
        checker.simulate_dry_run,
    ]

    for check in check_functions:
        try:
            check()
        except Exception as exc:  # pragma: no cover - defensive runtime safeguard
            checker.log_error(f"Check failed with exception: {exc}")

    success, _status = checker.generate_report()
    return 0 if success else 1
