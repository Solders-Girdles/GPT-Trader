from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .context import Colors

if TYPE_CHECKING:
    from gpt_trader.preflight.core import PreflightCheck


def generate_report(checker: PreflightCheck) -> tuple[bool, str]:
    """Render terminal summary and persist JSON report."""
    ctx = checker.context
    checker.section_header("PREFLIGHT REPORT")

    total_checks = len(ctx.successes) + len(ctx.warnings) + len(ctx.errors)
    print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
    print(f"  {Colors.GREEN}✅ Passed: {len(ctx.successes)}{Colors.RESET}")
    print(f"  {Colors.YELLOW}⚠️  Warnings: {len(ctx.warnings)}{Colors.RESET}")
    print(f"  {Colors.RED}❌ Failed: {len(ctx.errors)}{Colors.RESET}")

    if len(ctx.errors) == 0:
        if len(ctx.warnings) <= 3:
            status = "READY"
            color = Colors.GREEN
            message = "System is READY for production trading (with caution)"
        else:
            status = "REVIEW"
            color = Colors.YELLOW
            message = "System has warnings - review before proceeding"
    else:
        status = "NOT READY"
        color = Colors.RED
        message = "System is NOT READY - critical issues must be resolved"

    print(f"\n{Colors.BOLD}{color}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{color}STATUS: {status}{Colors.RESET}")
    print(f"{color}{message}{Colors.RESET}")
    print(f"{Colors.BOLD}{color}{'=' * 70}{Colors.RESET}")

    print(f"\n{Colors.BOLD}Recommendations:{Colors.RESET}")
    if status == "READY":
        print("1. Start with: uv run gpt-trader run --profile " f"{checker.profile} --dry-run")
        print("2. Monitor for 1 hour in dry-run mode")
        print(f"3. Begin live with: uv run gpt-trader run --profile {checker.profile}")
        print("4. Use tiny positions (0.001 BTC) initially")
        print("5. Monitor closely for first 24 hours")
    elif status == "REVIEW":
        print("1. Review all warnings above")
        print("2. Consider starting with paper trading: PERPS_PAPER=1")
        print("3. Ensure emergency procedures are documented")
        print("4. Test kill switch: RISK_KILL_SWITCH_ENABLED=1")
    else:
        print("1. Fix all critical errors listed above")
        print("2. Review config/environments/.env.production for configuration guidance")
        print("3. Run tests: uv run pytest tests/unit/gpt_trader")
        print("4. Verify credentials and API connectivity")

    timestamp = datetime.now(timezone.utc)
    report_path = Path(f"preflight_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json")
    report_data = {
        "timestamp": timestamp.isoformat(),
        "profile": checker.profile,
        "status": status,
        "successes": len(ctx.successes),
        "warnings": len(ctx.warnings),
        "errors": len(ctx.errors),
        "details": {
            "successes": ctx.successes,
            "warnings": ctx.warnings,
            "errors": ctx.errors,
        },
        "total_checks": total_checks,
    }

    try:
        with open(report_path, "w") as handle:
            json.dump(report_data, handle, indent=2)
        print(f"\n{Colors.CYAN}Report saved to: {report_path}{Colors.RESET}")
    except Exception as exc:
        print(f"\n{Colors.YELLOW}Could not save report: {exc}{Colors.RESET}")

    return len(ctx.errors) == 0, status
