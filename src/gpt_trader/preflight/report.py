from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, TypedDict

from gpt_trader.preflight.hints import DEFAULT_REMEDIATION_HINT

from .context import Colors

if TYPE_CHECKING:
    from gpt_trader.preflight.context import PreflightContext
    from gpt_trader.preflight.core import PreflightCheck


class FailureHint(TypedDict):
    message: str
    hint: str
    check: str | None


def _collect_failure_hints(context: "PreflightContext") -> list[FailureHint]:
    hints: list[FailureHint] = []
    for result in context.results:
        if result["status"] != "fail":
            continue
        details = result["details"]
        hints.append(
            FailureHint(
                message=result["message"],
                hint=details.get("hint") or DEFAULT_REMEDIATION_HINT,
                check=details.get("check"),
            )
        )
    return hints


def evaluate_preflight_status(
    *,
    success_count: int,
    warning_count: int,
    error_count: int,
) -> tuple[str, str]:
    """Return status label and message for the given totals."""
    if error_count == 0:
        if warning_count <= 3:
            return "READY", "System is READY for production trading (with caution)"
        return "REVIEW", "System has warnings - review before proceeding"
    return "NOT READY", "System is NOT READY - critical issues must be resolved"


def format_preflight_report(
    checker: PreflightCheck,
    *,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Return report payload without IO side effects."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    ctx = checker.context
    status, _message = evaluate_preflight_status(
        success_count=len(ctx.successes),
        warning_count=len(ctx.warnings),
        error_count=len(ctx.errors),
    )
    total_checks = len(ctx.successes) + len(ctx.warnings) + len(ctx.errors)
    failure_hints = _collect_failure_hints(ctx)

    return {
        "timestamp": timestamp.isoformat(),
        "profile": checker.profile,
        "status": status,
        "successes": len(ctx.successes),
        "warnings": len(ctx.warnings),
        "errors": len(ctx.errors),
        "details": {
            "successes": list(ctx.successes),
            "warnings": list(ctx.warnings),
            "errors": list(ctx.errors),
            "error_hints": failure_hints,
        },
        "total_checks": total_checks,
    }


def serialize_preflight_report(report_payload: Mapping[str, Any]) -> str:
    """Serialize report payload to JSON text."""
    return json.dumps(report_payload, indent=2)


def report_path_for_timestamp(timestamp: datetime, *, output_dir: Path | None = None) -> Path:
    """Resolve report path for the provided timestamp."""
    filename = f"preflight_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    if output_dir is None:
        return Path(filename)
    return output_dir / filename


def write_preflight_report(report_path: Path, report_content: str) -> None:
    """Persist report content to disk."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding="utf-8")


def generate_report(
    checker: PreflightCheck,
    *,
    report_dir: Path | None = None,
    report_path: Path | None = None,
) -> tuple[bool, str]:
    """Render terminal summary and persist JSON report."""
    ctx = checker.context
    failure_hints = _collect_failure_hints(ctx)
    checker.section_header("PREFLIGHT REPORT")

    print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
    print(f"  {Colors.GREEN}✅ Passed: {len(ctx.successes)}{Colors.RESET}")
    print(f"  {Colors.YELLOW}⚠️  Warnings: {len(ctx.warnings)}{Colors.RESET}")
    print(f"  {Colors.RED}❌ Failed: {len(ctx.errors)}{Colors.RESET}")

    status, message = evaluate_preflight_status(
        success_count=len(ctx.successes),
        warning_count=len(ctx.warnings),
        error_count=len(ctx.errors),
    )
    color = {
        "READY": Colors.GREEN,
        "REVIEW": Colors.YELLOW,
        "NOT READY": Colors.RED,
    }[status]

    print(f"\n{Colors.BOLD}{color}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{color}STATUS: {status}{Colors.RESET}")
    print(f"{color}{message}{Colors.RESET}")
    print(f"{Colors.BOLD}{color}{'=' * 70}{Colors.RESET}")

    if failure_hints:
        print(f"\n{Colors.BOLD}Remediation hints:{Colors.RESET}")
        for idx, hint in enumerate(failure_hints, start=1):
            print(f"{idx}. {hint['hint']} — {hint['message']}")

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
    report_payload = format_preflight_report(checker, timestamp=timestamp)
    report_content = serialize_preflight_report(report_payload)
    if report_dir is not None and report_path is not None:
        raise ValueError("report_dir and report_path are mutually exclusive")
    if report_path is None:
        report_path = report_path_for_timestamp(timestamp, output_dir=report_dir)

    try:
        write_preflight_report(report_path, report_content)
        print(f"\n{Colors.CYAN}Report saved to: {report_path}{Colors.RESET}")
    except Exception as exc:
        print(f"\n{Colors.YELLOW}Could not save report: {exc}{Colors.RESET}")

    return len(ctx.errors) == 0, status
