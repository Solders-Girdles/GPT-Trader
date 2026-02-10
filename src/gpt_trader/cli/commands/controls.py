"""Controls utilities for operational safety checks."""

from __future__ import annotations

import runpy
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gpt_trader.cli import options
from gpt_trader.cli.response import CliResponse

COMMAND_NAME = "controls summary"


def _load_controls_smoke_module() -> Any:
    try:
        from scripts.ops import controls_smoke as module

        return module
    except ModuleNotFoundError:
        module_path = Path(__file__).resolve().parents[4] / "scripts" / "ops" / "controls_smoke.py"
        namespace = runpy.run_path(str(module_path))
        return SimpleNamespace(**namespace)


controls_smoke = _load_controls_smoke_module()


def register(subparsers: Any) -> None:
    """Register the controls command and its subcommands."""
    parser = subparsers.add_parser(
        "controls",
        help="Inspect trading control smoke checks",
        description="Run dry-run summaries of the guard-level control smoke checks",
    )
    controls_subparsers = parser.add_subparsers(dest="controls_command", required=True)

    summary = controls_subparsers.add_parser(
        "summary",
        help="Print a deterministic summary of the control smoke checks (pass/warn/fail counts)",
    )
    options.add_output_options(summary, include_quiet=False)
    summary.add_argument(
        "--max-top-failures",
        type=int,
        default=controls_smoke.DEFAULT_SUMMARY_MAX_FAILURES,
        help="Maximum number of failing checks to include in the summary (default: %(default)s)",
    )
    summary.set_defaults(handler=_handle_summary, subcommand="summary")


def _handle_summary(args: Namespace) -> CliResponse:
    """Handle the controls summary subcommand."""
    limit = max(0, args.max_top_failures or 0)
    results = controls_smoke.run_smoke_checks()
    outcome = controls_smoke.determine_outcome(results)
    payload = controls_smoke.build_summary_payload(
        results,
        outcome,
        max_top_failures=limit,
    )

    severity_counts = payload["summary"]["severity_counts"]
    warnings: list[str] = []
    if severity_counts["fail"] > 0:
        warnings.append(
            f"{severity_counts['fail']} unexpected failures detected (exit_code={outcome.exit_code})"
        )
    elif severity_counts["warn"] > 0:
        warnings.append(
            f"{severity_counts['warn']} controls were guard-blocked (guard checks still functional)"
        )

    return CliResponse(
        success=True,
        command=COMMAND_NAME,
        data=payload,
        warnings=warnings,
        exit_code=outcome.exit_code,
        was_noop=True,
    )
