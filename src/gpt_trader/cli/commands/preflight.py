"""Preflight command for running production readiness checks."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from gpt_trader.preflight import run_preflight_cli
from gpt_trader.preflight.cli_args import add_preflight_arguments


def register(subparsers: Any) -> None:
    """Register the preflight command."""
    parser = subparsers.add_parser(
        "preflight",
        help="Run production preflight checks",
        description="Run production preflight checks for GPT-Trader",
    )
    add_preflight_arguments(parser)
    parser.set_defaults(handler=execute)


def execute(args: Namespace) -> int:
    """Execute the preflight command by delegating to the preflight CLI."""
    argv: list[str] = ["--profile", args.profile]
    if getattr(args, "verbose", False):
        argv.append("--verbose")
    if getattr(args, "warn_only", False):
        argv.append("--warn-only")
    if getattr(args, "diagnostics_bundle", False):
        argv.append("--diagnostics-bundle")
    if getattr(args, "report_dir", None):
        argv.extend(["--report-dir", str(args.report_dir)])
    if getattr(args, "report_path", None):
        argv.extend(["--report-path", str(args.report_path)])
    report_target = getattr(args, "report_target", None)
    report_target_value = getattr(report_target, "value", report_target)
    if report_target_value == "stdout":
        argv.extend(["--report-target", "stdout"])
    return run_preflight_cli(argv)
