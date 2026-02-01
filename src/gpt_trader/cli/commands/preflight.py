"""Preflight command for running production readiness checks."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from gpt_trader.preflight import run_preflight_cli

_PROFILE_CHOICES = ["dev", "canary", "prod"]


def register(subparsers: Any) -> None:
    """Register the preflight command."""
    parser = subparsers.add_parser(
        "preflight",
        help="Run production preflight checks",
        description="Run production preflight checks for GPT-Trader",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--profile",
        "-p",
        default="canary",
        choices=_PROFILE_CHOICES,
        help="Trading profile to validate (default: canary)",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Downgrade diagnostic failures to warnings (also: GPT_TRADER_PREFLIGHT_WARN_ONLY=1)",
    )
    parser.set_defaults(handler=execute)


def execute(args: Namespace) -> int:
    """Execute the preflight command by delegating to the preflight CLI."""
    argv: list[str] = ["--profile", args.profile]
    if getattr(args, "verbose", False):
        argv.append("--verbose")
    if getattr(args, "warn_only", False):
        argv.append("--warn-only")
    return run_preflight_cli(argv)
