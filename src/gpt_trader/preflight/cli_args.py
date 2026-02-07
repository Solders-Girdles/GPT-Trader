from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

PROFILE_CHOICES = ("dev", "canary", "prod")


@dataclass(frozen=True)
class PreflightCliArgs:
    verbose: bool
    profile: str
    warn_only: bool
    report_dir: Path | None
    report_path: Path | None


def add_preflight_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--profile",
        "-p",
        default="canary",
        choices=PROFILE_CHOICES,
        help="Trading profile to validate (default: canary)",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Downgrade diagnostic failures to warnings (also: GPT_TRADER_PREFLIGHT_WARN_ONLY=1)",
    )

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--report-dir",
        type=Path,
        help="Directory for preflight_report_*.json output (default: current working dir)",
    )
    output_group.add_argument(
        "--report-path",
        type=Path,
        help="Explicit file path for preflight report JSON",
    )


def build_preflight_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Production preflight check for GPT-Trader")
    add_preflight_arguments(parser)
    return parser


def parse_preflight_args(argv: Sequence[str] | None = None) -> PreflightCliArgs:
    parser = build_preflight_parser()
    args = parser.parse_args(argv)
    return _normalize_preflight_args(parser, args)


def _normalize_preflight_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> PreflightCliArgs:
    report_dir = _normalize_report_dir(parser, args.report_dir)
    report_path = _normalize_report_path(parser, args.report_path)

    return PreflightCliArgs(
        verbose=bool(args.verbose),
        profile=str(args.profile),
        warn_only=bool(args.warn_only),
        report_dir=report_dir,
        report_path=report_path,
    )


def _normalize_report_dir(parser: argparse.ArgumentParser, value: Path | None) -> Path | None:
    if value is None:
        return None
    resolved = value.expanduser().resolve(strict=False)
    if resolved.exists() and not resolved.is_dir():
        parser.error(f"Report directory points to a file: {resolved}")
    return resolved


def _normalize_report_path(parser: argparse.ArgumentParser, value: Path | None) -> Path | None:
    if value is None:
        return None
    resolved = value.expanduser().resolve(strict=False)
    if resolved.exists() and resolved.is_dir():
        parser.error(f"Report path must be a file, got directory: {resolved}")
    return resolved
