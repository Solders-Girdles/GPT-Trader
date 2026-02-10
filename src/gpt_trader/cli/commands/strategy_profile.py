"""CLI helpers for strategy profile diffs."""

from __future__ import annotations

import json
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import Any

from gpt_trader.app.config.profile_loader import DEFAULT_RUNTIME_PROFILE_NAME
from gpt_trader.cli.options import add_output_options
from gpt_trader.cli.response import CliErrorCode, CliResponse
from gpt_trader.features.strategy_dev.config.diff import (
    DEFAULT_IGNORED_FIELDS,
    ProfileDiffEntry,
    compute_profile_diff,
)

COMMAND_NAME = "strategy profile diff"
_DEFAULT_RUNTIME_PROFILE_FILENAME = "strategy_profile.json"
_STATUS_ORDER = ("changed", "missing", "unchanged")


def register(subparsers: Any) -> None:
    """Register the strategy profile subcommands."""
    parser = subparsers.add_parser(
        "strategy",
        help="Strategy development tools",
        description="Tools for strategy profile inspection and comparison.",
    )

    strategy_subparsers = parser.add_subparsers(
        dest="strategy_command",
        title="strategy subcommands",
        description="Available strategy commands",
    )

    diff_parser = strategy_subparsers.add_parser(
        "profile-diff",
        help="Diff a baseline StrategyProfile against runtime values",
    )
    diff_parser.add_argument(
        "--baseline",
        "-b",
        type=Path,
        required=True,
        help="Path to the baseline strategy profile (JSON or YAML)",
    )
    diff_parser.add_argument(
        "--runtime-profile",
        "-r",
        type=Path,
        help="Explicit runtime strategy profile file (overrides --runtime-root)",
    )
    diff_parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path("."),
        help="Root path containing runtime_data/<profile>",
    )
    diff_parser.add_argument(
        "--profile",
        type=str,
        default=DEFAULT_RUNTIME_PROFILE_NAME,
        help="Runtime profile directory name (defaults to dev)",
    )
    diff_parser.add_argument(
        "--ignore",
        dest="ignore_fields",
        action="append",
        default=[],
        help="Additional field names to ignore (can be repeated)",
    )
    add_output_options(diff_parser, include_quiet=False)
    diff_parser.set_defaults(handler=execute_profile_diff, subcommand="profile-diff")


def execute_profile_diff(args: Namespace) -> CliResponse | int:
    """Execute the strategy profile diff command."""
    output_format = getattr(args, "output_format", "text")
    baseline_path = Path(args.baseline)

    if args.runtime_profile:
        runtime_path = Path(args.runtime_profile)
    else:
        runtime_root = Path(args.runtime_root)
        runtime_path = runtime_root / "runtime_data" / args.profile / _DEFAULT_RUNTIME_PROFILE_FILENAME

    try:
        baseline_data = _load_profile_file(baseline_path)
    except FileNotFoundError:
        return CliResponse.error_response(
            command=COMMAND_NAME,
            code=CliErrorCode.FILE_NOT_FOUND,
            message=f"Baseline profile not found: {baseline_path}",
            details={"path": str(baseline_path)},
        )
    except ValueError as exc:
        return CliResponse.error_response(
            command=COMMAND_NAME,
            code=CliErrorCode.CONFIG_INVALID,
            message=f"Failed to load baseline profile: {exc}",
            details={"path": str(baseline_path)},
        )

    try:
        runtime_data = _load_profile_file(runtime_path)
    except FileNotFoundError:
        return CliResponse.error_response(
            command=COMMAND_NAME,
            code=CliErrorCode.FILE_NOT_FOUND,
            message=f"Runtime profile not found: {runtime_path}",
            details={"path": str(runtime_path)},
        )
    except ValueError as exc:
        return CliResponse.error_response(
            command=COMMAND_NAME,
            code=CliErrorCode.CONFIG_INVALID,
            message=f"Failed to load runtime profile: {exc}",
            details={"path": str(runtime_path)},
        )

    ignore_fields = set(DEFAULT_IGNORED_FIELDS)
    ignore_fields.update(args.ignore_fields or [])
    diff = compute_profile_diff(baseline_data, runtime_data, ignore_fields=ignore_fields)

    if output_format == "json":
        return CliResponse.success_response(
            command=COMMAND_NAME,
            data={
                "baseline_path": str(baseline_path),
                "runtime_profile_path": str(runtime_path),
                "diff": diff,
            },
        )

    print(f"Baseline: {baseline_path}")
    print(f"Runtime : {runtime_path}")
    print()
    print(_format_profile_diff_text(diff))
    return 0


def _load_profile_file(path: Path) -> dict[str, Any]:
    """Load a profile file from JSON or YAML."""
    if not path.exists():
        raise FileNotFoundError(path)

    content = path.read_text()
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml

            data = yaml.safe_load(content) or {}
        except ModuleNotFoundError as exc:
            raise ValueError("PyYAML is required to read YAML profiles") from exc
        except Exception as exc:
            raise ValueError(str(exc)) from exc
    else:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(str(exc)) from exc

    if not isinstance(data, dict):
        raise ValueError("Profile payload must be an object")

    return data


def _format_profile_diff_text(entries: list[ProfileDiffEntry]) -> str:
    """Render diff entries for text output."""
    counts = Counter(entry["status"] for entry in entries)
    lines: list[str] = []
    lines.append("Profile diff:")
    statuses = " ".join(f"{status}={counts.get(status, 0)}" for status in _STATUS_ORDER)
    lines.append(f"Statuses: {statuses}")
    lines.append("-" * 80)

    for entry in entries:
        lines.append(f"{entry['status']:<8} {entry['path']}")
        lines.append(f"  baseline: {_format_value(entry['baseline_value'])}")
        lines.append(f"  runtime : {_format_value(entry['runtime_value'])}")
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    """Format a diff value for text output."""
    if value is None:
        return "None"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)
