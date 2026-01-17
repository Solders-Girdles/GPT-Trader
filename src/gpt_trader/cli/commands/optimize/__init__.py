"""Optimization CLI subcommands."""

from __future__ import annotations

from typing import Any


def register(subparsers: Any) -> None:
    """
    Register the optimize command and its subcommands.

    Args:
        subparsers: argparse subparsers from the parent parser
    """
    parser = subparsers.add_parser(
        "optimize",
        help="Run and manage strategy optimization studies",
        description="Optimize trading strategy parameters using Optuna.",
    )

    optimize_subparsers = parser.add_subparsers(
        dest="optimize_command",
        title="subcommands",
        description="Available optimization commands",
    )

    # Import and register each subcommand
    from . import apply, compare, export, resume, run, view
    from . import list as list_cmd

    run.register(optimize_subparsers)
    view.register(optimize_subparsers)
    list_cmd.register(optimize_subparsers)
    compare.register(optimize_subparsers)
    export.register(optimize_subparsers)
    resume.register(optimize_subparsers)
    apply.register(optimize_subparsers)

    parser.set_defaults(handler=_default_handler)


def _default_handler(args: Any) -> int:
    """Default handler when no subcommand is specified."""
    print("Usage: gpt-trader optimize <subcommand>")
    print()
    print("Available subcommands:")
    print("  run      - Run an optimization study")
    print("  view     - View results of a specific run")
    print("  list     - List all optimization runs")
    print("  compare  - Compare multiple runs")
    print("  export   - Export results (JSON/CSV/YAML)")
    print("  resume   - Resume an interrupted study")
    print("  apply    - Apply optimized params to a config file")
    print()
    print("Use 'gpt-trader optimize <subcommand> --help' for more information.")
    return 0
