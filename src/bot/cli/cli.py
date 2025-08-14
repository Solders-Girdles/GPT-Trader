#!/usr/bin/env python3
"""
GPT-Trader Unified CLI
Main entry point for all command-line operations
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .commands import (
    BacktestCommand,
    OptimizeCommand,
    LiveCommand,
    PaperCommand,
    MonitorCommand,
    DashboardCommand,
    WizardCommand,
)
from .ml_commands import MLTrainCommand, AutoTradeCommand
from .cli_utils import setup_logging, print_banner, get_version


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser"""
    parser = argparse.ArgumentParser(
        prog="gpt-trader",
        description="GPT-Trader: Autonomous Portfolio Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gpt-trader backtest --symbol AAPL --start 2024-01-01 --end 2024-06-30
  gpt-trader optimize --symbol SPY --strategy demo_ma
  gpt-trader paper --portfolio tech_stocks
  gpt-trader dashboard
  gpt-trader monitor --refresh 5

For more help on a specific command:
  gpt-trader <command> --help
        """,
    )

    # Global options
    parser.add_argument("--version", action="version", version=f"GPT-Trader {get_version()}")

    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase verbosity (-vv for debug)"
    )

    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress non-essential output")

    parser.add_argument("--config", type=Path, help="Path to configuration file")

    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory path")

    # Subcommands
    subparsers = parser.add_subparsers(title="Commands", dest="command", help="Available commands")

    # Register commands
    BacktestCommand.add_parser(subparsers)
    OptimizeCommand.add_parser(subparsers)
    LiveCommand.add_parser(subparsers)
    PaperCommand.add_parser(subparsers)
    MonitorCommand.add_parser(subparsers)
    DashboardCommand.add_parser(subparsers)
    WizardCommand.add_parser(subparsers)
    MLTrainCommand.add_parser(subparsers)
    AutoTradeCommand.add_parser(subparsers)

    return parser


def main(args: Optional[list] = None) -> int:
    """Main CLI entry point"""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    # Setup logging based on verbosity
    if parsed_args.quiet:
        log_level = "ERROR"
    elif parsed_args.verbose >= 2:
        log_level = "DEBUG"
    elif parsed_args.verbose == 1:
        log_level = "INFO"
    else:
        log_level = "WARNING"

    setup_logging(log_level)

    # Show banner unless quiet
    if not parsed_args.quiet and not parsed_args.command:
        print_banner()
        parser.print_help()
        return 0

    # Execute command
    if not parsed_args.command:
        parser.print_help()
        return 0

    try:
        # Import command classes and execute
        command_map = {
            "backtest": BacktestCommand,
            "optimize": OptimizeCommand,
            "live": LiveCommand,
            "paper": PaperCommand,
            "monitor": MonitorCommand,
            "dashboard": DashboardCommand,
            "wizard": WizardCommand,
        }

        command_class = command_map.get(parsed_args.command)
        if command_class:
            command = command_class(parsed_args)
            return command.execute()
        else:
            print(f"Unknown command: {parsed_args.command}")
            return 1

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        return 130
    except Exception as e:
        if parsed_args.verbose >= 2:
            import traceback

            traceback.print_exc()
        else:
            print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
