# __main__.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import backtest as backtest_cmd
from . import deploy as deploy_cmd
from . import interactive as interactive_cmd
from . import live as live_cmd
from . import monitor as monitor_cmd
from . import optimize as optimize_cmd
from . import paper as paper_cmd
from . import rapid_evolution as rapid_cmd
from . import walk_forward as walk_cmd
from .cli_utils import (
    CLITheme,
    create_banner,
    handle_error,
    load_config_profile,
    setup_logging,
    validate_environment,
)
from .dashboard import TradingDashboard
from .help_system import add_help_command
from .main_menu import MainMenu
from .setup_wizard import SetupWizard
from .shortcuts import add_shortcuts_command, check_and_run_shortcut

console = Console()
theme = CLITheme()


class CustomFormatter(argparse.HelpFormatter):
    """Custom formatter for better help display."""

    def _format_action_invocation(self, action):
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            (metavar,) = self._metavar_formatter(action, default)(1)
            return metavar
        else:
            parts = []
            for option_string in action.option_strings:
                parts.append(option_string)
            return ", ".join(parts)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with enhanced help."""

    parser = argparse.ArgumentParser(
        prog="gpt-trader",
        description=theme.format_description(
            "🤖 GPT-Trader - AI-Powered Trading Strategy Platform"
        ),
        formatter_class=CustomFormatter,
        epilog=theme.format_epilog(
            "For detailed help on any command, use: gpt-trader <command> --help"
        ),
    )

    # Global options
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )

    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress non-essential output")

    parser.add_argument(
        "--profile",
        type=str,
        help="Load configuration profile from ~/.gpt-trader/profiles/<name>.yaml",
    )

    parser.add_argument(
        "--data-strict",
        choices=["strict", "repair"],
        default="strict",
        help="Data validation mode (strict: raise on bad OHLC, repair: attempt fixes)",
    )

    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    parser.add_argument("--version", action="version", version="%(prog)s 2.0.0")

    # Create subparsers with better organization
    subparsers = parser.add_subparsers(
        title="Available Commands",
        description="Choose from the following trading operations:",
        dest="command",
        metavar="<command>",
        help="Command description",
    )

    # Group commands logically
    # Interactive & Setup
    interactive_parser = interactive_cmd.add_subparser(subparsers)

    # Strategy Development
    backtest_parser = backtest_cmd.add_subparser(subparsers)
    optimize_parser = optimize_cmd.add_subparser(subparsers)
    walk_parser = walk_cmd.add_subparser(subparsers)
    rapid_cmd.add_subparser(subparsers)

    # Live Trading
    paper_cmd.add_subparser(subparsers)
    deploy_cmd.add_subparser(subparsers)
    monitor_cmd.add_subparser(subparsers)
    live_cmd.add_subparser(subparsers)

    # Add new commands
    shortcuts_parser = add_shortcuts_command(subparsers)
    help_parser = add_help_command(subparsers)

    # Add menu command
    menu_parser = subparsers.add_parser(
        "menu",
        help="Launch interactive main menu",
        description="Launch the interactive main menu system",
    )
    menu_parser.set_defaults(func=lambda args: MainMenu().run())

    # Add dashboard command
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        aliases=["dash"],
        help="Launch trading dashboard",
        description="Launch the live trading dashboard",
    )
    dashboard_parser.add_argument(
        "--mode",
        choices=["live", "paper", "backtest", "overview"],
        default="overview",
        help="Dashboard mode",
    )
    dashboard_parser.set_defaults(func=lambda args: TradingDashboard().run(args.mode))

    # Add wizard command
    wizard_parser = subparsers.add_parser(
        "wizard",
        help="Run setup wizard",
        description="Run the interactive setup wizard for new users",
    )
    wizard_parser.set_defaults(func=lambda args: SetupWizard().run())

    # Add aliases for common commands
    subparsers._name_parser_map["bt"] = backtest_parser
    subparsers._name_parser_map["opt"] = optimize_parser
    subparsers._name_parser_map["wf"] = walk_parser
    subparsers._name_parser_map["i"] = interactive_parser
    subparsers._name_parser_map["sh"] = shortcuts_parser
    subparsers._name_parser_map["h"] = help_parser
    subparsers._name_parser_map["m"] = menu_parser
    subparsers._name_parser_map["d"] = dashboard_parser
    subparsers._name_parser_map["w"] = wizard_parser

    return parser


def display_welcome_banner() -> None:
    """Display a welcome banner with system status."""
    banner = create_banner()
    console.print(banner)

    # Show system status
    status_table = Table(show_header=False, box=None, padding=(0, 2))
    status_table.add_column("Item", style="cyan")
    status_table.add_column("Status", style="green")

    status_table.add_row("📊 Data Directory", str(Path("data").resolve()))
    status_table.add_row("🔧 Config Loaded", "✓ Default")
    status_table.add_row("📈 Market Status", "Open" if is_market_open() else "Closed")

    console.print(Panel(status_table, title="[bold blue]System Status", border_style="blue"))
    console.print()


def is_market_open() -> bool:
    """Check if the market is currently open."""
    from datetime import datetime

    import pytz

    ny_tz = pytz.timezone("America/New_York")
    now = datetime.now(ny_tz)

    # Simple check - enhance with holiday calendar if needed
    if now.weekday() >= 5:  # Weekend
        return False

    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now <= market_close


def main() -> None:
    """Enhanced main entry point."""

    # Load environment variables from .env.local first
    env_local = Path(".env.local")
    if env_local.exists():
        load_dotenv(env_local, override=True)

    # Check for shortcuts first
    shortcut_result = check_and_run_shortcut(sys.argv)
    if shortcut_result is not None:
        sys.exit(shortcut_result)

    # Setup logging first
    logger = setup_logging()

    # Validate startup requirements (secrets, configuration, etc.)
    try:
        from bot.startup_validation import validate_startup

        validate_startup(raise_on_failure=True)
    except Exception as e:
        console.print(f"[red]❌ Startup validation failed: {e}[/red]")
        console.print("[yellow]Please check your configuration and try again.[/yellow]")
        sys.exit(1)

    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # If no command provided, launch menu
    if not args.command:
        MainMenu().run()
        sys.exit(0)

    # Handle color preference
    if args.no_color:
        console.no_color = True

    # Set verbosity
    if args.quiet:
        console.quiet = True
    elif args.verbose >= 2:
        logger.setLevel("DEBUG")
    elif args.verbose == 1:
        logger.setLevel("INFO")

    # Display banner for interactive sessions
    if not args.quiet and sys.stdout.isatty():
        display_welcome_banner()

    # Load profile if specified
    if args.profile:
        try:
            profile_config = load_config_profile(args.profile)
            args = merge_args_with_profile(args, profile_config)
            console.print(f"[green]✓[/green] Loaded profile: {args.profile}")
        except Exception as e:
            handle_error(f"Failed to load profile '{args.profile}': {e}")
            sys.exit(1)

    # Validate environment
    if not validate_environment():
        handle_error("Environment validation failed. Please check your setup.")
        sys.exit(1)

    # Map data validation mode
    args.strict_mode = args.data_strict == "strict"

    # Execute command
    if not hasattr(args, "func"):
        parser.print_help()
        console.print(
            "\n[yellow]💡 Tip:[/yellow] Use 'gpt-trader <command> --help' for detailed command usage"
        )
        sys.exit(1)

    try:
        with console.status("[bold green]Executing command..."):
            args.func(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Operation cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        handle_error(f"Command failed: {e}")
        if args.verbose >= 2:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def merge_args_with_profile(args: argparse.Namespace, profile: dict) -> argparse.Namespace:
    """Merge command-line args with profile settings."""
    # Flat merge for simple keys
    for key, value in profile.items():
        if key in ("features", "evolution", "optimizer", "risk", "monitoring"):
            continue
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    # Features: list of feature-set names to compose in research workflows
    if "features" in profile and not hasattr(args, "features"):
        args.features = profile.get("features", [])

    # Evolution/optimizer settings (research mode)
    evo = profile.get("evolution", {})
    if evo:
        for k, v in evo.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)
    opt = profile.get("optimizer", {})
    if opt:
        for k, v in opt.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)

    # Risk/policy controls used by paper/live
    risk = profile.get("risk", {})
    if risk:
        for k, v in risk.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)

    # Monitoring thresholds config
    mon = profile.get("monitoring", {})
    if mon:
        for k, v in mon.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)
    return args


if __name__ == "__main__":
    main()
