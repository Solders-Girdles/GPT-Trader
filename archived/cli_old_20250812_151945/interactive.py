"""Interactive CLI command with QoL improvements."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.panel import Panel

from .cli_utils import (
    CLITheme,
    DataValidator,
    InteractivePrompts,
    PerformanceMonitor,
    confirm_action,
    console,
    display_dependency_status,
    display_system_info,
    list_available_profiles,
    print_separator,
    save_config_profile,
)
from .shared_enhanced import EnhancedUniverseReader


def add_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "interactive",
        help="Interactive setup and guided trading",
        description="""
        Interactive setup and guided trading with step-by-step assistance.

        This command provides an interactive experience for:
        - System setup and validation
        - Configuration profile creation
        - Guided backtest setup
        - Interactive parameter selection
        """,
    )

    # Interactive options
    setup_group = p.add_argument_group("Setup Options")
    setup_group.add_argument("--setup", action="store_true", help="Run interactive setup wizard")
    setup_group.add_argument(
        "--create-profile", action="store_true", help="Create a new configuration profile"
    )
    setup_group.add_argument(
        "--guided-backtest", action="store_true", help="Run guided backtest setup"
    )
    setup_group.add_argument(
        "--system-check", action="store_true", help="Run comprehensive system check"
    )

    p.set_defaults(func=_handle_interactive)
    return p


def _handle_interactive(args: argparse.Namespace) -> None:
    """Handle interactive CLI commands."""

    if args.setup:
        run_setup_wizard()
    elif args.create_profile:
        create_configuration_profile()
    elif args.guided_backtest:
        run_guided_backtest()
    elif args.system_check:
        run_system_check()
    else:
        # Default: show interactive menu
        show_interactive_menu()


def show_interactive_menu() -> None:
    """Show interactive menu."""
    console.print(
        Panel(
            "[bold]Interactive GPT-Trader Setup[/bold]\n" "Choose an option to get started:",
            title="[bold cyan]Welcome to GPT-Trader",
            border_style="cyan",
        )
    )

    options = [
        ("1", "System Setup & Validation", run_setup_wizard),
        ("2", "Create Configuration Profile", create_configuration_profile),
        ("3", "Guided Backtest Setup", run_guided_backtest),
        ("4", "System Check", run_system_check),
        ("5", "Exit", lambda: None),
    ]

    for key, description, _ in options:
        console.print(f"[cyan]{key}[/cyan]. {description}")

    console.print()

    while True:
        choice = console.input("[yellow]Select option (1-5):[/yellow] ").strip()

        for key, description, func in options:
            if choice == key:
                if key == "5":
                    console.print(CLITheme.info("Goodbye!"))
                    return
                func()
                return

        console.print(CLITheme.error("Invalid option. Please select 1-5."))


def run_setup_wizard() -> None:
    """Run interactive setup wizard."""
    console.print(
        Panel(
            "[bold]GPT-Trader Setup Wizard[/bold]\n"
            "This wizard will help you set up GPT-Trader for optimal performance.",
            title="[bold cyan]Setup Wizard",
            border_style="cyan",
        )
    )

    print_separator("Step 1: System Check")

    # Check dependencies
    console.print("[bold]Checking dependencies...[/bold]")
    display_dependency_status()

    print_separator("Step 2: Environment Validation")

    # Validate environment
    console.print("[bold]Validating environment...[/bold]")
    from .cli_utils import validate_environment

    if validate_environment():
        console.print(CLITheme.success("Environment validation passed"))
    else:
        console.print(CLITheme.error("Environment validation failed"))
        return

    print_separator("Step 3: System Information")

    # Display system info
    console.print("[bold]System Information:[/bold]")
    display_system_info()

    print_separator("Step 4: Configuration")

    # Ask about configuration
    if confirm_action("Would you like to create a configuration profile?", default=True):
        create_configuration_profile()

    console.print(CLITheme.success("Setup wizard completed successfully!"))


def create_configuration_profile() -> None:
    """Create a new configuration profile interactively."""
    console.print(
        Panel(
            "[bold]Configuration Profile Creation[/bold]\n"
            "Create a new configuration profile for common settings.",
            title="[bold cyan]Profile Creation",
            border_style="cyan",
        )
    )

    # Get profile name
    profile_name = console.input("[yellow]Enter profile name:[/yellow] ").strip()
    if not profile_name:
        console.print(CLITheme.error("Profile name cannot be empty"))
        return

    # Check if profile already exists
    existing_profiles = list_available_profiles()
    if profile_name in existing_profiles:
        if not confirm_action(
            f"Profile '{profile_name}' already exists. Overwrite?", default=False
        ):
            return

    # Collect configuration
    config = {}

    console.print("\n[bold]Strategy Configuration:[/bold]")
    config["strategy"] = InteractivePrompts.prompt_strategy()

    console.print("\n[bold]Risk Management:[/bold]")
    risk_pct, max_positions = InteractivePrompts.prompt_risk_settings()
    config["risk_pct"] = risk_pct
    config["max_positions"] = max_positions

    console.print("\n[bold]Common Symbols:[/bold]")
    symbols = InteractivePrompts.prompt_symbols()
    config["symbols"] = ",".join(symbols)

    console.print("\n[bold]Data Validation:[/bold]")
    data_strict = console.input(
        "[yellow]Data validation mode (strict/repair) [repair]:[/yellow] "
    ).strip()
    config["data_strict"] = data_strict if data_strict in ["strict", "repair"] else "repair"

    console.print("\n[bold]Verbosity:[/bold]")
    verbose = console.input("[yellow]Verbosity level (0-2) [1]:[/yellow] ").strip()
    config["verbose"] = int(verbose) if verbose.isdigit() and 0 <= int(verbose) <= 2 else 1

    # Save profile
    try:
        save_config_profile(profile_name, config)
        console.print(CLITheme.success(f"Profile '{profile_name}' created successfully!"))

        # Show profile summary
        console.print("\n[bold]Profile Summary:[/bold]")
        for key, value in config.items():
            console.print(f"  [cyan]{key}[/cyan]: {value}")

    except Exception as e:
        console.print(CLITheme.error(f"Failed to create profile: {e}"))


def run_guided_backtest() -> None:
    """Run guided backtest setup."""
    console.print(
        Panel(
            "[bold]Guided Backtest Setup[/bold]\n"
            "This will guide you through setting up a backtest step by step.",
            title="[bold cyan]Guided Backtest",
            border_style="cyan",
        )
    )

    # Performance monitor
    monitor = PerformanceMonitor()
    monitor.start("Guided Backtest Setup")

    try:
        print_separator("Step 1: Symbol Selection")

        # Ask for symbol input method
        input_method = (
            console.input(
                "[yellow]Symbol input method (single/list/csv) [single]:[/yellow] "
            ).strip()
            or "single"
        )

        if input_method == "single":
            symbol = InteractivePrompts.prompt_symbol()
            symbols = [symbol]
            symbol_list = None
        elif input_method == "list":
            symbols = InteractivePrompts.prompt_symbols()
            symbol_list = None
        elif input_method == "csv":
            csv_path = console.input("[yellow]Enter CSV file path:[/yellow] ").strip()
            try:
                symbols = EnhancedUniverseReader.read_universe_csv(csv_path)
                symbol_list = csv_path
            except Exception as e:
                console.print(CLITheme.error(f"Failed to read CSV: {e}"))
                return
        else:
            console.print(CLITheme.error("Invalid input method"))
            return

        print_separator("Step 2: Date Range")

        # Get date range
        start_date, end_date = InteractivePrompts.prompt_date_range()

        # Validate date range
        try:
            start_dt, end_dt = DataValidator.validate_date_range(start_date, end_date)
            console.print(CLITheme.success(f"Date range: {start_date} to {end_date}"))
        except ValueError as e:
            console.print(CLITheme.error(f"Invalid date range: {e}"))
            return

        print_separator("Step 3: Strategy Configuration")

        # Strategy selection
        strategy = InteractivePrompts.prompt_strategy()

        # Strategy parameters
        if strategy == "trend_breakout":
            donchian = console.input("[yellow]Donchian lookback (days) [55]:[/yellow] ").strip()
            donchian = int(donchian) if donchian.isdigit() else 55

            atr = console.input("[yellow]ATR period (days) [20]:[/yellow] ").strip()
            atr = int(atr) if atr.isdigit() else 20

            atr_k = console.input("[yellow]ATR multiplier [2.0]:[/yellow] ").strip()
            atr_k = float(atr_k) if atr_k.replace(".", "").isdigit() else 2.0

        print_separator("Step 4: Risk Management")

        # Risk settings
        risk_pct, max_positions = InteractivePrompts.prompt_risk_settings()

        print_separator("Step 5: Execution Settings")

        # Execution settings
        cadence = console.input(
            "[yellow]Rebalancing frequency (daily/weekly) [daily]:[/yellow] "
        ).strip()
        cadence = cadence if cadence in ["daily", "weekly"] else "daily"

        regime_filter = confirm_action("Enable regime filter?", default=False)
        regime_window = None
        if regime_filter:
            regime_window = console.input("[yellow]Regime window (days) [200]:[/yellow] ").strip()
            regime_window = int(regime_window) if regime_window.isdigit() else 200

        print_separator("Step 6: Output Configuration")

        # Output settings
        run_tag = console.input("[yellow]Run tag (optional):[/yellow] ").strip()
        no_plot = confirm_action("Disable chart generation?", default=False)

        # Display configuration summary
        console.print("\n[bold]Backtest Configuration Summary:[/bold]")
        summary_data = {
            "Symbols": ", ".join(symbols) if len(symbols) <= 5 else f"{len(symbols)} symbols",
            "Date Range": f"{start_date} to {end_date}",
            "Strategy": strategy,
            "Risk per Trade": f"{risk_pct}%",
            "Max Positions": max_positions,
            "Cadence": cadence,
            "Regime Filter": "Enabled" if regime_filter else "Disabled",
        }

        if strategy == "trend_breakout":
            summary_data.update(
                {
                    "Donchian Lookback": donchian,
                    "ATR Period": atr,
                    "ATR Multiplier": atr_k,
                }
            )

        for key, value in summary_data.items():
            console.print(f"  [cyan]{key}[/cyan]: {value}")

        # Confirm execution
        if confirm_action("\nProceed with backtest?", default=True):
            # Build command
            cmd_parts = ["gpt-trader", "backtest"]

            if input_method == "single":
                cmd_parts.extend(["--symbol", symbols[0]])
            elif input_method == "list":
                cmd_parts.extend(["--symbols", ",".join(symbols)])
            else:
                cmd_parts.extend(["--symbol-list", symbol_list])

            cmd_parts.extend(
                [
                    "--start",
                    start_date,
                    "--end",
                    end_date,
                    "--strategy",
                    strategy,
                    "--risk-pct",
                    str(risk_pct),
                    "--max-positions",
                    str(max_positions),
                    "--cadence",
                    cadence,
                ]
            )

            if strategy == "trend_breakout":
                cmd_parts.extend(
                    [
                        "--donchian",
                        str(donchian),
                        "--atr",
                        str(atr),
                        "--atr-k",
                        str(atr_k),
                    ]
                )

            if regime_filter:
                cmd_parts.extend(["--regime", "on", "--regime-window", str(regime_window)])

            if run_tag:
                cmd_parts.extend(["--run-tag", run_tag])

            if no_plot:
                cmd_parts.append("--no-plot")

            console.print("\n[bold]Generated Command:[/bold]")
            console.print(f"[dim]{' '.join(cmd_parts)}[/dim]")

            if confirm_action("Execute this command now?", default=True):
                # Import and run backtest
                import argparse

                from .backtest import _handle_enhanced

                # Create args namespace
                args = argparse.Namespace()
                args.symbol = symbols[0] if input_method == "single" else None
                args.symbols = ",".join(symbols) if input_method == "list" else None
                args.symbol_list = symbol_list if input_method == "csv" else None
                args.start = start_date
                args.end = end_date
                args.strategy = strategy
                args.risk_pct = risk_pct
                args.max_positions = max_positions
                args.cadence = cadence
                args.regime = "on" if regime_filter else "off"
                args.regime_window = regime_window
                args.run_tag = run_tag
                args.no_plot = no_plot
                args.debug = False

                if strategy == "trend_breakout":
                    args.donchian = donchian
                    args.atr = atr
                    args.atr_k = atr_k

                # Run backtest
                _handle_enhanced(args)

        monitor.end("Guided Backtest Setup")

    except KeyboardInterrupt:
        console.print("\n[yellow]Setup cancelled by user[/yellow]")
        monitor.end("Guided Backtest Setup")
    except Exception as e:
        console.print(CLITheme.error(f"Setup failed: {e}"))
        monitor.end("Guided Backtest Setup")


def run_system_check() -> None:
    """Run comprehensive system check."""
    console.print(
        Panel(
            "[bold]System Check[/bold]\n" "Comprehensive system validation and diagnostics.",
            title="[bold cyan]System Check",
            border_style="cyan",
        )
    )

    monitor = PerformanceMonitor()
    monitor.start("System Check")

    try:
        print_separator("Dependencies")
        display_dependency_status()

        print_separator("Environment")
        from .cli_utils import validate_environment

        env_ok = validate_environment()

        print_separator("System Information")
        display_system_info()

        print_separator("Configuration Profiles")
        profiles = list_available_profiles()
        if profiles:
            console.print(f"Available profiles: {', '.join(profiles)}")
        else:
            console.print("No configuration profiles found")

        print_separator("Data Directory")
        data_dir = Path("data")
        if data_dir.exists():
            console.print(f"Data directory: {data_dir.resolve()}")

            # Check subdirectories
            subdirs = ["backtests", "experiments", "logs"]
            for subdir in subdirs:
                subdir_path = data_dir / subdir
                if subdir_path.exists():
                    console.print(f"  ✓ {subdir}/")
                else:
                    console.print(f"  ⚠ {subdir}/ (not found)")
        else:
            console.print("Data directory not found")

        print_separator("Summary")
        if env_ok:
            console.print(CLITheme.success("System check passed"))
        else:
            console.print(CLITheme.warning("System check completed with warnings"))

        monitor.end("System Check")

    except Exception as e:
        console.print(CLITheme.error(f"System check failed: {e}"))
        monitor.end("System Check")
