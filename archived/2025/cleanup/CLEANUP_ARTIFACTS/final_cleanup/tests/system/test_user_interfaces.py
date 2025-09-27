#!/usr/bin/env python3
"""Test script for verifying all new user interfaces."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import track

console = Console()


def test_imports():
    """Test that all modules can be imported."""
    console.print("[cyan]Testing imports...[/cyan]")

    modules = [
        "src.bot.cli.main_menu",
        "src.bot.cli.shortcuts",
        "src.bot.cli.help_system",
        "src.bot.cli.setup_wizard",
        "src.bot.cli.dashboard",
    ]

    for module in track(modules, description="Importing modules..."):
        try:
            __import__(module)
            console.print(f"  ✓ {module}")
        except ImportError as e:
            console.print(f"  ✗ {module}: {e}", style="red")
            return False

    return True


def test_main_menu():
    """Test main menu initialization."""
    console.print("\n[cyan]Testing Main Menu...[/cyan]")

    try:
        from src.bot.cli.main_menu import MainMenu

        menu = MainMenu()
        console.print("  ✓ MainMenu initialized")
        return True
    except Exception as e:
        console.print(f"  ✗ MainMenu failed: {e}", style="red")
        return False


def test_shortcuts():
    """Test shortcuts system."""
    console.print("\n[cyan]Testing Shortcuts...[/cyan]")

    try:
        from src.bot.cli.shortcuts import ShortcutManager

        # Test shortcut resolution
        resolved = ShortcutManager.resolve_shortcut("bt")
        if resolved == "backtest":
            console.print("  ✓ Shortcut resolution works")
        else:
            console.print("  ✗ Shortcut resolution failed", style="red")
            return False

        # Test quick actions
        action = ShortcutManager.get_quick_action("quick-test")
        if action:
            console.print("  ✓ Quick actions work")
        else:
            console.print("  ✗ Quick actions failed", style="red")
            return False

        return True
    except Exception as e:
        console.print(f"  ✗ Shortcuts failed: {e}", style="red")
        return False


def test_help_system():
    """Test help system."""
    console.print("\n[cyan]Testing Help System...[/cyan]")

    try:
        from src.bot.cli.help_system import HelpSystem

        # Test examples database
        if "backtest" in HelpSystem.EXAMPLES:
            console.print("  ✓ Help examples loaded")
        else:
            console.print("  ✗ Help examples not found", style="red")
            return False

        # Test tutorials
        if "getting_started" in HelpSystem.TUTORIALS:
            console.print("  ✓ Tutorials loaded")
        else:
            console.print("  ✗ Tutorials not found", style="red")
            return False

        return True
    except Exception as e:
        console.print(f"  ✗ Help system failed: {e}", style="red")
        return False


def test_wizard():
    """Test setup wizard."""
    console.print("\n[cyan]Testing Setup Wizard...[/cyan]")

    try:
        from src.bot.cli.setup_wizard import SetupWizard

        wizard = SetupWizard()
        console.print("  ✓ SetupWizard initialized")

        # Check config directory setup
        if wizard.config_dir:
            console.print("  ✓ Config directory configured")
        else:
            console.print("  ✗ Config directory not set", style="red")
            return False

        return True
    except Exception as e:
        console.print(f"  ✗ Setup wizard failed: {e}", style="red")
        return False


def test_dashboard():
    """Test dashboard."""
    console.print("\n[cyan]Testing Dashboard...[/cyan]")

    try:
        from src.bot.cli.dashboard import TradingDashboard

        dashboard = TradingDashboard()
        console.print("  ✓ TradingDashboard initialized")

        # Test layout creation
        layout = dashboard.create_overview_layout()
        if layout:
            console.print("  ✓ Dashboard layout created")
        else:
            console.print("  ✗ Dashboard layout failed", style="red")
            return False

        return True
    except Exception as e:
        console.print(f"  ✗ Dashboard failed: {e}", style="red")
        return False


def test_cli_integration():
    """Test CLI integration."""
    console.print("\n[cyan]Testing CLI Integration...[/cyan]")

    try:
        # Test that main CLI can import everything
        from src.bot.cli.__main__ import create_parser

        parser = create_parser()

        # Check for new commands
        if parser._subparsers:
            actions = list(parser._subparsers._group_actions[0].choices.keys())
            required = ["menu", "dashboard", "wizard", "shortcuts", "help"]

            missing = [cmd for cmd in required if cmd not in actions]
            if missing:
                console.print(f"  ✗ Missing commands: {missing}", style="red")
                return False
            else:
                console.print("  ✓ All new commands registered")

        return True
    except Exception as e:
        console.print(f"  ✗ CLI integration failed: {e}", style="red")
        return False


def main():
    """Run all tests."""
    console.print(
        Panel("[bold cyan]Testing User Interface Components[/bold cyan]", border_style="cyan")
    )

    tests = [
        ("Imports", test_imports),
        ("Main Menu", test_main_menu),
        ("Shortcuts", test_shortcuts),
        ("Help System", test_help_system),
        ("Setup Wizard", test_wizard),
        ("Dashboard", test_dashboard),
        ("CLI Integration", test_cli_integration),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            console.print(f"\n[red]Test {name} crashed: {e}[/red]")
            results.append((name, False))

    # Summary
    console.print("\n" + "=" * 50)
    console.print(Panel("[bold]Test Summary[/bold]", border_style="cyan"))

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[green]✓ PASSED[/green]" if result else "[red]✗ FAILED[/red]"
        console.print(f"  {name}: {status}")

    console.print(f"\n[bold]Total: {passed}/{total} passed[/bold]")

    if passed == total:
        console.print("\n[bold green]✨ All tests passed! User interfaces are ready.[/bold green]")
    else:
        console.print(f"\n[bold red]⚠️  {total - passed} tests failed.[/bold red]")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
