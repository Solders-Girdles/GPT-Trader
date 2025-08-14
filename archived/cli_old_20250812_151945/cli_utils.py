"""Utility functions and classes for the enhanced CLI."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generator

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

console = Console()


class CLITheme:
    """Consistent theming for CLI output."""

    # Color scheme
    PRIMARY = "cyan"
    SUCCESS = "green"
    WARNING = "yellow"
    ERROR = "red"
    INFO = "blue"
    SECONDARY = "magenta"
    MUTED = "dim"

    @staticmethod
    def format_description(text: str) -> str:
        """Format description text with styling."""
        return f"[bold {CLITheme.PRIMARY}]{text}[/bold {CLITheme.PRIMARY}]"

    @staticmethod
    def format_epilog(text: str) -> str:
        """Format epilog text with styling."""
        return f"[dim]{text}[/dim]"

    @staticmethod
    def success(text: str) -> str:
        return f"[{CLITheme.SUCCESS}]✓[/{CLITheme.SUCCESS}] {text}"

    @staticmethod
    def warning(text: str) -> str:
        return f"[{CLITheme.WARNING}]⚠️[/{CLITheme.WARNING}] {text}"

    @staticmethod
    def error(text: str) -> str:
        return f"[{CLITheme.ERROR}]✗[/{CLITheme.ERROR}] {text}"

    @staticmethod
    def info(text: str) -> str:
        return f"[{CLITheme.INFO}]ℹ[/{CLITheme.INFO}] {text}"

    @staticmethod
    def highlight(text: str) -> str:
        return f"[bold {CLITheme.SECONDARY}]{text}[/bold {CLITheme.SECONDARY}]"

    @staticmethod
    def muted(text: str) -> str:
        return f"[{CLITheme.MUTED}]{text}[/{CLITheme.MUTED}]"


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self) -> None:
        self.start_time = None
        self.metrics = {}

    def start(self, operation: str = "Operation") -> None:
        """Start timing an operation."""
        self.start_time = time.time()
        self.metrics[operation] = {"start": self.start_time}
        console.print(f"[dim]Starting {operation}...[/dim]")

    def end(self, operation: str = "Operation") -> None:
        """End timing an operation and display results."""
        if self.start_time and operation in self.metrics:
            end_time = time.time()
            duration = end_time - self.start_time
            self.metrics[operation]["end"] = end_time
            self.metrics[operation]["duration"] = duration

            if duration > 1.0:
                console.print(CLITheme.success(f"{operation} completed in {duration:.2f}s"))
            else:
                console.print(CLITheme.success(f"{operation} completed in {duration*1000:.0f}ms"))

    def get_duration(self, operation: str = "Operation") -> float:
        """Get duration for an operation."""
        if operation in self.metrics:
            return self.metrics[operation].get("duration", 0)
        return 0


class DataValidator:
    """Enhanced data validation utilities with security hardening."""

    def __init__(self):
        """Initialize with security-focused validator."""
        from bot.security.input_validation import InputValidator

        self.validator = InputValidator(strict_mode=True)

    @staticmethod
    def validate_date(date_str: str) -> datetime:
        """Validate and parse date string with security checks."""
        from bot.security.input_validation import get_validator

        return get_validator().validate_date(date_str)

    @staticmethod
    def validate_date_range(start: str, end: str) -> tuple[datetime, datetime]:
        """Validate date range with security checks."""
        from bot.security.input_validation import validate_date_range

        start_date, end_date = validate_date_range(start, end)

        if end_date > datetime.now():
            console.print(CLITheme.warning("End date is in the future"))

        return start_date, end_date

    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """Validate stock symbol with security checks."""
        from bot.security.input_validation import validate_trading_symbol

        return validate_trading_symbol(symbol)

    @staticmethod
    def validate_symbols(symbols: list[str]) -> list[str]:
        """Validate list of symbols."""
        if not symbols:
            raise ValueError("At least one symbol must be provided")

        validated = []
        for symbol in symbols:
            validated.append(DataValidator.validate_symbol(symbol))

        return list(set(validated))  # Remove duplicates

    @staticmethod
    def validate_percentage(value: float, name: str = "Percentage") -> float:
        """Validate percentage value."""
        if not isinstance(value, int | float):
            raise ValueError(f"{name} must be a number")

        if value < 0 or value > 100:
            raise ValueError(f"{name} must be between 0 and 100")

        return float(value)

    @staticmethod
    def validate_file_path(path: str, must_exist: bool = True) -> Path:
        """Validate file path."""
        file_path = Path(path)

        if must_exist and not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        return file_path


class InteractivePrompts:
    """Interactive prompt utilities."""

    @staticmethod
    def prompt_symbol(default: str = "AAPL") -> str:
        """Prompt for a stock symbol."""
        return Prompt.ask("Enter stock symbol", default=default, show_default=True).strip().upper()

    @staticmethod
    def prompt_symbols() -> list[str]:
        """Prompt for multiple stock symbols."""
        symbols_input = Prompt.ask("Enter symbols (comma-separated)", default="AAPL,MSFT,GOOGL")
        return [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

    @staticmethod
    def prompt_date_range() -> tuple[str, str]:
        """Prompt for date range."""
        start = Prompt.ask(
            "Start date (YYYY-MM-DD)",
            default=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        )
        end = Prompt.ask("End date (YYYY-MM-DD)", default=datetime.now().strftime("%Y-%m-%d"))
        return start, end

    @staticmethod
    def prompt_strategy() -> str:
        """Prompt for strategy selection."""
        strategies = ["trend_breakout", "demo_ma"]
        strategy = Prompt.ask("Select strategy", choices=strategies, default="trend_breakout")
        return strategy

    @staticmethod
    def prompt_risk_settings() -> tuple[float, int]:
        """Prompt for risk settings."""
        risk_pct = FloatPrompt.ask("Risk per trade (%)", default=0.5)
        max_positions = IntPrompt.ask("Max positions", default=10)
        return risk_pct, max_positions


def create_banner() -> Panel:
    """Create an ASCII art banner for the CLI."""
    banner_text = """
    ╔═══════════════════════════════════════════╗
    ║       🤖 GPT-TRADER PLATFORM v2.0 🤖      ║
    ║   AI-Powered Trading Strategy Development  ║
    ╚═══════════════════════════════════════════╝
    """
    return Panel(
        Text(banner_text, justify="center", style="bold cyan"), border_style="cyan", padding=(1, 2)
    )


def setup_logging() -> logging.Logger:
    """Setup enhanced logging with Rich handler."""
    # Clear any existing handlers to prevent duplicates
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    # Add a single Rich handler
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
        force=True,  # Force reconfiguration
    )
    return logging.getLogger("gpt-trader")


def load_config_profile(profile_name: str) -> dict[str, Any]:
    """Load a configuration profile from disk."""
    profile_dir = Path.home() / ".gpt-trader" / "profiles"
    profile_path = profile_dir / f"{profile_name}.yaml"

    if not profile_path.exists():
        raise FileNotFoundError(f"Profile '{profile_name}' not found at {profile_path}")

    with open(profile_path) as f:
        return yaml.safe_load(f)


def save_config_profile(profile_name: str, config: dict[str, Any]) -> None:
    """Save a configuration profile to disk."""
    profile_dir = Path.home() / ".gpt-trader" / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    profile_path = profile_dir / f"{profile_name}.yaml"

    with open(profile_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    console.print(CLITheme.success(f"Profile saved to {profile_path}"))


def list_available_profiles() -> list[str]:
    """List all available configuration profiles."""
    profile_dir = Path.home() / ".gpt-trader" / "profiles"
    if not profile_dir.exists():
        return []

    profiles = []
    for profile_file in profile_dir.glob("*.yaml"):
        profiles.append(profile_file.stem)

    return sorted(profiles)


def validate_environment() -> bool:
    """Validate that the environment is properly configured."""
    checks = []

    # Check data directory
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    checks.append(("Data directory", True))

    # Check for API keys if needed
    alpaca_configured = bool(os.getenv("ALPACA_API_KEY_ID") and os.getenv("ALPACA_API_SECRET_KEY"))

    if not alpaca_configured:
        console.print(
            CLITheme.warning("Alpaca API keys not configured (required for paper/live trading)")
        )

    # Check Python version
    py_version = sys.version_info
    py_ok = py_version >= (3, 8)
    checks.append(("Python 3.8+", py_ok))

    # Check available disk space
    try:
        disk_usage = shutil.disk_usage(".")
        free_gb = disk_usage.free / (1024**3)
        if free_gb < 1.0:
            console.print(CLITheme.warning(f"Low disk space: {free_gb:.1f}GB free"))
        checks.append(("Disk space", free_gb >= 0.1))
    except OSError:
        checks.append(("Disk space", True))  # Skip if can't check

    return all(check[1] for check in checks)


def handle_error(message: str, show_traceback: bool = False) -> None:
    """Handle and display errors consistently."""
    error_panel = Panel(CLITheme.error(message), title="[bold red]Error", border_style="red")
    console.print(error_panel)

    if show_traceback:
        import traceback

        console.print(Syntax(traceback.format_exc(), "python", theme="monokai"))


def create_progress_bar(description: str = "Processing...") -> Progress:
    """Create a consistent progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )


@contextmanager
def progress_context(description: str = "Processing...") -> Generator[tuple[Any, Any], None, None]:
    """Context manager for progress tracking."""
    with create_progress_bar(description) as progress:
        task = progress.add_task(description, total=None)
        yield progress, task


def format_results_table(data: list[dict], title: str = "Results") -> Table:
    """Create a formatted results table."""
    if not data:
        return Table(title=title, show_header=False)

    table = Table(title=title, show_header=True, header_style="bold cyan")

    # Add columns based on first row
    for key in data[0].keys():
        table.add_column(key.replace("_", " ").title())

    # Add rows
    for row in data:
        table.add_row(*[str(v) for v in row.values()])

    return table


def format_performance_summary(metrics: dict[str, Any]) -> Table:
    """Format performance metrics as a table."""
    table = Table(title="Performance Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for metric, value in metrics.items():
        if isinstance(value, float):
            formatted_value = f"{value:.2f}"
        else:
            formatted_value = str(value)
        table.add_row(metric, formatted_value)

    return table


def confirm_action(prompt: str, default: bool = False) -> bool:
    """Ask for user confirmation."""
    return Confirm.ask(prompt, default=default)


def prompt_with_suggestions(prompt: str, suggestions: list[str], default: str = "") -> str:
    """Prompt with suggestions."""
    console.print(f"[dim]Suggestions: {', '.join(suggestions)}[/dim]")
    return Prompt.ask(prompt, default=default)


def export_results(data: dict | list, format: str = "json", filename: str = None) -> str:
    """Export results in various formats."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}"

    if format.lower() == "json":
        from bot.utils.config import ConfigManager

        filepath = f"{filename}.json"
        ConfigManager.save_json_config(data, filepath)
    elif format.lower() == "yaml":
        from bot.utils.config import ConfigManager

        filepath = f"{filename}.yaml"
        ConfigManager.save_yaml_config(data, filepath)
    elif format.lower() == "csv":
        import pandas as pd

        filepath = f"{filename}.csv"
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        df.to_csv(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    console.print(CLITheme.success(f"Results exported to {filepath}"))
    return filepath


def display_system_info() -> None:
    """Display comprehensive system information."""
    import platform

    import psutil

    info_table = Table(title="System Information", show_header=True, header_style="bold cyan")
    info_table.add_column("Component", style="cyan")
    info_table.add_column("Value", style="green")

    # Python info
    info_table.add_row(
        "Python Version",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    info_table.add_row("Platform", platform.platform())

    # Memory info
    memory = psutil.virtual_memory()
    info_table.add_row("Memory Total", f"{memory.total / (1024**3):.1f}GB")
    info_table.add_row("Memory Available", f"{memory.available / (1024**3):.1f}GB")
    info_table.add_row("Memory Usage", f"{memory.percent:.1f}%")

    # Disk info
    disk = shutil.disk_usage(".")
    info_table.add_row("Disk Total", f"{disk.total / (1024**3):.1f}GB")
    info_table.add_row("Disk Free", f"{disk.free / (1024**3):.1f}GB")

    # Working directory
    info_table.add_row("Working Directory", str(Path.cwd()))

    console.print(info_table)


def create_summary_report(results: dict[str, Any], output_dir: Path = None) -> str:
    """Create a comprehensive summary report."""
    if output_dir is None:
        output_dir = Path("reports")

    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"summary_report_{timestamp}.md"

    with open(report_file, "w") as f:
        f.write("# GPT-Trader Summary Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Results Summary\n\n")
        for key, value in results.items():
            f.write(f"**{key}**: {value}\n\n")

    console.print(CLITheme.success(f"Summary report saved to {report_file}"))
    return str(report_file)


def check_dependencies() -> dict[str, bool]:
    """Check if all required dependencies are available."""
    dependencies = {
        "pandas": False,
        "numpy": False,
        "yfinance": False,
        "alpaca-py": False,
        "rich": False,
        "pytz": False,
        "pyyaml": False,
    }

    for dep in dependencies.keys():
        try:
            __import__(dep.replace("-", "_"))
            dependencies[dep] = True
        except ImportError:
            pass

    return dependencies


def display_dependency_status() -> None:
    """Display dependency status."""
    deps = check_dependencies()

    table = Table(title="Dependency Status", show_header=True, header_style="bold cyan")
    table.add_column("Dependency", style="cyan")
    table.add_column("Status", style="green")

    for dep, available in deps.items():
        status = "✓ Available" if available else "✗ Missing"
        table.add_row(dep, status)

    console.print(table)

    missing = [dep for dep, available in deps.items() if not available]
    if missing:
        console.print(CLITheme.warning(f"Missing dependencies: {', '.join(missing)}"))
        console.print("Install with: pip install " + " ".join(missing))


def get_terminal_size() -> tuple[int, int]:
    """Get terminal size for responsive layouts."""
    return shutil.get_terminal_size()


def is_interactive() -> bool:
    """Check if running in interactive mode."""
    return sys.stdout.isatty() and not os.getenv("CI")


def clear_screen() -> None:
    """Clear the terminal screen."""
    console.clear()


def print_separator(title: str = None) -> None:
    """Print a separator line."""
    if title:
        console.print(Rule(title, style="cyan"))
    else:
        console.print(Rule(style="dim"))


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"


def get_file_info(filepath: Path) -> dict[str, Any]:
    """Get comprehensive file information."""
    stat = filepath.stat()
    return {
        "size": format_file_size(stat.st_size),
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "created": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
        "permissions": oct(stat.st_mode)[-3:],
    }


def display_file_info(filepath: Path) -> None:
    """Display file information in a formatted table."""
    if not filepath.exists():
        console.print(CLITheme.error(f"File not found: {filepath}"))
        return

    info = get_file_info(filepath)

    table = Table(
        title=f"File Information: {filepath.name}", show_header=True, header_style="bold cyan"
    )
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    for prop, value in info.items():
        table.add_row(prop.title(), str(value))

    console.print(table)
