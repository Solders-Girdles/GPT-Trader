"""
GPT-Trader CLI Utilities
Shared utility functions for CLI operations
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Initialize Rich console
console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            RichHandler(console=console, rich_tracebacks=True, show_time=False, show_path=False)
        ],
    )


def print_banner() -> None:
    """Print GPT-Trader banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║     ██████╗ ██████╗ ████████╗    ████████╗██████╗    ║
    ║    ██╔════╝ ██╔══██╗╚══██╔══╝    ╚══██╔══╝██╔══██╗   ║
    ║    ██║  ███╗██████╔╝   ██║  █████╗  ██║   ██████╔╝   ║
    ║    ██║   ██║██╔═══╝    ██║  ╚════╝  ██║   ██╔══██╗   ║
    ║    ╚██████╔╝██║        ██║          ██║   ██║  ██║   ║
    ║     ╚═════╝ ╚═╝        ╚═╝          ╚═╝   ╚═╝  ╚═╝   ║
    ║                                                       ║
    ║         Autonomous Portfolio Management System        ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def get_version() -> str:
    """Get GPT-Trader version"""
    try:
        from importlib.metadata import version

        return version("gpt-trader")
    except:
        return "2.0.0"


def print_success(message: str) -> None:
    """Print success message"""
    console.print(f"✅ {message}", style="bold green")


def print_error(message: str) -> None:
    """Print error message"""
    console.print(f"❌ {message}", style="bold red")


def print_warning(message: str) -> None:
    """Print warning message"""
    console.print(f"⚠️  {message}", style="bold yellow")


def print_info(message: str) -> None:
    """Print info message"""
    console.print(f"ℹ️  {message}", style="cyan")


def confirm_action(prompt: str, default: bool = False) -> bool:
    """Ask for confirmation"""
    suffix = " [Y/n]" if default else " [y/N]"
    response = console.input(f"{prompt}{suffix}: ").strip().lower()

    if not response:
        return default

    return response in ["y", "yes"]


def print_table(headers: list[str], rows: list[list[Any]], title: str | None = None) -> None:
    """Print a formatted table"""
    table = Table(title=title, show_header=True, header_style="bold magenta")

    for header in headers:
        table.add_column(header)

    for row in rows:
        table.add_row(*[str(cell) for cell in row])

    console.print(table)


def format_currency(value: float, symbol: str = "$") -> str:
    """Format currency value"""
    return f"{symbol}{value:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage value"""
    return f"{value:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with thousands separator"""
    if decimals == 0:
        return f"{int(value):,}"
    return f"{value:,.{decimals}f}"


def format_date(date: datetime, format: str = "%Y-%m-%d") -> str:
    """Format date"""
    return date.strftime(format)


def format_timedelta(seconds: float) -> str:
    """Format time duration"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


def create_progress_bar(description: str = "Processing...") -> Progress:
    """Create a progress bar"""
    return Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    )


def parse_date(date_str: str) -> datetime:
    """Parse date string"""
    formats = ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y", "%d-%m-%Y", "%d/%m/%Y"]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Could not parse date: {date_str}")


def parse_symbols(symbol_str: str) -> list[str]:
    """Parse symbol string or file"""
    # Check if it's a file path
    path = Path(symbol_str)
    if path.exists() and path.suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(path)

        # Try common column names
        for col in ["symbol", "Symbol", "SYMBOL", "ticker", "Ticker"]:
            if col in df.columns:
                return df[col].tolist()

        # Use first column if no match
        return df.iloc[:, 0].tolist()

    # Otherwise parse as comma-separated list
    return [s.strip().upper() for s in symbol_str.split(",")]


def validate_symbol(symbol: str) -> bool:
    """Validate stock symbol"""
    # Basic validation - alphanumeric, 1-5 characters
    if not symbol or len(symbol) > 5:
        return False

    return symbol.isalnum()


def get_config_path() -> Path:
    """Get configuration file path"""
    # Check multiple locations
    paths = [
        Path(".env.local"),
        Path(".env"),
        Path.home() / ".gpt-trader" / "config.json",
        Path("/etc/gpt-trader/config.json"),
    ]

    for path in paths:
        if path.exists():
            return path

    return Path(".env.local")


def load_config() -> dict:
    """Load configuration from file"""
    config_path = get_config_path()

    if not config_path.exists():
        return {}

    if config_path.suffix == ".json":
        import json

        with open(config_path) as f:
            return json.load(f)
    else:
        # Parse .env file
        config = {}
        with open(config_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip()
        return config


def display_results(results: dict, title: str = "Results") -> None:
    """Display results in a formatted panel"""
    content = "\n".join([f"{k}: {v}" for k, v in results.items()])
    panel = Panel(content, title=title, border_style="green")
    console.print(panel)


def handle_exception(exc: Exception, verbose: bool = False) -> None:
    """Handle exception with appropriate formatting"""
    if verbose:
        console.print_exception()
    else:
        print_error(str(exc))
        print_info("Use -v for more details")


class Timer:
    """Context manager for timing operations"""

    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            print_info(f"{self.description} completed in {format_timedelta(elapsed)}")
        else:
            print_error(f"{self.description} failed after {format_timedelta(elapsed)}")


def create_backup(file_path: Path, backup_dir: Path | None = None) -> Path:
    """Create backup of a file"""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if backup_dir is None:
        backup_dir = file_path.parent / "backups"

    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"

    import shutil

    shutil.copy2(file_path, backup_path)

    return backup_path
