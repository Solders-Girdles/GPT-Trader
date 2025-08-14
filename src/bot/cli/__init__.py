"""
GPT-Trader Unified CLI
Simplified command-line interface for the autonomous portfolio management system
"""

from .cli import main
from .cli_utils import (
    confirm_action,
    format_currency,
    format_number,
    format_percentage,
    print_banner,
    print_error,
    print_info,
    print_success,
    print_warning,
    setup_logging,
)
from .commands import (
    BacktestCommand,
    DashboardCommand,
    LiveCommand,
    MonitorCommand,
    OptimizeCommand,
    PaperCommand,
    WizardCommand,
)
from .ml_commands import AutoTradeCommand, MLTrainCommand

__version__ = "2.0.0"
__all__ = [
    "main",
    "BacktestCommand",
    "OptimizeCommand",
    "LiveCommand",
    "PaperCommand",
    "MonitorCommand",
    "DashboardCommand",
    "WizardCommand",
    "MLTrainCommand",
    "AutoTradeCommand",
    "setup_logging",
    "print_banner",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
    "confirm_action",
    "format_currency",
    "format_percentage",
    "format_number",
]
