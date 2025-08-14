"""
GPT-Trader Unified CLI
Simplified command-line interface for the autonomous portfolio management system
"""

from .cli import main
from .commands import (
    BacktestCommand,
    OptimizeCommand,
    LiveCommand,
    PaperCommand,
    MonitorCommand,
    DashboardCommand,
    WizardCommand
)
from .ml_commands import (
    MLTrainCommand,
    AutoTradeCommand
)
from .utils import (
    setup_logging,
    print_banner,
    print_success,
    print_error,
    print_warning,
    print_info,
    confirm_action,
    format_currency,
    format_percentage,
    format_number
)

__version__ = '2.0.0'
__all__ = [
    'main',
    'BacktestCommand',
    'OptimizeCommand', 
    'LiveCommand',
    'PaperCommand',
    'MonitorCommand',
    'DashboardCommand',
    'WizardCommand',
    'MLTrainCommand',
    'AutoTradeCommand',
    'setup_logging',
    'print_banner',
    'print_success',
    'print_error',
    'print_warning',
    'print_info',
    'confirm_action',
    'format_currency',
    'format_percentage',
    'format_number'
]