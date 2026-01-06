"""TUI widgets package.

All widgets are exported from this module for convenient importing.
"""

from .account import AccountWidget
from .alert_inbox import AlertInbox
from .cfm_balance import CFMBalanceWidget
from .config import ConfigModal
from .execution import ExecutionWidget
from .footer import ContextualFooter
from .live_warning_modal import LiveWarningModal
from .logs import LogWidget
from .market import MarketWatchWidget
from .mode_indicator import ModeIndicator
from .mode_info_modal import ModeInfoModal
from .mode_selector import ModeSelector
from .onboarding_checklist import OnboardingChecklist
from .performance_dashboard import PerformanceDashboardWidget
from .portfolio import OrdersWidget, PositionsWidget, TradesWidget
from .portfolio_widget import PortfolioWidget
from .risk import RiskWidget
from .slim_status import SlimStatusWidget
from .status import BotStatusWidget
from .strategy import StrategyWidget
from .system import SystemHealthWidget
from .trading_stats import TradingStatsWidget

__all__ = [
    "AccountWidget",
    "AlertInbox",
    "BotStatusWidget",
    "CFMBalanceWidget",
    "ConfigModal",
    "ContextualFooter",
    "ExecutionWidget",
    "LiveWarningModal",
    "LogWidget",
    "MarketWatchWidget",
    "ModeIndicator",
    "ModeInfoModal",
    "ModeSelector",
    "OnboardingChecklist",
    "OrdersWidget",
    "PerformanceDashboardWidget",
    "PortfolioWidget",
    "PositionsWidget",
    "RiskWidget",
    "SlimStatusWidget",
    "StrategyWidget",
    "SystemHealthWidget",
    "TradesWidget",
    "TradingStatsWidget",
]
