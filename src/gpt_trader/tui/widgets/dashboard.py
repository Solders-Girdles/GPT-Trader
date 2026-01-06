"""
Dashboard widgets for the High-Fidelity TUI Bento Layout.

This module re-exports widgets from their focused submodules for backwards compatibility.
All widgets implement StateObserver protocol to receive state updates
via StateRegistry broadcast instead of direct property assignment.

Submodules:
- dashboard_market: TickerRow, MarketPulseWidget, calculate_price_change_percent
- dashboard_position: PositionCardWidget
- dashboard_system: SystemThresholds, DEFAULT_THRESHOLDS, SystemMonitorWidget
"""

from __future__ import annotations

# Re-export from market module
from gpt_trader.tui.widgets.dashboard_market import (
    MarketPulseWidget,
    TickerRow,
    calculate_price_change_percent,
)

# Re-export from position module
from gpt_trader.tui.widgets.dashboard_position import PositionCardWidget

# Re-export from system module
from gpt_trader.tui.widgets.dashboard_system import (
    DEFAULT_THRESHOLDS,
    SystemMonitorWidget,
    SystemThresholds,
)

__all__ = [
    # Market widgets
    "calculate_price_change_percent",
    "TickerRow",
    "MarketPulseWidget",
    # Position widgets
    "PositionCardWidget",
    # System widgets
    "SystemThresholds",
    "DEFAULT_THRESHOLDS",
    "SystemMonitorWidget",
]
