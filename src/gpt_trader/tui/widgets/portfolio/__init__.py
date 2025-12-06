"""
Portfolio widgets for displaying positions, orders, and trades.

This package contains focused, single-responsibility widgets extracted
from the original positions.py as part of the TUI architecture refactoring.
"""

from gpt_trader.tui.widgets.portfolio.orders_widget import OrdersWidget
from gpt_trader.tui.widgets.portfolio.positions_widget import PositionsWidget
from gpt_trader.tui.widgets.portfolio.trades_widget import TradesWidget

__all__ = ["PositionsWidget", "OrdersWidget", "TradesWidget"]
