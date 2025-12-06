"""
DEPRECATED: This module has been reorganized.

PortfolioWidget has been moved to portfolio_widget.py.
PositionsWidget, OrdersWidget, TradesWidget have been moved to the portfolio/ package.

Please update your imports:

    # For PortfolioWidget:
    from gpt_trader.tui.widgets.portfolio_widget import PortfolioWidget

    # For individual widgets:
    from gpt_trader.tui.widgets.portfolio import PositionsWidget, OrdersWidget, TradesWidget

This file will be removed in a future release.
"""

from __future__ import annotations

import warnings

# Re-export from new locations for backward compatibility
from gpt_trader.tui.widgets.portfolio_widget import PortfolioWidget  # noqa: F401

# Issue deprecation warning on import
warnings.warn(
    "Import from gpt_trader.tui.widgets.portfolio is deprecated for PortfolioWidget. "
    "Use gpt_trader.tui.widgets.portfolio_widget instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PortfolioWidget"]
