"""
DEPRECATED: Portfolio widgets have been moved to the portfolio package.

This module is maintained for backward compatibility only.
Please update your imports to use the new locations:

    # Old (deprecated):
    from gpt_trader.tui.widgets.positions import PositionsWidget, OrdersWidget, TradesWidget

    # New (preferred):
    from gpt_trader.tui.widgets.portfolio import PositionsWidget, OrdersWidget, TradesWidget

This file will be removed in a future release.
"""

from __future__ import annotations

import warnings

from gpt_trader.tui.widgets.portfolio import (
    OrdersWidget,
    PositionsWidget,
    TradesWidget,
)

# Issue deprecation warning on import
warnings.warn(
    "Import from gpt_trader.tui.widgets.positions is deprecated. "
    "Use gpt_trader.tui.widgets.portfolio instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PositionsWidget", "OrdersWidget", "TradesWidget"]
