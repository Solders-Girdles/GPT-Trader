"""Helpers for strategy monitor alert tests."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from gpt_trader.features.strategy_dev.monitor.metrics import PerformanceSnapshot


def create_test_snapshot(**kwargs) -> PerformanceSnapshot:
    """Create a test snapshot with defaults."""
    defaults = {
        "timestamp": datetime.now(),
        "equity": Decimal("10000"),
        "cash": Decimal("5000"),
        "positions_value": Decimal("5000"),
        "total_return": 0.0,
        "daily_return": 0.0,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "drawdown": 0.0,
        "max_drawdown": 0.0,
        "volatility": 0.0,
        "open_positions": 0,
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "current_regime": "UNKNOWN",
        "regime_confidence": 0.0,
    }
    defaults.update(kwargs)
    return PerformanceSnapshot(**defaults)
