"""Tests for empty risk metrics defaults and time-in-market estimates."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.backtesting.metrics.risk import (
    RiskMetrics,
    _calculate_time_in_market,
    _empty_risk_metrics,
)


class TestEmptyRiskMetrics:
    """Tests for _empty_risk_metrics function."""

    def test_returns_valid_risk_metrics(self) -> None:
        result = _empty_risk_metrics()
        assert isinstance(result, RiskMetrics)

    def test_all_values_are_zero_or_none(self) -> None:
        result = _empty_risk_metrics()
        assert result.max_drawdown_pct == Decimal("0")
        assert result.max_drawdown_usd == Decimal("0")
        assert result.total_return_pct == Decimal("0")
        assert result.sharpe_ratio is None
        assert result.sortino_ratio is None
        assert result.calmar_ratio is None

    def test_leverage_defaults_to_one(self) -> None:
        result = _empty_risk_metrics()
        assert result.max_leverage_used == Decimal("1")
        assert result.avg_leverage_used == Decimal("1")


class TestCalculateTimeInMarket:
    """Tests for _calculate_time_in_market function."""

    def test_empty_equity_curve_returns_zero(self) -> None:
        broker = MagicMock()
        broker.get_equity_curve.return_value = []
        result = _calculate_time_in_market(broker)
        assert result == Decimal("0")

    def test_single_point_returns_zero(self) -> None:
        broker = MagicMock()
        broker.get_equity_curve.return_value = [(datetime.now(), Decimal("100"))]
        result = _calculate_time_in_market(broker)
        assert result == Decimal("0")

    def test_no_positions_returns_zero(self) -> None:
        now = datetime.now()
        broker = MagicMock()
        broker.get_equity_curve.return_value = [
            (now, Decimal("100")),
            (now + timedelta(hours=1), Decimal("101")),
        ]
        broker.positions = {}
        result = _calculate_time_in_market(broker)
        assert result == Decimal("0")

    def test_with_positions_returns_estimate(self) -> None:
        now = datetime.now()
        broker = MagicMock()
        broker.get_equity_curve.return_value = [
            (now + timedelta(hours=i), Decimal("100")) for i in range(10)
        ]
        broker.positions = {"BTC-USD": MagicMock()}
        result = _calculate_time_in_market(broker)
        # Should return ~50% estimate when positions exist
        assert result > Decimal("0")
