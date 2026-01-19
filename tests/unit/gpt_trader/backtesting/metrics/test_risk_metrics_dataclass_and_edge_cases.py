"""Tests for RiskMetrics dataclass and calculate_risk_metrics edge cases."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.backtesting.metrics.risk import RiskMetrics, calculate_risk_metrics


class TestRiskMetricsDataclass:
    """Tests for RiskMetrics dataclass."""

    def test_all_fields_accessible(self) -> None:
        metrics = RiskMetrics(
            max_drawdown_pct=Decimal("10"),
            max_drawdown_usd=Decimal("1000"),
            avg_drawdown_pct=Decimal("5"),
            drawdown_duration_days=10,
            total_return_pct=Decimal("15"),
            annualized_return_pct=Decimal("18"),
            daily_return_avg=Decimal("0.05"),
            daily_return_std=Decimal("1.2"),
            sharpe_ratio=Decimal("1.5"),
            sortino_ratio=Decimal("2.0"),
            calmar_ratio=Decimal("1.8"),
            volatility_annualized=Decimal("20"),
            downside_volatility=Decimal("15"),
            max_leverage_used=Decimal("3"),
            avg_leverage_used=Decimal("2"),
            time_in_market_pct=Decimal("75"),
            var_95_daily=Decimal("2"),
            var_99_daily=Decimal("3"),
        )
        assert metrics.max_drawdown_pct == Decimal("10")
        assert metrics.sharpe_ratio == Decimal("1.5")
        assert metrics.drawdown_duration_days == 10

    def test_optional_ratios_can_be_none(self) -> None:
        metrics = RiskMetrics(
            max_drawdown_pct=Decimal("0"),
            max_drawdown_usd=Decimal("0"),
            avg_drawdown_pct=Decimal("0"),
            drawdown_duration_days=0,
            total_return_pct=Decimal("0"),
            annualized_return_pct=Decimal("0"),
            daily_return_avg=Decimal("0"),
            daily_return_std=Decimal("0"),
            sharpe_ratio=None,
            sortino_ratio=None,
            calmar_ratio=None,
            volatility_annualized=Decimal("0"),
            downside_volatility=Decimal("0"),
            max_leverage_used=Decimal("1"),
            avg_leverage_used=Decimal("1"),
            time_in_market_pct=Decimal("0"),
            var_95_daily=Decimal("0"),
            var_99_daily=Decimal("0"),
        )
        assert metrics.sharpe_ratio is None
        assert metrics.sortino_ratio is None
        assert metrics.calmar_ratio is None


class TestCalculateRiskMetricsEdgeCases:
    """Edge case tests for calculate_risk_metrics."""

    def test_zero_equity_returns_empty_metrics(self) -> None:
        """Test that equity curves starting at zero return empty metrics."""
        now = datetime.now()
        broker = MagicMock()
        broker._initial_equity = Decimal("0")
        broker.get_equity.return_value = Decimal("100")
        broker._max_drawdown = Decimal("0")
        broker._max_drawdown_usd = Decimal("0")
        broker.positions = {}
        # Equity curve with zero values generates no valid returns
        broker.get_equity_curve.return_value = [
            (now, Decimal("0")),
            (now + timedelta(days=1), Decimal("0")),
        ]
        result = calculate_risk_metrics(broker)
        assert result.sharpe_ratio is None

    def test_same_day_equity_curve(self) -> None:
        """Test equity curve within same day (years=0 path)."""
        now = datetime.now()
        broker = MagicMock()
        broker._initial_equity = Decimal("10000")
        broker.get_equity.return_value = Decimal("10500")
        broker._max_drawdown = Decimal("0")
        broker._max_drawdown_usd = Decimal("0")
        broker.positions = {}
        # Same timestamp for all entries (0 days duration)
        broker.get_equity_curve.return_value = [
            (now, Decimal("10000")),
            (now, Decimal("10100")),
            (now, Decimal("10500")),
        ]
        result = calculate_risk_metrics(broker)
        # Should still calculate return even if years=0
        assert result.total_return_pct == Decimal("5")
