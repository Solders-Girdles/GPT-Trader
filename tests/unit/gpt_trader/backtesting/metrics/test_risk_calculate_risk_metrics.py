"""Tests for calculate_risk_metrics behavior."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.backtesting.metrics.risk import calculate_risk_metrics


class TestCalculateRiskMetrics:
    """Tests for calculate_risk_metrics function."""

    def _create_mock_broker(
        self,
        initial_equity: float = 10000.0,
        final_equity: float = 11000.0,
        equity_curve: list[tuple[datetime, Decimal]] | None = None,
        max_drawdown: float = 0.05,
    ) -> MagicMock:
        broker = MagicMock()
        broker._initial_equity = Decimal(str(initial_equity))
        broker.get_equity.return_value = Decimal(str(final_equity))
        broker._max_drawdown = Decimal(str(max_drawdown))
        broker._max_drawdown_usd = Decimal(str(initial_equity * max_drawdown))
        broker.positions = {}

        if equity_curve is None:
            now = datetime.now()
            equity_curve = [
                (now + timedelta(days=i), Decimal(str(initial_equity + i * 100))) for i in range(30)
            ]
        broker.get_equity_curve.return_value = equity_curve

        return broker

    def test_insufficient_data_returns_empty_metrics(self) -> None:
        broker = self._create_mock_broker()
        broker.get_equity_curve.return_value = []
        result = calculate_risk_metrics(broker)
        assert result.max_drawdown_pct == Decimal("0")
        assert result.sharpe_ratio is None

    def test_single_equity_point_returns_empty(self) -> None:
        broker = self._create_mock_broker()
        broker.get_equity_curve.return_value = [(datetime.now(), Decimal("10000"))]
        result = calculate_risk_metrics(broker)
        assert result.total_return_pct == Decimal("0")

    def test_calculates_total_return(self) -> None:
        broker = self._create_mock_broker(
            initial_equity=10000.0,
            final_equity=12000.0,
        )
        result = calculate_risk_metrics(broker)
        # 20% return
        assert float(result.total_return_pct) == pytest.approx(20.0, rel=0.1)

    def test_calculates_sharpe_ratio(self) -> None:
        # Create a broker with consistent positive returns
        now = datetime.now()
        equity_curve = [(now + timedelta(days=i), Decimal(str(10000 + i * 50))) for i in range(100)]
        broker = self._create_mock_broker(
            initial_equity=10000.0,
            final_equity=float(equity_curve[-1][1]),
            equity_curve=equity_curve,
        )
        result = calculate_risk_metrics(broker)
        # Should have a positive Sharpe ratio for positive returns
        assert result.sharpe_ratio is not None
        assert result.sharpe_ratio > 0

    def test_calculates_volatility(self) -> None:
        broker = self._create_mock_broker()
        result = calculate_risk_metrics(broker)
        # Volatility should be computed
        assert result.volatility_annualized >= Decimal("0")
        assert result.daily_return_std >= Decimal("0")

    def test_calculates_var_metrics(self) -> None:
        broker = self._create_mock_broker()
        result = calculate_risk_metrics(broker)
        # VaR should be calculated
        assert result.var_95_daily is not None
        assert result.var_99_daily is not None

    def test_calmar_ratio_when_drawdown_exists(self) -> None:
        broker = self._create_mock_broker(max_drawdown=0.1)
        result = calculate_risk_metrics(broker)
        # Calmar should be calculated when drawdown > 0
        assert result.calmar_ratio is not None

    def test_calmar_ratio_none_when_no_drawdown(self) -> None:
        broker = self._create_mock_broker(max_drawdown=0.0)
        result = calculate_risk_metrics(broker)
        # Calmar should be None when no drawdown
        assert result.calmar_ratio is None

    def test_uses_custom_risk_free_rate(self) -> None:
        broker = self._create_mock_broker()
        result_low_rfr = calculate_risk_metrics(broker, risk_free_rate=Decimal("0.01"))
        result_high_rfr = calculate_risk_metrics(broker, risk_free_rate=Decimal("0.10"))
        # Higher risk-free rate should result in lower Sharpe
        if result_low_rfr.sharpe_ratio and result_high_rfr.sharpe_ratio:
            assert result_low_rfr.sharpe_ratio >= result_high_rfr.sharpe_ratio

    def test_handles_negative_returns(self) -> None:
        now = datetime.now()
        equity_curve = [(now + timedelta(days=i), Decimal(str(10000 - i * 100))) for i in range(30)]
        broker = self._create_mock_broker(
            initial_equity=10000.0,
            final_equity=7100.0,
            equity_curve=equity_curve,
            max_drawdown=0.29,
        )
        result = calculate_risk_metrics(broker)
        # Should handle negative returns properly
        assert float(result.total_return_pct) < 0
        # Sortino may still be calculated with downside deviation
        assert result.downside_volatility >= Decimal("0")
