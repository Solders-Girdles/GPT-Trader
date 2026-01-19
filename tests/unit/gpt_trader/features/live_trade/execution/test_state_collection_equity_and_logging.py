"""Tests for `StateCollector` equity calculation and collateral logging."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.core import Balance
from gpt_trader.features.live_trade.execution.state_collection import StateCollector


class TestCalculateEquityFromBalances:
    """Tests for calculate_equity_from_balances method."""

    def test_sums_collateral_balances(self, collector: StateCollector) -> None:
        """Test that collateral balances are summed correctly."""
        balances = [
            Balance(asset="USD", total=Decimal("100"), available=Decimal("50")),
            Balance(asset="USDC", total=Decimal("200"), available=Decimal("100")),
            Balance(asset="BTC", total=Decimal("1"), available=Decimal("1")),
        ]

        available, collateral, total = collector.calculate_equity_from_balances(balances)

        assert available == Decimal("150")
        assert total == Decimal("300")
        assert len(collateral) == 2

    def test_falls_back_to_usd_balance(self, collector: StateCollector) -> None:
        """Test fallback to USD balance when no collateral matches."""
        collector.collateral_assets = {"NONEXISTENT"}
        balances = [
            Balance(asset="USD", total=Decimal("1000"), available=Decimal("800")),
            Balance(asset="BTC", total=Decimal("1"), available=Decimal("1")),
        ]

        available, collateral, total = collector.calculate_equity_from_balances(balances)

        assert available == Decimal("800")
        assert total == Decimal("1000")
        assert len(collateral) == 1
        assert collateral[0].asset == "USD"

    def test_returns_zeros_when_no_balances_match(self, collector: StateCollector) -> None:
        """Test returns zeros when no balances match."""
        collector.collateral_assets = {"NONEXISTENT"}
        balances = [
            Balance(asset="BTC", total=Decimal("1"), available=Decimal("1")),
        ]

        available, collateral, total = collector.calculate_equity_from_balances(balances)

        assert available == Decimal("0")
        assert total == Decimal("0")
        assert len(collateral) == 0

    def test_handles_empty_balances(self, collector: StateCollector) -> None:
        """Test handling of empty balance list."""
        available, collateral, total = collector.calculate_equity_from_balances([])

        assert available == Decimal("0")
        assert total == Decimal("0")
        assert len(collateral) == 0


class TestLogCollateralUpdate:
    """Tests for log_collateral_update method."""

    def test_skips_empty_collateral_list(self, collector: StateCollector) -> None:
        """Test that empty collateral list is skipped."""
        collector._production_logger = MagicMock()

        collector.log_collateral_update([], Decimal("100"), Decimal("100"), [])

        collector._production_logger.log_balance_update.assert_not_called()

    def test_sets_initial_collateral_value(self, collector: StateCollector) -> None:
        """Test that initial collateral value is set."""
        collector._production_logger = MagicMock()
        collateral = [Balance(asset="USD", total=Decimal("100"), available=Decimal("50"))]

        collector.log_collateral_update(collateral, Decimal("100"), Decimal("100"), collateral)

        assert collector._last_collateral_available == Decimal("50")

    def test_logs_significant_change(self, collector: StateCollector) -> None:
        """Test that significant changes are logged."""
        collector._production_logger = MagicMock()
        collector._last_collateral_available = Decimal("50")
        collateral = [Balance(asset="USD", total=Decimal("110"), available=Decimal("60"))]

        collector.log_collateral_update(collateral, Decimal("100"), Decimal("110"), collateral)

        collector._production_logger.log_balance_update.assert_called_once()

    def test_handles_logger_exception(self, collector: StateCollector) -> None:
        """Test that logger exceptions are suppressed."""
        collector._production_logger = MagicMock()
        collector._production_logger.log_balance_update.side_effect = RuntimeError("Log error")
        collateral = [Balance(asset="USD", total=Decimal("100"), available=Decimal("50"))]

        # Should not raise
        collector.log_collateral_update(collateral, Decimal("100"), Decimal("100"), collateral)
        collector._production_logger.log_balance_update.assert_called_once()
