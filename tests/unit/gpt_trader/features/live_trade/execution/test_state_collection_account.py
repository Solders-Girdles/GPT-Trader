"""Tests for `StateCollector` account state collection, equity calculation, and logging."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import Balance
from gpt_trader.features.live_trade.execution.state_collection import StateCollector


class TestCollectAccountState:
    """Tests for collect_account_state method."""

    def test_collects_balances_and_positions(
        self, collector: StateCollector, mock_broker: MagicMock
    ) -> None:
        """Test collecting balances and positions."""
        mock_broker.list_balances.return_value = [
            Balance(asset="USD", total=Decimal("1000"), available=Decimal("800"))
        ]
        mock_broker.list_positions.return_value = []

        balances, equity, collateral, total, positions = collector.collect_account_state()

        assert len(balances) == 1
        assert equity == Decimal("800")
        assert len(collateral) == 1
        assert total == Decimal("1000")
        assert positions == []

    def test_handles_broker_without_list_balances(
        self, mock_broker: MagicMock, mock_config
    ) -> None:
        """Test handling broker without list_balances method."""
        del mock_broker.list_balances
        mock_broker.list_positions.return_value = []

        collector = StateCollector(mock_broker, mock_config)
        balances, equity, collateral, total, positions = collector.collect_account_state()

        assert balances == []
        assert equity == Decimal("0")
        assert collateral == []
        assert total == Decimal("0")
        assert positions == []

    def test_handles_balance_exception_in_integration_mode(
        self, mock_broker: MagicMock, mock_config
    ) -> None:
        """Test that balance exceptions are suppressed in integration mode."""
        mock_broker.list_balances.side_effect = RuntimeError("API error")
        mock_broker.list_positions.return_value = []

        collector = StateCollector(mock_broker, mock_config, integration_mode=True)
        balances, equity, _, _, _ = collector.collect_account_state()

        # Should use default balance in integration mode
        assert len(balances) == 1
        assert balances[0].asset == "USD"

    def test_raises_balance_exception_in_normal_mode(
        self, collector: StateCollector, mock_broker: MagicMock
    ) -> None:
        """Test that balance exceptions are raised in normal mode."""
        mock_broker.list_balances.side_effect = RuntimeError("API error")

        with pytest.raises(RuntimeError, match="API error"):
            collector.collect_account_state()

    def test_handles_position_exception_in_integration_mode(
        self, mock_broker: MagicMock, mock_config
    ) -> None:
        """Test that position exceptions are suppressed in integration mode."""
        mock_broker.list_balances.return_value = [
            Balance(asset="USD", total=Decimal("1000"), available=Decimal("800"))
        ]
        mock_broker.list_positions.side_effect = RuntimeError("API error")

        collector = StateCollector(mock_broker, mock_config, integration_mode=True)
        _, _, _, _, positions = collector.collect_account_state()

        assert positions == []

    def test_provides_default_balance_in_integration_mode(
        self, mock_broker: MagicMock, mock_config
    ) -> None:
        """Test that default balance is provided in integration mode."""
        mock_broker.list_balances.return_value = []
        mock_broker.list_positions.return_value = []

        collector = StateCollector(mock_broker, mock_config, integration_mode=True)
        balances, equity, _, _, _ = collector.collect_account_state()

        assert len(balances) == 1
        assert equity == Decimal("100000")


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
