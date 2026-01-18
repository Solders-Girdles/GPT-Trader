"""Tests for `StateCollector.collect_account_state`."""

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
