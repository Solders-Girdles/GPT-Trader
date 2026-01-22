"""Tests for HybridPaperBroker initialization and status methods."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.brokerages.paper.hybrid as hybrid_module
from gpt_trader.core import Balance, Position
from gpt_trader.features.brokerages.paper.hybrid import HybridPaperBroker


class TestHybridPaperBrokerInit:
    """Test HybridPaperBroker initialization."""

    @pytest.fixture
    def auth_mock(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        mock_auth = MagicMock()
        monkeypatch.setattr(hybrid_module, "SimpleAuth", mock_auth)
        return mock_auth

    @pytest.fixture
    def client_mock(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        mock_client = MagicMock()
        monkeypatch.setattr(hybrid_module, "CoinbaseClient", mock_client)
        return mock_client

    def test_init_creates_client(self, auth_mock: MagicMock, client_mock: MagicMock) -> None:
        """Test initialization creates Coinbase client."""
        broker = HybridPaperBroker(
            api_key="test_key",
            private_key="test_private_key",
        )

        auth_mock.assert_called_once_with(key_name="test_key", private_key="test_private_key")
        client_mock.assert_called_once()
        assert broker._initial_equity == Decimal("10000")
        assert broker._slippage_bps == 5
        assert broker._commission_bps == Decimal("5")

    def test_init_with_custom_parameters(
        self, auth_mock: MagicMock, client_mock: MagicMock
    ) -> None:
        """Test initialization with custom parameters."""
        broker = HybridPaperBroker(
            api_key="test_key",
            private_key="test_private_key",
            initial_equity=Decimal("50000"),
            slippage_bps=10,
            commission_bps=Decimal("10"),
        )

        assert broker._initial_equity == Decimal("50000")
        assert broker._slippage_bps == 10
        assert broker._commission_bps == Decimal("10")

    def test_init_creates_usd_balance(self, auth_mock: MagicMock, client_mock: MagicMock) -> None:
        """Test initialization creates USD balance."""
        broker = HybridPaperBroker(
            api_key="test_key",
            private_key="test_private_key",
            initial_equity=Decimal("25000"),
        )

        assert "USD" in broker._balances
        assert broker._balances["USD"].total == Decimal("25000")
        assert broker._balances["USD"].available == Decimal("25000")


class TestHybridPaperBrokerStatus:
    """Test HybridPaperBroker status methods."""

    @pytest.fixture
    def broker(self, broker_factory) -> HybridPaperBroker:
        """Create broker fixture."""
        return broker_factory(initial_equity=Decimal("10000"))

    def test_is_connected_always_true(self, broker: HybridPaperBroker) -> None:
        """Test is_connected returns True."""
        assert broker.is_connected() is True

    def test_is_stale_always_false(self, broker: HybridPaperBroker) -> None:
        """Test is_stale returns False."""
        assert broker.is_stale("BTC-USD") is False

    def test_get_status_returns_status(self, broker: HybridPaperBroker) -> None:
        """Test get_status returns status dict."""
        result = broker.get_status()

        assert result["mode"] == "paper"
        assert result["initial_equity"] == 10000.0
        assert result["current_equity"] == 10000.0
        assert result["positions"] == 0
        assert result["orders_executed"] == 0


class TestHybridPaperBrokerPositionsBalances:
    """Test HybridPaperBroker position and balance methods."""

    @pytest.fixture
    def broker(self, broker_factory) -> HybridPaperBroker:
        """Create broker fixture with mocked client."""
        return broker_factory(initial_equity=Decimal("10000"))

    def test_list_positions_empty(self, broker: HybridPaperBroker) -> None:
        """Test list_positions returns empty list initially."""
        result = broker.list_positions()

        assert result == []

    def test_list_positions_returns_positions(self, broker: HybridPaperBroker) -> None:
        """Test list_positions returns stored positions."""
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("51000"),
            unrealized_pnl=Decimal("500"),
            realized_pnl=Decimal("0"),
            side="long",
            leverage=1,
        )

        result = broker.list_positions()

        assert len(result) == 1
        assert result[0].symbol == "BTC-USD"

    def test_get_positions_alias(self, broker: HybridPaperBroker) -> None:
        """Test get_positions is alias for list_positions."""
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("51000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
            leverage=1,
        )

        result = broker.get_positions()

        assert len(result) == 1

    def test_list_balances_returns_balances(self, broker: HybridPaperBroker) -> None:
        """Test list_balances returns balances."""
        result = broker.list_balances()

        assert len(result) == 1
        assert result[0].asset == "USD"
        assert result[0].total == Decimal("10000")

    def test_get_balances_alias(self, broker: HybridPaperBroker) -> None:
        """Test get_balances is alias for list_balances."""
        result = broker.get_balances()

        assert len(result) == 1

    def test_get_equity_cash_only(self, broker: HybridPaperBroker) -> None:
        """Test get_equity with cash only."""
        result = broker.get_equity()

        assert result == Decimal("10000")

    def test_get_equity_with_long_position(self, broker: HybridPaperBroker) -> None:
        """Test get_equity includes unrealized PnL from long position."""
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
            leverage=1,
        )
        broker._last_prices["BTC-USD"] = Decimal("51000")  # Price went up
        broker._balances["USD"] = Balance(
            asset="USD", total=Decimal("5000"), available=Decimal("5000")
        )

        result = broker.get_equity()

        # 5000 cash + (51000 - 50000) * 0.1 = 5000 + 100 = 5100
        assert result == Decimal("5100")

    def test_get_equity_with_short_position(self, broker: HybridPaperBroker) -> None:
        """Test get_equity includes unrealized PnL from short position."""
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("-0.1"),  # Short
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="short",
            leverage=1,
        )
        broker._last_prices["BTC-USD"] = Decimal("49000")  # Price went down (profit for short)
        broker._balances["USD"] = Balance(
            asset="USD", total=Decimal("15000"), available=Decimal("15000")
        )

        result = broker.get_equity()

        # 15000 cash + (50000 - 49000) * 0.1 = 15000 + 100 = 15100
        assert result == Decimal("15100")
