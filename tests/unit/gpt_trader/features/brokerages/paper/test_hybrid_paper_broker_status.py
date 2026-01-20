"""Tests for `HybridPaperBroker` status methods."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

import gpt_trader.features.brokerages.paper.hybrid as hybrid_module
from gpt_trader.features.brokerages.paper.hybrid import HybridPaperBroker


class TestHybridPaperBrokerStatus:
    """Test HybridPaperBroker status methods."""

    @pytest.fixture
    def broker(self, monkeypatch: pytest.MonkeyPatch):
        """Create broker fixture."""
        monkeypatch.setattr(hybrid_module, "CoinbaseClient", Mock())
        monkeypatch.setattr(hybrid_module, "SimpleAuth", Mock())
        return HybridPaperBroker(
            api_key="test_key",
            private_key="test_private_key",
            initial_equity=Decimal("10000"),
        )

    def test_is_connected_always_true(self, broker) -> None:
        """Test is_connected returns True."""
        assert broker.is_connected() is True

    def test_is_stale_always_false(self, broker) -> None:
        """Test is_stale returns False."""
        assert broker.is_stale("BTC-USD") is False

    def test_start_market_data_prefetches_quotes(self, broker) -> None:
        """Test start_market_data prefetches quotes."""
        broker._client = Mock()
        broker._client.get_market_product_ticker.return_value = {
            "best_bid": "50000",
            "best_ask": "50100",
            "trades": [{"price": "50050"}],
        }

        broker.start_market_data(["BTC-USD", "ETH-USD"])

        assert broker._client.get_market_product_ticker.call_count == 2

    def test_stop_market_data_noop(self, broker) -> None:
        """Test stop_market_data is no-op."""
        broker._last_prices["BTC-USD"] = Decimal("50000")

        result = broker.stop_market_data()

        assert result is None
        assert broker._last_prices["BTC-USD"] == Decimal("50000")

    def test_get_status_returns_status(self, broker) -> None:
        """Test get_status returns status dict."""
        result = broker.get_status()

        assert result["mode"] == "paper"
        assert result["initial_equity"] == 10000.0
        assert result["current_equity"] == 10000.0
        assert result["positions"] == 0
        assert result["orders_executed"] == 0
