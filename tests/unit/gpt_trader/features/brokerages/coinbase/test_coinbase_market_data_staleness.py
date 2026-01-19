"""Coinbase market data staleness tests."""

from __future__ import annotations

import pytest

from gpt_trader.features.brokerages.coinbase.models import APIConfig
from tests.unit.gpt_trader.features.brokerages.coinbase.minimal_brokerage import (
    MinimalCoinbaseBrokerage,
)

pytestmark = pytest.mark.endpoints


class TestCoinbaseMarketDataStaleness:
    def test_staleness_detection_fresh_vs_stale_toggles(self) -> None:
        config = APIConfig(
            api_key="test",
            api_secret="test",
            passphrase="test",
            base_url="https://api.sandbox.coinbase.com",
            sandbox=True,
            enable_derivatives=True,
            api_mode="advanced",
            auth_type="JWT",
        )
        adapter = MinimalCoinbaseBrokerage(config)
        symbol = "BTC-PERP"
        assert adapter.is_stale(symbol, threshold_seconds=10) is True
        assert adapter.is_stale(symbol, threshold_seconds=1) is True
        adapter.start_market_data([symbol])
        assert adapter.is_stale(symbol, threshold_seconds=10) is False
        assert adapter.is_stale(symbol, threshold_seconds=1) is False

    def test_staleness_behavior_matches_validator(self) -> None:
        config = APIConfig(
            api_key="test",
            api_secret="test",
            passphrase="test",
            base_url="https://api.sandbox.coinbase.com",
            sandbox=True,
            enable_derivatives=True,
            api_mode="advanced",
            auth_type="JWT",
        )
        adapter = MinimalCoinbaseBrokerage(config)
        symbol = "ETH-PERP"
        assert adapter.is_stale(symbol) is True
        adapter.start_market_data([symbol])
        assert adapter.is_stale(symbol) is False
        assert adapter.is_stale(symbol, threshold_seconds=1) is False
