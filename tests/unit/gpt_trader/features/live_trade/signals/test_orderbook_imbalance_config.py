"""Tests for OrderbookImbalanceSignalConfig."""

from gpt_trader.features.live_trade.signals.orderbook_imbalance import (
    OrderbookImbalanceSignalConfig,
)


class TestOrderbookImbalanceSignalConfig:
    """Tests for OrderbookImbalanceSignalConfig."""

    def test_default_values(self) -> None:
        """Test default config values."""
        config = OrderbookImbalanceSignalConfig()
        assert config.levels == 5
        assert config.imbalance_threshold == 0.2
        assert config.strong_imbalance_threshold == 0.5

    def test_custom_values(self) -> None:
        """Test custom config values."""
        config = OrderbookImbalanceSignalConfig(
            levels=10,
            imbalance_threshold=0.3,
            strong_imbalance_threshold=0.6,
        )
        assert config.levels == 10
        assert config.imbalance_threshold == 0.3
