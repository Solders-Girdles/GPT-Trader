"""Tests for VWAPSignalConfig."""

from gpt_trader.features.live_trade.signals.vwap import VWAPSignalConfig


class TestVWAPSignalConfig:
    """Tests for VWAPSignalConfig."""

    def test_default_values(self) -> None:
        """Test default config values."""
        config = VWAPSignalConfig()
        assert config.deviation_threshold == 0.01
        assert config.strong_deviation_threshold == 0.025
        assert config.min_trades == 20

    def test_custom_values(self) -> None:
        """Test custom config values."""
        config = VWAPSignalConfig(
            deviation_threshold=0.005,
            strong_deviation_threshold=0.02,
            min_trades=50,
        )
        assert config.deviation_threshold == 0.005
        assert config.min_trades == 50
