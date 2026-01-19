"""Tests for PositionSizingConfig."""

from gpt_trader.features.intelligence.sizing.position_sizer import PositionSizingConfig


class TestPositionSizingConfig:
    def test_default_config(self):
        """Test default configuration values."""
        config = PositionSizingConfig()

        assert config.base_position_fraction == 0.02
        assert config.max_position_fraction == 0.10
        assert config.min_position_fraction == 0.005
        assert config.enable_volatility_scaling is True
        assert config.enable_kelly_sizing is False

    def test_regime_scale_factors(self):
        """Test regime scale factors are set."""
        config = PositionSizingConfig()

        assert config.regime_scale_factors["CRISIS"] == 0.2
        assert config.regime_scale_factors["BULL_QUIET"] == 1.2

    def test_to_dict(self):
        """Test serialization to dict."""
        config = PositionSizingConfig()
        data = config.to_dict()

        assert "base_position_fraction" in data
        assert "regime_scale_factors" in data
        assert data["base_position_fraction"] == 0.02

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "base_position_fraction": 0.05,
            "max_position_fraction": 0.15,
            "enable_kelly_sizing": True,
        }

        config = PositionSizingConfig.from_dict(data)

        assert config.base_position_fraction == 0.05
        assert config.max_position_fraction == 0.15
        assert config.enable_kelly_sizing is True
