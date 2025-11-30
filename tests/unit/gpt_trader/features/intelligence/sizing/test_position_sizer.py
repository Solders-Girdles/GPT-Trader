"""Tests for PositionSizer module."""

from decimal import Decimal

import pytest

from gpt_trader.features.intelligence.regime.detector import MarketRegimeDetector
from gpt_trader.features.intelligence.regime.models import RegimeConfig
from gpt_trader.features.intelligence.sizing.position_sizer import (
    PositionSizer,
    PositionSizingConfig,
    SizingResult,
)


class TestPositionSizingConfig:
    """Tests for PositionSizingConfig."""

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

        # Crisis should have low factor
        assert config.regime_scale_factors["CRISIS"] == 0.2
        # Bull quiet should be favorable
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


class TestPositionSizer:
    """Tests for PositionSizer."""

    @pytest.fixture
    def regime_detector(self):
        """Create regime detector with some price history."""
        detector = MarketRegimeDetector(RegimeConfig())

        # Feed some price data to warm up
        base_price = Decimal("50000")
        for i in range(100):
            # Small oscillations around base price
            delta = Decimal(str(i % 10 - 5)) * 10
            detector.update("BTC-USD", base_price + delta)

        return detector

    @pytest.fixture
    def sizer(self, regime_detector):
        """Create position sizer with detector."""
        return PositionSizer(regime_detector=regime_detector)

    def test_basic_sizing(self, sizer):
        """Test basic position size calculation."""
        result = sizer.calculate_size(
            symbol="BTC-USD",
            current_price=Decimal("50000"),
            equity=Decimal("10000"),
        )

        assert isinstance(result, SizingResult)
        assert result.position_fraction > 0
        assert result.position_value > 0
        assert result.position_quantity > 0

    def test_position_fraction_limits(self, sizer):
        """Test position fraction respects limits."""
        result = sizer.calculate_size(
            symbol="BTC-USD",
            current_price=Decimal("50000"),
            equity=Decimal("10000"),
            decision_confidence=1.0,
        )

        config = sizer.config
        assert result.position_fraction >= config.min_position_fraction
        assert result.position_fraction <= config.max_position_fraction

    def test_confidence_affects_size(self):
        """Test that higher confidence increases size."""
        # Use higher base fraction and disable min fraction to see confidence effect
        config = PositionSizingConfig(
            enable_confidence_scaling=True,
            base_position_fraction=0.10,  # Higher base
            min_position_fraction=0.0,  # No floor
            regime_scale_factors={"UNKNOWN": 1.0},  # Don't penalize unknown
        )
        sizer = PositionSizer(config=config)

        result_low = sizer.calculate_size(
            symbol="TEST",
            current_price=Decimal("100"),
            equity=Decimal("10000"),
            decision_confidence=0.3,
        )

        result_high = sizer.calculate_size(
            symbol="TEST",
            current_price=Decimal("100"),
            equity=Decimal("10000"),
            decision_confidence=0.9,
        )

        # Confidence factor should make high confidence result larger
        assert result_high.confidence_factor > result_low.confidence_factor
        assert result_high.position_fraction > result_low.position_fraction

    def test_regime_affects_size(self, regime_detector):
        """Test that regime affects position size."""
        sizer = PositionSizer(
            regime_detector=regime_detector,
            config=PositionSizingConfig(enable_volatility_scaling=False),
        )

        # Get result for current regime
        result = sizer.calculate_size(
            symbol="BTC-USD",
            current_price=Decimal("50000"),
            equity=Decimal("10000"),
        )

        # Regime factor should be applied
        assert result.regime_factor > 0

    def test_sizing_without_detector(self):
        """Test sizing works without regime detector."""
        sizer = PositionSizer(regime_detector=None)

        result = sizer.calculate_size(
            symbol="TEST",
            current_price=Decimal("100"),
            equity=Decimal("10000"),
        )

        # Should still produce valid result
        assert result.position_fraction > 0
        assert result.regime == "UNKNOWN"

    def test_position_value_calculation(self):
        """Test position value is calculated correctly."""
        sizer = PositionSizer(
            config=PositionSizingConfig(
                base_position_fraction=0.05,
                enable_volatility_scaling=False,
                enable_confidence_scaling=False,
            )
        )

        result = sizer.calculate_size(
            symbol="TEST",
            current_price=Decimal("100"),
            equity=Decimal("10000"),
        )

        # With 5% base and unknown regime (0.5 factor * ~0.75 confidence adjustment)
        # Position value should be proportional to equity * fraction
        expected_value = Decimal("10000") * Decimal(str(result.position_fraction))
        assert result.position_value == pytest.approx(expected_value, rel=0.01)

    def test_position_quantity_calculation(self):
        """Test position quantity is calculated correctly."""
        sizer = PositionSizer()

        result = sizer.calculate_size(
            symbol="TEST",
            current_price=Decimal("100"),
            equity=Decimal("10000"),
        )

        # Quantity should be value / price
        expected_quantity = result.position_value / Decimal("100")
        assert result.position_quantity == pytest.approx(expected_quantity, rel=0.001)

    def test_record_trade_result(self):
        """Test recording trade results."""
        sizer = PositionSizer(config=PositionSizingConfig(enable_kelly_sizing=True))

        # Record some wins
        sizer.record_trade_result("TEST", is_win=True)
        sizer.record_trade_result("TEST", is_win=True)
        sizer.record_trade_result("TEST", is_win=False)

        wins, losses = sizer._win_rates.get("TEST", (0, 0))
        assert wins == 2
        assert losses == 1

    def test_portfolio_heat_tracking(self):
        """Test portfolio heat is tracked."""
        sizer = PositionSizer()

        sizer.add_position_risk("BTC-USD", 0.02)
        sizer.add_position_risk("ETH-USD", 0.01)

        heat = sizer.get_portfolio_heat()

        assert heat["total_heat"] == pytest.approx(0.03)
        assert "BTC-USD" in heat["positions"]
        assert "ETH-USD" in heat["positions"]

    def test_portfolio_heat_limits_sizing(self):
        """Test portfolio heat limits position sizing."""
        config = PositionSizingConfig(max_portfolio_heat=0.03)
        sizer = PositionSizer(config=config)

        # Add existing position risk
        sizer.add_position_risk("BTC-USD", 0.025)

        # New position should be limited
        result = sizer.calculate_size(
            symbol="ETH-USD",
            current_price=Decimal("3000"),
            equity=Decimal("10000"),
            existing_positions=sizer._portfolio_positions,
        )

        # Should be limited by remaining heat
        assert result.position_fraction <= 0.005 + 0.001  # Small tolerance

    def test_sizing_result_to_dict(self):
        """Test SizingResult serialization."""
        sizer = PositionSizer()

        result = sizer.calculate_size(
            symbol="TEST",
            current_price=Decimal("100"),
            equity=Decimal("10000"),
        )

        data = result.to_dict()

        assert "position_fraction" in data
        assert "factors" in data
        assert "risk" in data
        assert "context" in data
        assert "reasoning" in data

    def test_serialize_deserialize_state(self):
        """Test state serialization."""
        sizer = PositionSizer()

        sizer.record_trade_result("TEST", is_win=True)
        sizer.add_position_risk("TEST", 0.02)

        # Serialize
        state = sizer.serialize_state()

        # Create new sizer and restore
        new_sizer = PositionSizer()
        new_sizer.deserialize_state(state)

        assert new_sizer._win_rates["TEST"] == (1, 0)
        assert new_sizer._portfolio_positions["TEST"] == 0.02


class TestSizingResult:
    """Tests for SizingResult."""

    def test_to_dict_format(self):
        """Test to_dict produces expected format."""
        result = SizingResult(
            position_fraction=0.025,
            position_value=Decimal("250"),
            position_quantity=Decimal("0.005"),
            base_size=0.02,
            regime_factor=1.2,
            volatility_factor=1.0,
            confidence_factor=0.8,
            kelly_factor=1.0,
            estimated_risk=0.001,
            risk_reward_ratio=2.0,
            regime="BULL_QUIET",
            regime_confidence=0.85,
            atr_value=500.0,
            reasoning="Test reasoning",
        )

        data = result.to_dict()

        assert data["position_fraction"] == 0.025
        assert data["position_value"] == "250"
        assert data["factors"]["regime_factor"] == 1.2
        assert data["context"]["regime"] == "BULL_QUIET"
        assert data["reasoning"] == "Test reasoning"
