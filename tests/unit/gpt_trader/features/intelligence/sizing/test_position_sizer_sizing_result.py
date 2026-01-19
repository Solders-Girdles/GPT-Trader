"""Tests for SizingResult."""

from decimal import Decimal

from gpt_trader.features.intelligence.sizing.position_sizer import SizingResult


class TestSizingResult:
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
