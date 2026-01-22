"""Tests for PositionSizer state and sizing result helpers."""

from decimal import Decimal

import pytest

from gpt_trader.features.intelligence.sizing.position_sizer import (
    PositionSizer,
    PositionSizingConfig,
    SizingResult,
)


class TestPositionSizerState:
    def test_record_trade_result(self):
        """Test recording trade results."""
        sizer = PositionSizer(config=PositionSizingConfig(enable_kelly_sizing=True))

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

        sizer.add_position_risk("BTC-USD", 0.025)

        result = sizer.calculate_size(
            symbol="ETH-USD",
            current_price=Decimal("3000"),
            equity=Decimal("10000"),
            existing_positions=sizer._portfolio_positions,
        )

        assert result.position_fraction <= 0.005 + 0.001

    def test_serialize_deserialize_state(self):
        """Test state serialization."""
        sizer = PositionSizer()

        sizer.record_trade_result("TEST", is_win=True)
        sizer.add_position_risk("TEST", 0.02)

        state = sizer.serialize_state()

        new_sizer = PositionSizer()
        new_sizer.deserialize_state(state)

        assert new_sizer._win_rates["TEST"] == (1, 0)
        assert new_sizer._portfolio_positions["TEST"] == 0.02


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
