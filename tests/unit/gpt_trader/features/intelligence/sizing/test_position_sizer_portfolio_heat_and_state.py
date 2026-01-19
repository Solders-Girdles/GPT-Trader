"""Tests for PositionSizer stateful helpers (heat tracking, win rates)."""

from decimal import Decimal

import pytest

from gpt_trader.features.intelligence.sizing.position_sizer import (
    PositionSizer,
    PositionSizingConfig,
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
