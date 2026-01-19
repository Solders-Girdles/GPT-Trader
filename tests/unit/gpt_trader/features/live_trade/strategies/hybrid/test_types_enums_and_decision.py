"""Tests for hybrid strategy enums and decision model."""

from decimal import Decimal

from gpt_trader.features.live_trade.strategies.hybrid.types import (
    Action,
    HybridDecision,
    TradingMode,
)


class TestTradingMode:
    """Tests for TradingMode enum."""

    def test_trading_modes(self):
        """All trading modes are defined."""
        assert TradingMode.SPOT_ONLY.value == "spot_only"
        assert TradingMode.CFM_ONLY.value == "cfm_only"
        assert TradingMode.HYBRID.value == "hybrid"


class TestAction:
    """Tests for Action enum."""

    def test_actions(self):
        """All actions are defined."""
        assert Action.BUY.value == "buy"
        assert Action.SELL.value == "sell"
        assert Action.HOLD.value == "hold"
        assert Action.CLOSE.value == "close"
        assert Action.CLOSE_LONG.value == "close_long"
        assert Action.CLOSE_SHORT.value == "close_short"


class TestHybridDecision:
    """Tests for HybridDecision dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        decision = HybridDecision(
            action=Action.HOLD,
            symbol="BTC-USD",
            mode=TradingMode.SPOT_ONLY,
        )
        assert decision.quantity == Decimal("0")
        assert decision.leverage == 1
        assert decision.reason == ""
        assert decision.confidence == 0.0
        assert decision.indicators == {}

    def test_full_decision(self):
        """Can create decision with all fields."""
        decision = HybridDecision(
            action=Action.BUY,
            symbol="BTC-20DEC30-CDE",
            mode=TradingMode.CFM_ONLY,
            quantity=Decimal("0.5"),
            leverage=5,
            reason="Entry signal",
            confidence=0.8,
            indicators={"rsi": 30},
        )
        assert decision.action == Action.BUY
        assert decision.symbol == "BTC-20DEC30-CDE"
        assert decision.mode == TradingMode.CFM_ONLY
        assert decision.quantity == Decimal("0.5")
        assert decision.leverage == 5
        assert decision.reason == "Entry signal"
        assert decision.confidence == 0.8
        assert decision.indicators == {"rsi": 30}

    def test_is_actionable_buy(self):
        """BUY is actionable."""
        decision = HybridDecision(action=Action.BUY, symbol="BTC-USD", mode=TradingMode.SPOT_ONLY)
        assert decision.is_actionable() is True

    def test_is_actionable_sell(self):
        """SELL is actionable."""
        decision = HybridDecision(action=Action.SELL, symbol="BTC-USD", mode=TradingMode.SPOT_ONLY)
        assert decision.is_actionable() is True

    def test_is_actionable_hold(self):
        """HOLD is not actionable."""
        decision = HybridDecision(action=Action.HOLD, symbol="BTC-USD", mode=TradingMode.SPOT_ONLY)
        assert decision.is_actionable() is False

    def test_is_actionable_close(self):
        """CLOSE is actionable."""
        decision = HybridDecision(action=Action.CLOSE, symbol="BTC-USD", mode=TradingMode.SPOT_ONLY)
        assert decision.is_actionable() is True

    def test_to_dict(self):
        """Decision serializes to dict."""
        decision = HybridDecision(
            action=Action.BUY,
            symbol="BTC-USD",
            mode=TradingMode.SPOT_ONLY,
            quantity=Decimal("1.5"),
            leverage=3,
            reason="test",
            confidence=0.9,
            indicators={"ma": 50000},
        )
        data = decision.to_dict()
        assert data["action"] == "buy"
        assert data["symbol"] == "BTC-USD"
        assert data["mode"] == "spot_only"
        assert data["quantity"] == "1.5"
        assert data["leverage"] == 3
        assert data["reason"] == "test"
        assert data["confidence"] == 0.9
        assert data["indicators"] == {"ma": 50000}
