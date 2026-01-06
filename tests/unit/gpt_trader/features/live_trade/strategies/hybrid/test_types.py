"""Tests for hybrid strategy types."""

from decimal import Decimal

from gpt_trader.features.live_trade.strategies.hybrid.types import (
    Action,
    HybridDecision,
    HybridMarketData,
    HybridPositionState,
    HybridStrategyConfig,
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


class TestHybridMarketData:
    """Tests for HybridMarketData dataclass."""

    def test_spot_only_data(self):
        """Can create with spot data only."""
        data = HybridMarketData(
            symbol="BTC-USD",
            spot_price=Decimal("50000"),
        )
        assert data.symbol == "BTC-USD"
        assert data.spot_price == Decimal("50000")
        assert data.futures_price is None
        assert data.basis is None
        assert data.basis_percentage is None

    def test_full_market_data(self):
        """Can create with all market data."""
        data = HybridMarketData(
            symbol="BTC",
            spot_price=Decimal("50000"),
            futures_price=Decimal("50500"),
            spot_bid=Decimal("49990"),
            spot_ask=Decimal("50010"),
            futures_bid=Decimal("50490"),
            futures_ask=Decimal("50510"),
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("50250"),
        )
        assert data.futures_price == Decimal("50500")
        assert data.spot_bid == Decimal("49990")
        assert data.funding_rate == Decimal("0.0001")

    def test_basis_calculation(self):
        """Basis calculated correctly."""
        data = HybridMarketData(
            symbol="BTC",
            spot_price=Decimal("50000"),
            futures_price=Decimal("50500"),
        )
        assert data.basis == Decimal("500")

    def test_basis_percentage_premium(self):
        """Basis percentage calculated for premium."""
        data = HybridMarketData(
            symbol="BTC",
            spot_price=Decimal("50000"),
            futures_price=Decimal("50500"),
        )
        assert data.basis_percentage == Decimal("1")  # 500/50000 * 100 = 1%

    def test_basis_percentage_discount(self):
        """Basis percentage calculated for discount."""
        data = HybridMarketData(
            symbol="BTC",
            spot_price=Decimal("50000"),
            futures_price=Decimal("49750"),
        )
        assert data.basis_percentage == Decimal("-0.5")  # -250/50000 * 100 = -0.5%

    def test_basis_percentage_no_futures(self):
        """Basis percentage is None without futures."""
        data = HybridMarketData(
            symbol="BTC",
            spot_price=Decimal("50000"),
        )
        assert data.basis_percentage is None

    def test_basis_percentage_zero_spot(self):
        """Basis percentage is None with zero spot price."""
        data = HybridMarketData(
            symbol="BTC",
            spot_price=Decimal("0"),
            futures_price=Decimal("50000"),
        )
        assert data.basis_percentage is None


class TestHybridStrategyConfig:
    """Tests for HybridStrategyConfig dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        config = HybridStrategyConfig()
        assert config.enabled is True
        assert config.base_symbol == "BTC"
        assert config.quote_currency == "USD"
        assert config.enable_spot is True
        assert config.enable_cfm is True
        assert config.cfm_max_leverage == 5
        assert config.cfm_default_leverage == 1
        assert config.basis_entry_threshold_pct == 0.5
        assert config.basis_exit_threshold_pct == 0.1

    def test_auto_generate_spot_symbol(self):
        """Spot symbol is auto-generated."""
        config = HybridStrategyConfig(base_symbol="ETH", quote_currency="USD")
        assert config.spot_symbol == "ETH-USD"

    def test_custom_spot_symbol(self):
        """Custom spot symbol is preserved."""
        config = HybridStrategyConfig(spot_symbol="CUSTOM-USD")
        assert config.spot_symbol == "CUSTOM-USD"

    def test_full_config(self):
        """Can set all configuration values."""
        config = HybridStrategyConfig(
            enabled=True,
            base_symbol="SOL",
            quote_currency="USDT",
            enable_spot=True,
            spot_position_size_pct=0.30,
            enable_cfm=True,
            cfm_position_size_pct=0.20,
            cfm_max_leverage=10,
            cfm_default_leverage=3,
            cfm_symbol="SLP-20DEC30-CDE",
            basis_entry_threshold_pct=0.8,
            basis_exit_threshold_pct=0.2,
            max_total_exposure_pct=0.9,
            stop_loss_pct=0.03,
            take_profit_pct=0.15,
        )
        assert config.base_symbol == "SOL"
        assert config.cfm_symbol == "SLP-20DEC30-CDE"
        assert config.cfm_max_leverage == 10
        assert config.cfm_default_leverage == 3
        assert config.spot_position_size_pct == 0.30


class TestHybridPositionState:
    """Tests for HybridPositionState dataclass."""

    def test_default_state(self):
        """Default is flat with no positions."""
        state = HybridPositionState()
        assert state.spot_quantity == Decimal("0")
        assert state.spot_entry_price is None
        assert state.spot_side == "flat"
        assert state.cfm_quantity == Decimal("0")
        assert state.cfm_entry_price is None
        assert state.cfm_side == "flat"
        assert state.cfm_leverage == 1

    def test_has_spot_position(self):
        """Detects spot position."""
        state = HybridPositionState(spot_quantity=Decimal("1"))
        assert state.has_spot_position is True
        assert state.has_cfm_position is False

    def test_has_cfm_position(self):
        """Detects CFM position."""
        state = HybridPositionState(cfm_quantity=Decimal("0.5"))
        assert state.has_spot_position is False
        assert state.has_cfm_position is True

    def test_is_basis_position(self):
        """Detects basis trade (long spot, short futures)."""
        state = HybridPositionState(
            spot_quantity=Decimal("1"),
            spot_side="long",
            cfm_quantity=Decimal("1"),
            cfm_side="short",
        )
        assert state.is_basis_position is True

    def test_is_not_basis_position_wrong_sides(self):
        """Not basis if sides don't match long/short pattern."""
        state = HybridPositionState(
            spot_quantity=Decimal("1"),
            spot_side="long",
            cfm_quantity=Decimal("1"),
            cfm_side="long",  # Same side - not basis
        )
        assert state.is_basis_position is False

    def test_is_not_basis_position_missing_position(self):
        """Not basis if only one position."""
        state = HybridPositionState(
            spot_quantity=Decimal("1"),
            spot_side="long",
        )
        assert state.is_basis_position is False

    def test_to_dict(self):
        """Position state serializes to dict."""
        state = HybridPositionState(
            spot_quantity=Decimal("1"),
            spot_entry_price=Decimal("50000"),
            spot_side="long",
            cfm_quantity=Decimal("0.5"),
            cfm_entry_price=Decimal("50500"),
            cfm_side="short",
            cfm_leverage=3,
        )
        data = state.to_dict()
        assert data["spot_quantity"] == "1"
        assert data["spot_entry_price"] == "50000"
        assert data["spot_side"] == "long"
        assert data["cfm_quantity"] == "0.5"
        assert data["cfm_entry_price"] == "50500"
        assert data["cfm_side"] == "short"
        assert data["cfm_leverage"] == 3
