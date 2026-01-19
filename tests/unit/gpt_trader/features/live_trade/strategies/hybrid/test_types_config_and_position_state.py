"""Tests for hybrid strategy configuration and position state models."""

from decimal import Decimal

from gpt_trader.features.live_trade.strategies.hybrid.types import (
    HybridPositionState,
    HybridStrategyConfig,
)


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
            cfm_side="long",
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
