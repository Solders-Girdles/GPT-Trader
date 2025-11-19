"""
Reduce-only mode tests for pre-trade checks.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.live_trade.risk.pre_trade_checks import (
    PreTradeValidator,
    ValidationError,
)


class TestPreTradeValidatorReduceOnly:
    """Reduce-only mode tests for PreTradeValidator."""

    # -------------------------------------------------------------------------
    # REDUCE ONLY MODE TESTS
    # -------------------------------------------------------------------------

    def test_reduce_only_mode_blocks_increasing_trades(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test reduce-only mode blocks position increases."""
        config = RiskConfig(enable_pre_trade_liq_projection=False)
        validator = PreTradeValidator(config, mock_event_store, is_reduce_only_mode=lambda: True)

        positions = {"BTC-USD": {"side": "long", "quantity": "1", "price": "50000"}}

        # Block increase to long position
        with pytest.raises(ValidationError, match="Reduce-only mode"):
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("0.5"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions=positions,
            )

        # Block increase to short position
        with pytest.raises(ValidationError, match="Reduce-only mode"):
            validator.pre_trade_validate(
                "BTC-USD",
                "sell",
                qty=Decimal("0.5"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={"BTC-USD": {"side": "short", "quantity": "1", "price": "50000"}},
            )

    def test_reduce_only_mode_allows_position_reduction(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test reduce-only mode allows position reduction."""
        config = RiskConfig(enable_pre_trade_liq_projection=False)
        validator = PreTradeValidator(config, mock_event_store, is_reduce_only_mode=lambda: True)

        # Allow reduction of long position - adjust quantity to stay within exposure limits
        positions = {
            "BTC-USD": {"side": "long", "quantity": "0.01", "price": "50000"}
        }  # Small existing position
        try:
            validator.pre_trade_validate(
                "BTC-USD",
                "sell",
                qty=Decimal("0.005"),  # Small reduction to stay within exposure limits
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions=positions,
            )
        except ValidationError:
            pytest.fail("Should not raise ValidationError for position reduction")

        # Allow reduction of short position
        positions = {
            "BTC-USD": {"side": "short", "quantity": "0.01", "price": "50000"}
        }  # Small existing position
        try:
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("0.005"),  # Small reduction to stay within exposure limits
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions=positions,
            )
        except ValidationError:
            pytest.fail("Should not raise ValidationError for position reduction")

    def test_reduce_only_mode_blocks_new_position_when_no_existing(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test reduce-only mode blocks new positions when no existing position."""
        config = RiskConfig(enable_pre_trade_liq_projection=False)
        validator = PreTradeValidator(config, mock_event_store, is_reduce_only_mode=lambda: True)

        with pytest.raises(ValidationError, match="Reduce-only mode"):
            validator.pre_trade_validate(
                "BTC-USD",
                "sell",
                qty=Decimal("0.1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={},
            )
