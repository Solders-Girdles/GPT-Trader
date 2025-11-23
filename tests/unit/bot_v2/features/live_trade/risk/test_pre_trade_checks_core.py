"""
Basic tests for pre-trade validation system (initialization, helpers, kill switch).
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from unittest.mock import Mock

import pytest

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.live_trade.risk.pre_trade_checks import (
    PreTradeValidator,
    ValidationError,
    _coalesce_quantity,
    _to_decimal,
)


class TestPreTradeValidatorCore:
    """Core test suite for PreTradeValidator functionality (initialization, helpers, kill switch)."""

    # -------------------------------------------------------------------------
    # VALIDATION HELPER FUNCTIONS
    # -------------------------------------------------------------------------

    def test_coalesce_quantity_success(self):
        """Test _coalesce_quantity with valid inputs."""
        result = _coalesce_quantity(None, Decimal("1.5"), Decimal("2.0"))
        assert result == Decimal("1.5")

        result = _coalesce_quantity(Decimal("3.0"), None, Decimal("4.0"))
        assert result == Decimal("3.0")

        result = _coalesce_quantity(None, None, Decimal("5.0"))
        assert result == Decimal("5.0")

    def test_coalesce_quantity_failure(self):
        """Test _coalesce_quantity raises when all values are None."""
        with pytest.raises(TypeError, match="quantity must be provided"):
            _coalesce_quantity(None, None, None)

    def test_to_decimal_valid_inputs(self):
        """Test _to_decimal with various valid input types."""
        assert _to_decimal("123.45") == Decimal("123.45")
        assert _to_decimal(Decimal("67.89")) == Decimal("67.89")
        assert _to_decimal(123) == Decimal("123")
        assert _to_decimal(None) == Decimal("0")
        assert _to_decimal("") == Decimal("0")
        assert _to_decimal("null") == Decimal("0")

    def test_to_decimal_invalid_inputs(self):
        """Test _to_decimal gracefully handles invalid inputs."""
        assert _to_decimal("invalid") == Decimal("0")
        assert _to_decimal([], Decimal("1.0")) == Decimal("1.0")
        assert _to_decimal({}, Decimal("2.0")) == Decimal("2.0")
        # Test with infinity - handle more gracefully
        try:
            result = _to_decimal(float("inf"))
            # If it doesn't raise, it should be a valid Decimal or default
            assert isinstance(result, Decimal)
        except (ValueError, InvalidOperation, TypeError):
            # If it raises during Decimal conversion, it should return default
            assert _to_decimal(float("inf")) == Decimal("0")

    # -------------------------------------------------------------------------
    # INITIALIZATION TESTS
    # -------------------------------------------------------------------------

    def test_validator_initialization_default(self, conservative_risk_config, mock_event_store):
        """Test PreTradeValidator initialization with default parameters."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        assert validator.config is conservative_risk_config
        assert validator.event_store is mock_event_store
        assert hasattr(validator, "_is_reduce_only_mode")
        assert hasattr(validator, "_impact_estimator")

    def test_validator_initialization_with_callbacks(
        self, conservative_risk_config, mock_event_store
    ):
        """Test PreTradeValidator initialization with custom callbacks."""
        reduce_only_cb = Mock(return_value=False)
        impact_estimator = Mock()

        validator = PreTradeValidator(
            conservative_risk_config,
            mock_event_store,
            is_reduce_only_mode=reduce_only_cb,
            impact_estimator=impact_estimator,
        )

        assert validator._is_reduce_only_mode is reduce_only_cb
        assert validator._impact_estimator is impact_estimator

    # -------------------------------------------------------------------------
    # KILL SWITCH TESTS
    # -------------------------------------------------------------------------

    def test_kill_switch_blocks_all_trades(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test kill switch blocks all trade attempts."""
        config = RiskConfig(kill_switch_enabled=True, enable_pre_trade_liq_projection=False)
        validator = PreTradeValidator(config, mock_event_store)

        with pytest.raises(ValidationError, match="Kill switch enabled"):
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("0.1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={},
            )

        # Verify metric was emitted
        assert len(mock_event_store.metrics) >= 1
        metric = mock_event_store.get_last_metric()
        assert metric["metrics"].get("event_type") == "kill_switch"

    def test_kill_switch_disabled_allows_trades(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test trades proceed when kill switch is disabled."""
        config = RiskConfig(kill_switch_enabled=False, enable_pre_trade_liq_projection=False)
        validator = PreTradeValidator(config, mock_event_store)

        # Should not raise - adjust quantity to stay within exposure limits
        try:
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("0.01"),  # Reduced to stay within 20% exposure limit ($500/$10000 = 5%)
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={},
            )
        except ValidationError as e:
            pytest.fail(f"Unexpected ValidationError: {e}")
