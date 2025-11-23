"""
Exposure and correlation risk tests for pre-trade checks.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from bot_v2.features.live_trade.risk.pre_trade_checks import (
    PreTradeValidator,
    ValidationError,
)


class TestPreTradeValidatorExposure:
    """Exposure and correlation risk tests for PreTradeValidator."""

    # -------------------------------------------------------------------------
    # EXPOSURE LIMIT VALIDATION TESTS
    # -------------------------------------------------------------------------

    def test_validate_exposure_limits_within_bounds(
        self, conservative_risk_config, mock_event_store
    ):
        """Test exposure validation passes when within limits."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # 5% of $10000 equity = $500 notional limit, $450 is within bounds
        try:
            validator.validate_exposure_limits(
                "BTC-USD", notional=Decimal("450"), equity=Decimal("10000"), current_positions={}
            )
        except ValidationError:
            pytest.fail("Should not raise ValidationError for valid exposure")

    def test_validate_exposure_limits_exceeds_symbol_cap(
        self, conservative_risk_config, mock_event_store
    ):
        """Test exposure validation fails when symbol cap exceeded."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # 20% of $10000 equity = $2000 notional limit, $2500 exceeds limit
        with pytest.raises(ValidationError, match="Symbol exposure.*exceeds.*cap"):
            validator.validate_exposure_limits(
                "BTC-USD", notional=Decimal("2500"), equity=Decimal("10000"), current_positions={}
            )

    def test_validate_exposure_limits_with_existing_positions(
        self, conservative_risk_config, mock_event_store
    ):
        """Test exposure validation considers existing positions."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Add existing position that uses part of the allocation
        current_positions = {
            "BTC-USD": {"side": "long", "quantity": "0.02", "price": "50000", "mark_price": "50000"}
        }  # $1000 notional

        # New position should exceed total symbol cap
        with pytest.raises(ValidationError, match="Symbol exposure.*exceeds.*cap"):
            validator.validate_exposure_limits(
                "BTC-USD",
                notional=Decimal("1500"),  # Combined would be $2500 > 20% of $10000 = $2000
                equity=Decimal("10000"),
                current_positions=current_positions,
            )

    def test_validate_exposure_limits_zero_equity(self, conservative_risk_config, mock_event_store):
        """Test exposure validation with zero equity."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        with pytest.raises(ValidationError, match="Symbol exposure.*exceeds.*cap"):
            validator.validate_exposure_limits(
                "BTC-USD", notional=Decimal("100"), equity=Decimal("0"), current_positions={}
            )

    # -------------------------------------------------------------------------
    # CORRELATION RISK VALIDATION TESTS
    # -------------------------------------------------------------------------

    def test_validate_correlation_risk_within_limits(
        self, aggressive_risk_config, mock_event_store
    ):
        """Test correlation risk validation passes when within limits."""
        validator = PreTradeValidator(aggressive_risk_config, mock_event_store)

        # Add positions with moderate correlation
        current_positions = {
            "BTC-USD": {"side": "long", "quantity": "0.1", "price": "50000", "mark_price": "50000"},
            "ETH-USD": {"side": "long", "quantity": "1", "price": "3000", "mark_price": "3000"},
        }

        try:
            validator.validate_correlation_risk(
                "BTC-USD", notional=Decimal("1000"), current_positions=current_positions
            )
        except ValidationError:
            pytest.fail("Should not raise ValidationError for acceptable correlation")

    def test_validate_correlation_risk_exceeds_limit(
        self, conservative_risk_config, mock_event_store
    ):
        """Test correlation risk validation fails when limit exceeded."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Add highly correlated positions (both large crypto longs)
        current_positions = {
            "BTC-USD": {"side": "long", "quantity": "0.5", "price": "50000", "mark_price": "50000"},
            "ETH-USD": {"side": "long", "quantity": "10", "price": "3000", "mark_price": "3000"},
            "SOL-USD": {"side": "long", "quantity": "100", "price": "100", "mark_price": "100"},
        }

        with pytest.raises(ValidationError, match="correlation risk"):
            validator.validate_correlation_risk(
                "BTC-USD",
                notional=Decimal("5000"),  # Large additional exposure
                current_positions=current_positions,
            )

    def test_validate_correlation_risk_no_existing_positions(
        self, conservative_risk_config, mock_event_store
    ):
        """Test correlation risk validation with no existing positions."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        try:
            validator.validate_correlation_risk(
                "BTC-USD", notional=Decimal("10000"), current_positions={}
            )
        except ValidationError:
            pytest.fail("Should not raise ValidationError with no existing positions")
