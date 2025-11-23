"""
Leverage validation tests for pre-trade checks.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

import pytest

from bot_v2.features.live_trade.risk.pre_trade_checks import (
    PreTradeValidator,
    ValidationError,
)


class TestPreTradeValidatorLeverage:
    """Leverage validation tests for PreTradeValidator."""

    # -------------------------------------------------------------------------
    # LEVERAGE VALIDATION TESTS
    # -------------------------------------------------------------------------

    def test_validate_leverage_within_limits(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test leverage validation passes when within limits."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Within 1x leverage limit
        try:
            validator.validate_leverage(
                "BTC-USD",
                quantity=Decimal("0.1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
            )
        except ValidationError:
            pytest.fail("Should not raise ValidationError for valid leverage")

    def test_validate_leverage_exceeds_symbol_limit(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test leverage validation fails when symbol limit exceeded."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Exceeds 1x leverage limit (1 BTC * $50000 = $50000 notional, equity $10000)
        with pytest.raises(ValidationError, match="Leverage.*exceeds.*cap"):
            validator.validate_leverage(
                "BTC-USD",
                quantity=Decimal("1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
            )

    def test_validate_leverage_edge_cases(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test leverage validation edge cases."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Zero equity
        with pytest.raises(ValidationError, match="Leverage.*exceeds.*cap"):
            validator.validate_leverage(
                "BTC-USD",
                quantity=Decimal("0.1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("0"),
            )

        # Zero quantity
        try:
            validator.validate_leverage(
                "BTC-USD",
                quantity=Decimal("0"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
            )
        except ValidationError:
            pytest.fail("Zero quantity should be valid")

        # Very small position
        try:
            validator.validate_leverage(
                "BTC-USD",
                quantity=Decimal("0.0001"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
            )
        except ValidationError:
            pytest.fail("Small position should be valid")

    def test_validate_leverage_with_existing_positions(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test leverage validation with high leverage values."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Test with high leverage that exceeds symbol cap
        with pytest.raises(ValidationError, match="Leverage.*exceeds.*cap"):
            validator.validate_leverage(
                "BTC-USD",
                quantity=Decimal("5"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
            )

    # -------------------------------------------------------------------------
    # LIQUIDATION BUFFER VALIDATION TESTS
    # -------------------------------------------------------------------------

    @patch("bot_v2.features.live_trade.risk.pre_trade_checks.effective_mmr")
    def test_validate_liquidation_buffer_sufficient_margin(
        self, mock_effective_mmr, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test liquidation buffer validation passes with sufficient margin."""
        mock_effective_mmr.return_value = Decimal("0.1")  # 10% MMR
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        try:
            validator.validate_liquidation_buffer(
                "BTC-USD",
                quantity=Decimal("0.01"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
            )
        except ValidationError:
            pytest.fail("Should not raise ValidationError for sufficient liquidation buffer")

    @patch("bot_v2.features.live_trade.risk.pre_trade_checks.effective_mmr")
    def test_validate_liquidation_buffer_insufficient_margin(
        self, mock_effective_mmr, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test liquidation buffer validation fails with insufficient margin."""
        mock_effective_mmr.return_value = Decimal("0.15")  # 15% MMR
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        with pytest.raises(ValidationError, match="Insufficient liquidation buffer"):
            validator.validate_liquidation_buffer(
                "BTC-USD",
                quantity=Decimal("0.1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("1000"),  # Low equity
            )

    @patch("bot_v2.features.live_trade.risk.pre_trade_checks.effective_mmr")
    def test_validate_liquidation_buffer_zero_equity(
        self, mock_effective_mmr, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test liquidation buffer validation with zero equity."""
        mock_effective_mmr.return_value = Decimal("0.1")
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        with pytest.raises(ValidationError, match="Insufficient liquidation buffer"):
            validator.validate_liquidation_buffer(
                "BTC-USD",
                quantity=Decimal("0.1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("0"),
            )
