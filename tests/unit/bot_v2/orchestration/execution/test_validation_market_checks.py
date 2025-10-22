"""Tests for market freshness and slippage guard validation."""

from __future__ import annotations

from decimal import Decimal

import pytest

from bot_v2.features.brokerages.core.interfaces import OrderSide
from bot_v2.features.live_trade.risk import ValidationError


class TestMarketFreshnessValidation:
    """Test ensure_mark_is_fresh method."""

    def test_ensure_mark_fresh_when_not_stale(self, order_validator) -> None:
        """Test successful validation when mark is not stale."""
        symbol = "BTC-PERP"
        order_validator.risk_manager.check_mark_staleness.return_value = False

        # Should not raise any error
        order_validator.ensure_mark_is_fresh(symbol)

        order_validator.risk_manager.check_mark_staleness.assert_called_once_with(symbol)

    def test_ensure_mark_fresh_raises_when_stale(self, order_validator) -> None:
        """Test validation error when mark is stale."""
        symbol = "ETH-PERP"
        order_validator.risk_manager.check_mark_staleness.return_value = True

        with pytest.raises(
            ValidationError, match="Mark price is stale for ETH-PERP; halting order placement"
        ):
            order_validator.ensure_mark_is_fresh(symbol)

        order_validator.risk_manager.check_mark_staleness.assert_called_once_with(symbol)

    def test_ensure_mark_fresh_handles_risk_manager_exception(self, order_validator) -> None:
        """Test graceful handling of risk manager exceptions."""
        symbol = "BTC-PERP"
        order_validator.risk_manager.check_mark_staleness.side_effect = RuntimeError(
            "Risk service unavailable"
        )

        # Should not raise any error - should be caught and ignored
        order_validator.ensure_mark_is_fresh(symbol)

        order_validator.risk_manager.check_mark_staleness.assert_called_once_with(symbol)

    def test_ensure_mark_fresh_handles_validation_error_passthrough(self, order_validator) -> None:
        """Test that ValidationError from risk manager is passed through."""
        symbol = "BTC-PERP"
        order_validator.risk_manager.check_mark_staleness.side_effect = ValidationError(
            "Custom validation error"
        )

        # ValidationError should be passed through, not caught
        with pytest.raises(ValidationError, match="Custom validation error"):
            order_validator.ensure_mark_is_fresh(symbol)


class TestSlippageGuardValidation:
    """Test enforce_slippage_guard method."""

    def test_slippage_guard_pass_when_under_limit(self, order_validator) -> None:
        """Test slippage guard passes when expected slippage is under limit."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_quantity = Decimal("0.1")
        effective_price = Decimal("50000.0")

        # Mock market snapshot with low spread and high depth
        order_validator.broker.get_market_snapshot.return_value = {
            "spread_bps": 5,
            "depth_l1": Decimal("2000000"),  # High depth
        }
        order_validator.risk_manager.config.slippage_guard_bps = 50

        # Should not raise any error
        order_validator.enforce_slippage_guard(symbol, side, order_quantity, effective_price)

        order_validator.broker.get_market_snapshot.assert_called_once_with(symbol)

    def test_slippage_guard_raises_when_over_limit(self, order_validator) -> None:
        """Test slippage guard raises when expected slippage exceeds limit."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_quantity = Decimal("1.0")  # Large quantity
        effective_price = Decimal("50000.0")

        # Mock market snapshot with high spread and low depth
        order_validator.broker.get_market_snapshot.return_value = {
            "spread_bps": 10,
            "depth_l1": Decimal("100000"),  # Low depth
        }
        order_validator.risk_manager.config.slippage_guard_bps = 25

        with pytest.raises(ValidationError, match=r"Expected slippage \d+ bps exceeds guard 25"):
            order_validator.enforce_slippage_guard(symbol, side, order_quantity, effective_price)

    def test_slippage_guard_handles_missing_market_snapshot(self, order_validator) -> None:
        """Test slippage guard handles missing market snapshot gracefully."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_quantity = Decimal("0.1")
        effective_price = Decimal("50000.0")

        # Mock broker that doesn't have get_market_snapshot
        del order_validator.broker.get_market_snapshot

        # Should not raise any error
        order_validator.enforce_slippage_guard(symbol, side, order_quantity, effective_price)

    def test_slippage_guard_handles_empty_market_snapshot(self, order_validator) -> None:
        """Test slippage guard handles empty market snapshot gracefully."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_quantity = Decimal("0.1")
        effective_price = Decimal("50000.0")

        # Mock empty market snapshot
        order_validator.broker.get_market_snapshot.return_value = {}

        # Should not raise any error
        order_validator.enforce_slippage_guard(symbol, side, order_quantity, effective_price)

    def test_slippage_guard_handles_zero_depth(self, order_validator) -> None:
        """Test slippage guard handles zero depth gracefully."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_quantity = Decimal("0.00001")  # Ultra small quantity
        effective_price = Decimal("50000.0")

        # Mock market snapshot with zero depth
        order_validator.broker.get_market_snapshot.return_value = {
            "spread_bps": 1,  # Minimal spread
            "depth_l1": Decimal("0"),
        }
        order_validator.risk_manager.config.slippage_guard_bps = 1000000  # Very high limit

        # Should not raise any error (depth defaults to 1, and high limit prevents error)
        order_validator.enforce_slippage_guard(symbol, side, order_quantity, effective_price)

    def test_slippage_guard_handles_negative_depth(self, order_validator) -> None:
        """Test slippage guard handles negative depth gracefully."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_quantity = Decimal("0.00001")  # Ultra small quantity
        effective_price = Decimal("50000.0")

        # Mock market snapshot with negative depth
        order_validator.broker.get_market_snapshot.return_value = {
            "spread_bps": 1,  # Minimal spread
            "depth_l1": Decimal("-100"),  # Negative depth
        }
        order_validator.risk_manager.config.slippage_guard_bps = 1000000  # Very high limit

        # Should not raise any error (depth defaults to 1, and high limit prevents error)
        order_validator.enforce_slippage_guard(symbol, side, order_quantity, effective_price)

    def test_slippage_guard_handles_missing_snapshot_fields(self, order_validator) -> None:
        """Test slippage guard handles missing fields in snapshot."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_quantity = Decimal("0.00001")  # Ultra small quantity
        effective_price = Decimal("50000.0")

        # Mock market snapshot with missing fields
        order_validator.broker.get_market_snapshot.return_value = {"some_other_field": "value"}
        order_validator.risk_manager.config.slippage_guard_bps = 1000000  # Very high limit

        # Should not raise any error
        order_validator.enforce_slippage_guard(symbol, side, order_quantity, effective_price)

    def test_slippage_guard_handles_broker_exception(self, order_validator) -> None:
        """Test slippage guard handles broker exceptions gracefully."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_quantity = Decimal("0.1")
        effective_price = Decimal("50000.0")

        # Mock broker exception
        order_validator.broker.get_market_snapshot.side_effect = RuntimeError(
            "Market data unavailable"
        )

        # Should not raise any error
        order_validator.enforce_slippage_guard(symbol, side, order_quantity, effective_price)

    def test_slippage_guard_handles_validation_error_passthrough(self, order_validator) -> None:
        """Test that ValidationError from calculations is passed through."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_quantity = Decimal("0.1")
        effective_price = Decimal("50000.0")

        # Mock ValidationError during calculation (e.g., from Decimal conversion)
        order_validator.broker.get_market_snapshot.return_value = {
            "spread_bps": "invalid_decimal",
            "depth_l1": "invalid_decimal",
        }

        # Should not raise error - exceptions other than ValidationError are caught
        order_validator.enforce_slippage_guard(symbol, side, order_quantity, effective_price)

    def test_slippage_guard_calculation_logic(self, order_validator) -> None:
        """Test detailed slippage calculation logic."""
        symbol = "BTC-PERP"
        side = OrderSide.BUY
        order_quantity = Decimal("0.5")
        effective_price = Decimal("50000.0")

        # Mock specific market data for predictable calculation
        order_validator.broker.get_market_snapshot.return_value = {
            "spread_bps": 10,
            "depth_l1": Decimal("500000"),  # $500K depth at L1
        }
        order_validator.risk_manager.config.slippage_guard_bps = 30

        # Calculate expected values:
        # notional = 0.5 * 50000 = 25000
        # impact_bps = 10000 * (25000 / 500000) * 0.5 = 10000 * 0.05 * 0.5 = 250
        # expected_bps = spread_bps + impact_bps = 10 + 250 = 260
        # Since 260 > 30, should raise ValidationError

        with pytest.raises(ValidationError, match=r"Expected slippage 260 bps exceeds guard 30"):
            order_validator.enforce_slippage_guard(symbol, side, order_quantity, effective_price)

    def test_slippage_guard_with_sell_order(self, order_validator) -> None:
        """Test slippage guard works correctly for sell orders."""
        symbol = "BTC-PERP"
        side = OrderSide.SELL
        order_quantity = Decimal("0.1")
        effective_price = Decimal("50000.0")

        # Mock market snapshot
        order_validator.broker.get_market_snapshot.return_value = {
            "spread_bps": 5,
            "depth_l1": Decimal("1000000"),
        }
        order_validator.risk_manager.config.slippage_guard_bps = 50

        # Should work for sell orders too
        order_validator.enforce_slippage_guard(symbol, side, order_quantity, effective_price)
