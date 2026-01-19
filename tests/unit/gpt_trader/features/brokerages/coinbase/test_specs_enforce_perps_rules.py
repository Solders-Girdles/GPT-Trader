"""Tests for enforce-perps rules utilities."""

from __future__ import annotations

from decimal import Decimal

import pytest


class TestEnforcePerpsRules:
    """Test enforce_perp_rules function from utilities module.

    These tests verify the enforce_perp_rules utility that combines
    quantization and validation for perps orders.
    """

    @staticmethod
    def make_product():
        """Helper to create test product."""
        from gpt_trader.core import MarketType, Product

        return Product(
            symbol="BTC-PERP",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            step_size=Decimal("0.001"),
            min_size=Decimal("0.001"),
            price_increment=Decimal("0.05"),  # 5c ticks to observe rounding
            min_notional=Decimal("10"),
            leverage_max=20,
        )

    @pytest.mark.perps
    def test_quantity_rounds_to_step_and_enforces_min_size(self):
        """Test enforce_perp_rules rounds quantity and enforces minimums."""
        from gpt_trader.features.brokerages.coinbase.utilities import (
            InvalidRequestError,
            enforce_perp_rules,
        )

        p = self.make_product()

        # Below min size should raise
        with pytest.raises(InvalidRequestError):
            enforce_perp_rules(p, quantity=Decimal("0.0005"), price=Decimal("50000"))

        # Rounds down to nearest step
        q, pr = enforce_perp_rules(p, quantity=Decimal("1.234567"), price=Decimal("50000.1234"))
        assert q == Decimal("1.234")
        assert pr == Decimal("50000.10")  # price increment 0.05 → rounds down

    @pytest.mark.perps
    def test_min_notional_enforced(self):
        """Test enforce_perp_rules enforces minimum notional value."""
        from gpt_trader.features.brokerages.coinbase.utilities import (
            InvalidRequestError,
            enforce_perp_rules,
        )

        p = self.make_product()

        # Small quantity with valid rounding but too small notional fails
        with pytest.raises(InvalidRequestError):
            enforce_perp_rules(p, quantity=Decimal("0.001"), price=Decimal("1000.00"))
