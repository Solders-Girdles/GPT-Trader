#!/usr/bin/env python3
"""
Simple test to verify quantization is applied in live_trade.py.
"""

from decimal import Decimal
from unittest.mock import Mock, patch
from bot_v2.features.brokerages.core.interfaces import Product, MarketType


def test_enforce_perp_rules_is_imported_and_called():
    """Verify that enforce_perp_rules is imported and would be called."""

    # This test simply verifies the import and function structure
    # The actual testing happens in scripts/validation/validate_perps_strategy_integration.py

    # Verify we can import the function
    from bot_v2.features.brokerages.coinbase.utilities import enforce_perp_rules

    # Test the function with mock data
    product = Product(
        symbol="BTC-PERP",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        step_size=Decimal("0.001"),
        min_size=Decimal("0.001"),
        price_increment=Decimal("0.01"),
        min_notional=Decimal("10"),
    )

    # Test quantization
    quantity = Decimal("1.23456789")
    price = Decimal("50123.456789")

    quantized_quantity, quantized_price = enforce_perp_rules(
        product=product,
        quantity=quantity,
        price=price,
    )

    # Verify quantization happened
    assert quantized_quantity == Decimal("1.234")  # Rounded to step_size (0.001)
    assert quantized_price == Decimal("50123.45")  # Rounded to price_increment (0.01)

    print("✓ enforce_perp_rules imported and functioning")


def test_run_strategy_applies_quantization(monkeypatch):
    """Behavioral: run_strategy applies quantization before execution."""
    # Arrange a connected environment
    from bot_v2.features.live_trade import live_trade

    live_trade.disconnect()
    live_trade.connect_broker(broker_name="simulated")

    # Provide a deterministic mark price via mark_cache to avoid network calls
    symbol = "BTC-PERP"
    mark_cache = {symbol: Decimal("50000")}

    # Dummy strategy that produces a BUY decision with target notional
    from bot_v2.features.live_trade.strategies.perps_baseline import Decision, Action

    class DummyStrategy:
        def decide(self, **kwargs):
            return Decision(action=Action.BUY, reason="test", target_notional=Decimal("1000"))

    # Patch the quantization helper to capture invocation and control return
    expected_quantity = Decimal("1000") / Decimal("50000")  # 0.02
    with patch("bot_v2.features.brokerages.coinbase.utilities.enforce_perp_rules") as mock_enforce:
        mock_enforce.return_value = (expected_quantity, mark_cache[symbol])

        # Act
        decisions = live_trade.run_strategy(DummyStrategy(), [symbol], mark_cache=mark_cache)

        # Assert quantization helper was called with expected arguments
        assert mock_enforce.called, "Quantization should be applied for actionable decisions"
        kwargs = mock_enforce.call_args.kwargs
        assert kwargs["product"].symbol == symbol
        assert kwargs["quantity"] == expected_quantity
        assert kwargs["price"] == mark_cache[symbol]

        # Sanity: decision produced for the symbol
        assert symbol in decisions

    # Cleanup
    live_trade.disconnect()


if __name__ == "__main__":
    test_enforce_perp_rules_is_imported_and_called()
    test_run_strategy_code_has_quantization()
    print("\n✅ All quantization tests passed!")
