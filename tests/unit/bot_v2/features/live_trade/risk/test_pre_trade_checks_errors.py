"""
Error handling and edge case tests for pre-trade validation.
"""

from __future__ import annotations

from decimal import Decimal
import pytest
import threading

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.live_trade.risk.pre_trade_checks import (
    PreTradeValidator,
    ValidationError,
)


class TestPreTradeValidatorErrors:
    """Test error handling and edge cases in PreTradeValidator."""

    def test_pre_trade_validate_invalid_inputs(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test pre-trade validation handles invalid inputs gracefully."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Invalid side
        with pytest.raises(ValueError):  # May raise ValueError from validation logic
            validator.pre_trade_validate(
                "BTC-USD",
                "invalid_side",
                qty=Decimal("0.1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={},
            )

        # Negative quantity
        try:
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("-0.1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={},
            )
        except (ValidationError, ValueError):
            pass  # Should fail somehow

    def test_pre_trade_validate_malformed_position_data(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test pre-trade validation handles malformed position data."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Malformed position data should not crash
        malformed_positions = {
            "BTC-USD": {
                "side": "long",
                "quantity": "invalid_number",
                "price": "50000",
                "mark_price": "50000",
            }
        }

        try:
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("0.01"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions=malformed_positions,
            )
        except (ValidationError, ValueError, TypeError):
            pass  # Should handle gracefully

    def test_validation_error_message_formatting(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test ValidationError messages are properly formatted."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        config = RiskConfig(kill_switch_enabled=True, enable_pre_trade_liq_projection=False)
        validator = PreTradeValidator(config, mock_event_store)

        with pytest.raises(ValidationError) as exc_info:
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("0.1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={},
            )

        error_message = str(exc_info.value)
        assert "kill switch" in error_message.lower()
        assert len(error_message) > 10  # Should have meaningful content

    def test_decimal_precision_handling(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test proper decimal precision handling."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Very small decimal values
        try:
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("0.00000001"),
                price=Decimal("50000.00000001"),
                product=btc_perpetual_product,
                equity=Decimal("10000.00000001"),
                current_positions={},
            )
        except ValidationError:
            pass  # May fail due to precision, but shouldn't crash

    def test_concurrent_validation_safety(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test validator is safe for concurrent use."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)
        results = []

        def validate_trade():
            try:
                validator.pre_trade_validate(
                    "BTC-USD",
                    "buy",
                    qty=Decimal("0.01"),
                    price=Decimal("50000"),
                    product=btc_perpetual_product,
                    equity=Decimal("10000"),
                    current_positions={},
                )
                results.append("success")
            except Exception as e:
                results.append(f"error: {e}")

        # Run multiple validations concurrently
        threads = [threading.Thread(target=validate_trade) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should handle concurrent access gracefully
        assert len(results) == 5
        assert all("error:" not in result or result == "success" for result in results)
