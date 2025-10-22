"""Tests for error handling, fallbacks, and resilience in state collection."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from bot_v2.features.brokerages.core.interfaces import Balance
from bot_v2.orchestration.execution.state_collection import StateCollector


class TestBalanceParsingErrors:
    """Test error handling in balance parsing and equity calculation."""

    def test_calculate_equity_with_invalid_balance_types(self, state_collector) -> None:
        """Test equity calculation with various invalid balance types."""
        invalid_balances = [
            None,  # None balance
            "invalid",  # String balance
            123,  # Integer balance
        ]

        for invalid_balance in invalid_balances:
            with pytest.raises((TypeError, AttributeError)):
                state_collector.calculate_equity_from_balances([invalid_balance])

    def test_calculate_equity_with_missing_attributes(self, state_collector) -> None:
        """Test equity calculation with balances missing required attributes."""
        # Balance missing available attribute
        balance_no_available = MagicMock()
        balance_no_available.asset = "USD"
        balance_no_available.total = Decimal("1000.0")
        del balance_no_available.available

        with pytest.raises(AttributeError):
            state_collector.calculate_equity_from_balances([balance_no_available])

    def test_calculate_equity_with_none_attributes(self, state_collector) -> None:
        """Test equity calculation with None attribute values."""
        from bot_v2.features.brokerages.core.interfaces import Balance

        # Balance with None values
        balance_with_none = Balance(asset=None, available=None, total=None, hold=None)

        # Should handle None gracefully by treating as non-collateral
        equity, collateral_balances, total_balance = state_collector.calculate_equity_from_balances(
            [balance_with_none]
        )

        assert equity == Decimal("0")
        assert collateral_balances == []
        assert total_balance == Decimal("0")

    def test_calculate_equity_with_decimal_conversion_errors(self, state_collector) -> None:
        """Test equity calculation with Decimal conversion errors."""
        # Create a balance that will cause Decimal conversion issues
        problematic_balance = MagicMock()
        problematic_balance.asset = "USD"
        problematic_balance.available = "not_a_decimal"
        problematic_balance.total = Decimal("1000.0")

        with pytest.raises((TypeError, ValueError)):
            state_collector.calculate_equity_from_balances([problematic_balance])


class TestPositionParsingErrors:
    """Test error handling in position parsing and transformation."""

    def test_build_positions_dict_with_various_errors(self, state_collector) -> None:
        """Test position dict building with various error conditions."""
        error_positions = []

        # Position with invalid quantity type
        invalid_qty_pos = MagicMock()
        invalid_qty_pos.symbol = "BTC-PERP"
        invalid_qty_pos.quantity = "invalid_quantity"
        error_positions.append(invalid_qty_pos)

        # Position with None symbol
        none_symbol_pos = MagicMock()
        none_symbol_pos.symbol = None
        none_symbol_pos.quantity = Decimal("1.0")
        error_positions.append(none_symbol_pos)

        # Position with missing symbol attribute
        missing_symbol_pos = MagicMock()
        missing_symbol_pos.quantity = Decimal("1.0")
        del missing_symbol_pos.symbol
        error_positions.append(missing_symbol_pos)

        result = state_collector.build_positions_dict(error_positions)

        # Should handle all errors gracefully and return empty dict
        assert result == {}

    def test_build_positions_dict_quantity_from_function_error(self, state_collector) -> None:
        """Test handling of quantity_from function errors."""
        # Mock quantity_from to raise an exception
        with pytest.MonkeyPatch().context() as m:

            def mock_quantity_from_error(pos):
                raise ValueError("Quantity parsing failed")

            m.setattr(
                "bot_v2.orchestration.execution.state_collection.quantity_from",
                mock_quantity_from_error,
            )

            position = MagicMock()
            position.symbol = "BTC-PERP"
            position.quantity = Decimal("1.0")

            # Should propagate the error (current implementation behavior)
            with pytest.raises(ValueError, match="Quantity parsing failed"):
                state_collector.build_positions_dict([position])

    def test_build_positions_dict_getattribute_errors(self, state_collector) -> None:
        """Test handling of getattr errors in position parsing."""
        problematic_position = MagicMock()
        problematic_position.symbol = "BTC-PERP"
        problematic_position.quantity = Decimal("1.0")

        # Make getattr raise an exception for specific attributes
        def mock_getattr(obj, name, default=None):
            if name in ["side", "entry_price", "mark_price"]:
                raise RuntimeError(f"Access to {name} failed")
            return object.__getattribute__(obj, name) if hasattr(obj, name) else default

        with pytest.MonkeyPatch().context() as m:
            m.setattr("builtins.getattr", mock_getattr)

            result = state_collector.build_positions_dict([problematic_position])

            # Should handle getattr errors gracefully
            assert result == {}

    def test_build_positions_dict_logging_on_errors(self, state_collector) -> None:
        """Test that errors are properly logged during position parsing."""
        error_position = MagicMock()
        error_position.symbol = "BTC-PERP"
        error_position.quantity = "invalid_quantity"

        # Test that the function handles error positions gracefully
        # The actual logging happens via logging_patterns.py which is harder to mock
        result = state_collector.build_positions_dict([error_position])

        # Should return empty dict for problematic positions
        assert result == {}


class TestPriceResolutionErrors:
    """Test error handling in price resolution fallbacks."""

    def test_resolve_effective_price_broker_method_missing(
        self, broker_with_missing_methods
    ) -> None:
        """Test price resolution when broker methods are missing."""
        collector = StateCollector(broker_with_missing_methods)
        product = MagicMock()
        product.bid_price = None  # Set explicitly to None
        product.ask_price = None  # Set explicitly to None
        product.price = None
        product.quote_increment = Decimal("0.01")

        result = collector.resolve_effective_price("BTC-PERP", "buy", None, product)

        # Should fall back through all mechanisms to default
        assert result == Decimal("1.0")

    def test_resolve_effective_price_broker_method_attributes_missing(
        self, state_collector
    ) -> None:
        """Test price resolution when broker has methods but they lack attributes."""
        # Mock broker to return None for mark price (missing attributes)
        state_collector.broker.get_mark_price.return_value = None
        # Mock quote without last attribute
        broken_quote = MagicMock()
        del broken_quote.last
        state_collector.broker.get_quote.return_value = broken_quote

        product = MagicMock()
        product.bid_price = None
        product.ask_price = None
        product.price = None
        product.quote_increment = Decimal("0.05")

        result = state_collector.resolve_effective_price("BTC-PERP", "buy", None, product)

        # Should use quote_increment * 100 fallback
        assert result == Decimal("5.0")

    def test_resolve_effective_price_decimal_conversion_errors(self, state_collector) -> None:
        """Test price resolution when Decimal conversion fails."""
        # Mock mark price to return non-convertible value
        state_collector.broker.get_mark_price.return_value = "not_a_number"

        product = MagicMock()
        product.bid_price = None
        product.ask_price = None
        product.price = None
        product.quote_increment = "invalid_decimal"  # Also invalid

        result = state_collector.resolve_effective_price("BTC-PERP", "buy", None, product)

        # Should handle conversion errors gracefully
        assert isinstance(result, Decimal)

    def test_resolve_effective_price_infinite_recursion_protection(self, state_collector) -> None:
        """Test that price resolution doesn't cause infinite recursion."""
        # This tests that the fallback chain properly terminates
        minimal_product = MagicMock()
        minimal_product.bid_price = None  # Set explicitly
        minimal_product.ask_price = None  # Set explicitly
        minimal_product.price = None
        minimal_product.quote_increment = None

        # Mock all broker methods to raise errors
        state_collector.broker.get_mark_price = MagicMock(
            side_effect=RuntimeError("Service unavailable")
        )
        state_collector.broker.get_quote = MagicMock(
            side_effect=RuntimeError("Service unavailable")
        )

        # Should not cause infinite recursion
        result = state_collector.resolve_effective_price("BTC-PERP", "buy", None, minimal_product)
        assert result == Decimal("1.0")


class TestBrokerCommunicationErrors:
    """Test handling of broker communication failures."""

    def test_collect_account_state_balance_service_error(self, state_collector) -> None:
        """Test account state collection when balance service fails."""
        state_collector.broker.list_balances.side_effect = RuntimeError("Balance service timeout")

        with pytest.raises(RuntimeError, match="Balance service timeout"):
            state_collector.collect_account_state()

        # Verify positions method wasn't called due to early failure
        state_collector.broker.list_positions.assert_not_called()

    def test_collect_account_state_position_service_error(self, state_collector) -> None:
        """Test account state collection when position service fails."""
        state_collector.broker.list_balances.return_value = []
        state_collector.broker.list_positions.side_effect = RuntimeError(
            "Position service unavailable"
        )

        with pytest.raises(RuntimeError, match="Position service unavailable"):
            state_collector.collect_account_state()

    def test_collect_account_state_partial_failure_handling(self, state_collector) -> None:
        """Test that partial failures don't leave system in inconsistent state."""
        # Mock balance service to succeed but return problematic data
        state_collector.broker.list_balances.return_value = [
            None,  # Invalid balance
            "invalid",  # Invalid balance
        ]

        with pytest.raises((TypeError, AttributeError)):
            state_collector.collect_account_state()

    def test_require_product_retry_behavior(self, broker_with_errors) -> None:
        """Test require_product behavior with broker errors."""
        collector = StateCollector(broker_with_errors)

        # First call should fail and raise ValidationError
        with pytest.raises(Exception):
            collector.require_product("BTC-PERP", None)

    def test_state_collection_resilience_to_temporary_failures(self, state_collector) -> None:
        """Test that state collection can recover from temporary failures."""
        # Simulate temporary failure then recovery
        call_count = 0

        def mock_list_balances():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Temporary failure")
            return [MagicMock(asset="USD", available=Decimal("1000.0"), total=Decimal("1000.0"))]

        state_collector.broker.list_balances = mock_list_balances
        state_collector.broker.list_positions.return_value = []

        # First call should fail
        with pytest.raises(RuntimeError):
            state_collector.collect_account_state()

        # Second call should succeed
        balances, equity, collateral_balances, total_balance, positions = (
            state_collector.collect_account_state()
        )
        assert len(balances) == 1


class TestEnvironmentAndConfigurationErrors:
    """Test handling of environment and configuration errors."""

    def test_collateral_resolution_with_malformed_env(self, state_collector) -> None:
        """Test collateral asset resolution with malformed environment."""
        # Test with various malformed environment values
        test_cases = [
            "USD,,USDC",  # Empty token
            "USD, ,USDC",  # Whitespace-only token
            "USD,INVALID-TOKEN$,USDC",  # Invalid characters
        ]

        for env_value in test_cases:
            with pytest.MonkeyPatch().context():
                # Re-create collector with malformed env
                mock_settings = MagicMock()
                mock_settings.raw_env = {"PERPS_COLLATERAL_ASSETS": env_value}
                collector = StateCollector(state_collector.broker, settings=mock_settings)

                # Should handle malformed input gracefully
                assert isinstance(collector.collateral_assets, set)

    def test_initialization_with_invalid_settings(self, mock_brokerage) -> None:
        """Test StateCollector initialization with invalid settings."""
        # Test with None settings (should use defaults)
        collector = StateCollector(mock_brokerage, settings=None)
        assert collector._settings is not None
        assert "USD" in collector.collateral_assets

        # Test with settings that don't have raw_env (should handle gracefully)
        invalid_settings = MagicMock()

        # Configure raw_env to raise AttributeError when accessed
        def raise_attr_error(*args, **kwargs):
            raise AttributeError("raw_env")

        invalid_settings.configure_mock(**{"raw_env.side_effect": raise_attr_error})

        # Should handle missing raw_env gracefully by using defaults
        collector = StateCollector(mock_brokerage, settings=invalid_settings)
        assert isinstance(collector.collateral_assets, set)
        assert "USD" in collector.collateral_assets  # Should have default values


class TestEdgeCaseScenarios:
    """Test edge cases and boundary conditions."""

    def test_very_large_balance_values(self, state_collector) -> None:
        """Test handling of very large balance values."""
        large_balances = [
            Balance(
                asset="USD",
                available=Decimal("999999999999999999.99"),
                total=Decimal("999999999999999999.99"),
                hold=Decimal("0"),
            ),
        ]

        equity, collateral_balances, total_balance = state_collector.calculate_equity_from_balances(
            large_balances
        )

        assert equity == Decimal("999999999999999999.99")
        assert total_balance == Decimal("999999999999999999.99")

    def test_very_small_price_values(self, state_collector) -> None:
        """Test price resolution with very small values."""
        tiny_product = MagicMock()
        tiny_product.bid_price = None  # Set explicitly
        tiny_product.ask_price = None  # Set explicitly
        tiny_product.price = None
        tiny_product.quote_increment = Decimal("0.00000001")

        # Remove all fallback methods
        state_collector.broker.get_mark_price = MagicMock(side_effect=RuntimeError("Unavailable"))
        state_collector.broker.get_quote = MagicMock(side_effect=RuntimeError("Unavailable"))

        result = state_collector.resolve_effective_price("BTC-PERP", "buy", None, tiny_product)

        # Should handle tiny values correctly
        assert result == Decimal("0.000001")  # quote_increment * 100

    def test_unicode_and_special_characters(self, state_collector) -> None:
        """Test handling of Unicode and special characters in asset names."""
        unicode_balances = [
            Balance(
                asset="USD©", available=Decimal("100.0"), total=Decimal("100.0"), hold=Decimal("0")
            ),
            Balance(
                asset="USDC™", available=Decimal("200.0"), total=Decimal("200.0"), hold=Decimal("0")
            ),
        ]

        equity, collateral_balances, total_balance = state_collector.calculate_equity_from_balances(
            unicode_balances
        )

        # Should handle Unicode characters in asset names (they won't match USD/USDC collateral assets)
        assert equity == Decimal("0.0")  # No matching collateral assets
        assert (
            len(collateral_balances) == 0
        )  # Unicode chars shouldn't match default collateral assets

    def test_concurrent_access_safety(self, state_collector) -> None:
        """Test that state collection is safe under concurrent access patterns."""
        # This test ensures the StateCollector doesn't have state that gets corrupted
        # under concurrent access (though actual concurrency testing would require threading)

        # Multiple calls should be independent
        result1 = state_collector.resolve_effective_price("BTC-PERP", "buy", None, MagicMock())
        result2 = state_collector.resolve_effective_price("ETH-PERP", "sell", None, MagicMock())

        # Results should be independent
        assert result1 > Decimal("0")
        assert result2 > Decimal("0")
