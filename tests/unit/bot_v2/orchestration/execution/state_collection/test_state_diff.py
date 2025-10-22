"""Tests for position state transformation and price resolution."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from bot_v2.features.live_trade.risk import ValidationError
from bot_v2.orchestration.execution.state_collection import StateCollector


class TestBuildPositionsDict:
    """Test position dictionary building and transformation."""

    def test_build_positions_dict_with_valid_positions(self, state_collector, sample_positions) -> None:
        """Test building positions dict with valid position data."""
        result = state_collector.build_positions_dict(sample_positions)

        # Should include only non-zero positions
        assert len(result) == 2  # BTC-PERP and ETH-PERP (SOL-PERP has 0 quantity)

        # Verify BTC-PERP position
        btc_position = result["BTC-PERP"]
        assert btc_position["quantity"] == Decimal("0.5")
        assert btc_position["side"] == "long"
        assert btc_position["entry_price"] == Decimal("45000.0")
        assert btc_position["mark_price"] == Decimal("50000.0")

        # Verify ETH-PERP position
        eth_position = result["ETH-PERP"]
        assert eth_position["quantity"] == Decimal("-2.0")
        assert eth_position["side"] == "short"
        assert eth_position["entry_price"] == Decimal("3000.0")
        assert eth_position["mark_price"] == Decimal("3200.0")

    def test_build_positions_dict_filters_zero_quantity(self, state_collector) -> None:
        """Test that positions with zero quantity are filtered out."""
        zero_position = MagicMock()
        zero_position.symbol = "BTC-PERP"
        zero_position.quantity = Decimal("0.0")
        zero_position.side = "long"

        result = state_collector.build_positions_dict([zero_position])

        assert result == {}  # Should be empty

    def test_build_positions_dict_filters_none_quantity(self, state_collector) -> None:
        """Test that positions with None quantity are filtered out."""
        none_position = MagicMock()
        none_position.symbol = "BTC-PERP"
        none_position.quantity = None
        none_position.side = "long"

        result = state_collector.build_positions_dict([none_position])

        assert result == {}  # Should be empty

    def test_build_positions_dict_handles_parsing_errors(self, state_collector, error_positions) -> None:
        """Test graceful handling of position parsing errors."""
        result = state_collector.build_positions_dict(error_positions)

        # Should return empty dict due to parsing errors
        assert result == {}

    def test_build_positions_dict_handles_missing_attributes(self, state_collector) -> None:
        """Test handling of positions with missing attributes."""
        incomplete_position = MagicMock()
        incomplete_position.symbol = "BTC-PERP"
        incomplete_position.quantity = Decimal("1.0")
        # Missing side, entry_price, mark_price attributes (will cause Decimal conversion errors)

        result = state_collector.build_positions_dict([incomplete_position])

        # Current implementation logs errors and excludes problematic positions
        assert result == {}  # Position with missing attributes should be excluded

    def test_build_positions_dict_empty_list(self, state_collector) -> None:
        """Test building positions dict with empty list."""
        result = state_collector.build_positions_dict([])

        assert result == {}

    def test_build_positions_dict_duplicate_symbols(self, state_collector) -> None:
        """Test handling of duplicate symbols in position list."""
        position1 = MagicMock()
        position1.symbol = "BTC-PERP"
        position1.quantity = Decimal("0.5")
        position1.side = "long"
        position1.entry_price = Decimal("45000.0")  # Add missing attributes
        position1.mark_price = Decimal("50000.0")

        position2 = MagicMock()
        position2.symbol = "BTC-PERP"
        position2.quantity = Decimal("0.3")
        position2.side = "short"
        position2.entry_price = Decimal("46000.0")  # Add missing attributes
        position2.mark_price = Decimal("51000.0")

        result = state_collector.build_positions_dict([position1, position2])

        # Last position should win
        assert len(result) == 1
        assert result["BTC-PERP"]["quantity"] == Decimal("0.3")
        assert result["BTC-PERP"]["side"] == "short"

    def test_build_positions_dict_quantity_conversion(self, state_collector) -> None:
        """Test quantity conversion from different formats."""
        # Mock the quantity_from function to return specific values
        with pytest.MonkeyPatch().context() as m:
            def mock_quantity_from(pos):
                if hasattr(pos, 'quantity'):
                    return Decimal(str(pos.quantity))
                return None

            m.setattr("bot_v2.orchestration.execution.state_collection.quantity_from", mock_quantity_from)

            position = MagicMock()
            position.symbol = "BTC-PERP"
            position.quantity = "1.5"  # String quantity
            position.side = "long"
            position.entry_price = Decimal("45000.0")  # Add missing attributes
            position.mark_price = Decimal("50000.0")

            result = state_collector.build_positions_dict([position])

            assert result["BTC-PERP"]["quantity"] == Decimal("1.5")


class TestResolveEffectivePrice:
    """Test price resolution with multiple fallback mechanisms."""

    def test_resolve_effective_price_with_valid_price(self, state_collector, sample_product) -> None:
        """Test price resolution when valid price is provided."""
        price = Decimal("50000.0")
        result = state_collector.resolve_effective_price("BTC-PERP", "buy", price, sample_product)

        assert result == price

    def test_resolve_effective_price_zero_price_fallback(self, state_collector, sample_product) -> None:
        """Test price resolution when provided price is zero."""
        price = Decimal("0")
        result = state_collector.resolve_effective_price("BTC-PERP", "buy", price, sample_product)

        assert result > Decimal("0")  # Should fallback to mark price

    def test_resolve_effective_price_none_price_fallback(self, state_collector, sample_product) -> None:
        """Test price resolution when provided price is None."""
        result = state_collector.resolve_effective_price("BTC-PERP", "buy", None, sample_product)

        assert result == Decimal("50005.0")  # Should use mark price

    def test_resolve_effective_price_negative_price_fallback(self, state_collector, sample_product) -> None:
        """Test price resolution when provided price is negative."""
        price = Decimal("-100")
        result = state_collector.resolve_effective_price("BTC-PERP", "buy", price, sample_product)

        assert result > Decimal("0")  # Should fallback to mark price

    def test_resolve_effective_price_uses_mark_price(self, state_collector, sample_product) -> None:
        """Test price resolution uses mark price from broker."""
        state_collector.broker.get_mark_price.return_value = Decimal("50100.0")

        result = state_collector.resolve_effective_price("BTC-PERP", "buy", None, sample_product)

        assert result == Decimal("50100.0")
        state_collector.broker.get_mark_price.assert_called_once_with("BTC-PERP")

    def test_resolve_effective_price_mark_price_zero_fallback(self, state_collector, sample_product) -> None:
        """Test mark price fallback when mark price is zero."""
        state_collector.broker.get_mark_price.return_value = Decimal("0")

        result = state_collector.resolve_effective_price("BTC-PERP", "buy", None, sample_product)

        # Should fall back to mid-price
        expected_mid = (Decimal("50000.0") + Decimal("50010.0")) / Decimal("2")
        assert result == expected_mid

    def test_resolve_effective_price_mark_price_error_fallback(self, state_collector, sample_product) -> None:
        """Test mark price fallback when mark price service errors."""
        state_collector.broker.get_mark_price.side_effect = RuntimeError("Mark price unavailable")

        result = state_collector.resolve_effective_price("BTC-PERP", "buy", None, sample_product)

        # Should fall back to mid-price
        expected_mid = (Decimal("50000.0") + Decimal("50010.0")) / Decimal("2")
        assert result == expected_mid

    def test_resolve_effective_price_uses_mid_price(self, state_collector, sample_product) -> None:
        """Test price resolution uses mid-price from product."""
        state_collector.broker.get_mark_price.side_effect = RuntimeError("Mark price unavailable")

        result = state_collector.resolve_effective_price("BTC-PERP", "buy", None, sample_product)

        expected_mid = (Decimal("50000.0") + Decimal("50010.0")) / Decimal("2")
        assert result == expected_mid

    def test_resolve_effective_price_mid_price_incomplete_fallback(self, state_collector) -> None:
        """Test mid-price fallback when bid/ask are incomplete."""
        incomplete_product = MagicMock()
        incomplete_product.bid_price = Decimal("50000.0")
        incomplete_product.ask_price = None  # Missing ask
        incomplete_product.price = Decimal("50005.0")
        incomplete_product.quote_increment = Decimal("0.1")

        result = state_collector.resolve_effective_price("BTC-PERP", "buy", None, incomplete_product)

        assert result == Decimal("50005.0")  # Should use product price

    def test_resolve_effective_price_uses_quote_fallback(self, state_collector, sample_product) -> None:
        """Test price resolution uses broker quote as fallback."""
        # Make mark price and mid-price unavailable
        state_collector.broker.get_mark_price.side_effect = RuntimeError("Mark price unavailable")
        sample_product.bid_price = None
        sample_product.ask_price = None

        # Configure quote fallback
        quote_mock = MagicMock()
        quote_mock.last = Decimal("50200.0")
        state_collector.broker.get_quote.return_value = quote_mock

        result = state_collector.resolve_effective_price("BTC-PERP", "buy", None, sample_product)

        assert result == Decimal("50200.0")
        state_collector.broker.get_quote.assert_called_once_with("BTC-PERP")

    def test_resolve_effective_price_quote_fallback_errors(self, state_collector, sample_product) -> None:
        """Test quote fallback when quote service errors."""
        # Make mark price and mid-price unavailable
        state_collector.broker.get_mark_price.side_effect = RuntimeError("Mark price unavailable")
        sample_product.bid_price = None
        sample_product.ask_price = None
        state_collector.broker.get_quote.side_effect = RuntimeError("Quote unavailable")

        result = state_collector.resolve_effective_price("BTC-PERP", "buy", None, sample_product)

        # Should use product price
        assert result == Decimal("50005.0")

    def test_resolve_effective_price_uses_product_price(self, state_collector) -> None:
        """Test price resolution uses product price as final fallback."""
        product_with_price = MagicMock()
        product_with_price.bid_price = None  # Set explicitly to None
        product_with_price.ask_price = None  # Set explicitly to None
        product_with_price.price = Decimal("50300.0")
        product_with_price.quote_increment = Decimal("0.1")

        # Make all other methods unavailable
        state_collector.broker.get_mark_price.side_effect = RuntimeError("Mark price unavailable")
        state_collector.broker.get_quote.side_effect = RuntimeError("Quote unavailable")

        result = state_collector.resolve_effective_price("BTC-PERP", "buy", None, product_with_price)

        assert result == Decimal("50300.0")

    def test_resolve_effective_price_final_default_fallback(self, state_collector) -> None:
        """Test final default price fallback."""
        minimal_product = MagicMock()
        minimal_product.bid_price = None  # Set explicitly to None
        minimal_product.ask_price = None  # Set explicitly to None
        minimal_product.price = None
        minimal_product.quote_increment = Decimal("0.01")

        # Make all other methods unavailable
        state_collector.broker.get_mark_price.side_effect = RuntimeError("Mark price unavailable")
        state_collector.broker.get_quote.side_effect = RuntimeError("Quote unavailable")

        result = state_collector.resolve_effective_price("BTC-PERP", "buy", None, minimal_product)

        # Should use quote_increment * 100
        assert result == Decimal("1.0")  # 0.01 * 100

    def test_resolve_effective_price_final_default_no_quote_increment(self, state_collector) -> None:
        """Test final default when no quote_increment available."""
        minimal_product = MagicMock()
        minimal_product.bid_price = None  # Set explicitly to None
        minimal_product.ask_price = None  # Set explicitly to None
        minimal_product.price = None
        minimal_product.quote_increment = None

        # Make all other methods unavailable
        state_collector.broker.get_mark_price.side_effect = RuntimeError("Mark price unavailable")
        state_collector.broker.get_quote.side_effect = RuntimeError("Quote unavailable")

        result = state_collector.resolve_effective_price("BTC-PERP", "buy", None, minimal_product)

        # Should use default 0.01 * 100
        assert result == Decimal("1.0")

    def test_resolve_effective_price_all_methods_unavailable(self, broker_with_missing_methods) -> None:
        """Test price resolution when broker has no optional methods."""
        collector = StateCollector(broker_with_missing_methods)
        minimal_product = MagicMock()
        minimal_product.bid_price = None  # Set explicitly to None
        minimal_product.ask_price = None  # Set explicitly to None
        minimal_product.price = None
        minimal_product.quote_increment = None

        result = collector.resolve_effective_price("BTC-PERP", "buy", None, minimal_product)

        # Should use default fallback
        assert result == Decimal("1.0")


class TestRequireProduct:
    """Test product validation and resolution."""

    def test_require_product_with_valid_product(self, state_collector, sample_product) -> None:
        """Test require_product with valid product provided."""
        result = state_collector.require_product("BTC-PERP", sample_product)

        assert result == sample_product
        state_collector.broker.get_product.assert_not_called()

    def test_require_product_fetches_from_broker(self, state_collector) -> None:
        """Test require_product fetches from broker when product is None."""
        mock_product = MagicMock()
        mock_product.symbol = "BTC-PERP"
        state_collector.broker.get_product.return_value = mock_product

        result = state_collector.require_product("BTC-PERP", None)

        assert result == mock_product
        state_collector.broker.get_product.assert_called_once_with("BTC-PERP")

    def test_require_product_raises_when_not_found(self, state_collector) -> None:
        """Test require_product raises ValidationError when product not found."""
        state_collector.broker.get_product.return_value = None

        with pytest.raises(ValidationError, match="Product not found: BTC-PERP"):
            state_collector.require_product("BTC-PERP", None)

    def test_require_product_broker_error_propagates(self, broker_with_errors) -> None:
        """Test require_product propagates broker errors."""
        collector = StateCollector(broker_with_errors)

        with pytest.raises(ValidationError, match="Product not found: BTC-PERP"):
            collector.require_product("BTC-PERP", None)

    def test_require_product_with_none_product_and_successful_fetch(self, state_collector) -> None:
        """Test require_product when product is None but fetch succeeds."""
        fetched_product = MagicMock()
        fetched_product.symbol = "ETH-PERP"
        fetched_product.price = Decimal("3000.0")
        state_collector.broker.get_product.return_value = fetched_product

        result = state_collector.require_product("ETH-PERP", None)

        assert result == fetched_product
        assert result.symbol == "ETH-PERP"
        state_collector.broker.get_product.assert_called_once_with("ETH-PERP")