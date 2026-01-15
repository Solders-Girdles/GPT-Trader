"""Tests for Coinbase REST service base functionality."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from gpt_trader.core import (
    InsufficientFunds,
    InvalidRequestError,
    NotFoundError,
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
)
from gpt_trader.errors import ValidationError
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.models import APIConfig, Product
from gpt_trader.features.brokerages.coinbase.rest.base import CoinbaseRestServiceCore
from gpt_trader.features.brokerages.coinbase.rest.position_state_store import PositionStateStore
from gpt_trader.features.brokerages.coinbase.utilities import ProductCatalog
from gpt_trader.persistence.event_store import EventStore


class TestCoinbaseRestServiceCore:
    """Test CoinbaseRestServiceCore class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.client = Mock(spec=CoinbaseClient)
        self.endpoints = Mock(spec=CoinbaseEndpoints)
        self.config = Mock(spec=APIConfig)
        self.product_catalog = Mock(spec=ProductCatalog)
        self.market_data = Mock(spec=MarketDataService)
        self.event_store = Mock(spec=EventStore)

        self.position_store = PositionStateStore()
        self.service = CoinbaseRestServiceCore(
            client=self.client,
            endpoints=self.endpoints,
            config=self.config,
            product_catalog=self.product_catalog,
            market_data=self.market_data,
            event_store=self.event_store,
            position_store=self.position_store,
        )

        # Mock product
        self.mock_product = Mock(spec=Product)
        self.mock_product.product_id = "BTC-USD"
        self.mock_product.step_size = Decimal("0.00000001")
        self.mock_product.price_increment = Decimal("0.01")
        self.mock_product.min_size = Decimal("0.001")
        self.mock_product.min_notional = Decimal("10")

    def test_service_init(self) -> None:
        """Test service initialization."""
        assert self.service.client == self.client
        assert self.service.endpoints == self.endpoints
        assert self.service.config == self.config
        assert self.service.product_catalog == self.product_catalog
        assert self.service.market_data == self.market_data
        assert self.service._event_store == self.event_store
        assert isinstance(self.service.positions, dict)
        assert len(self.service.positions) == 0

    def test_build_order_payload_limit_order(self) -> None:
        """Test building limit order payload."""
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=None,
            tif=TimeInForce.GTC,
            client_id="test_client_123",
            reduce_only=False,
            leverage=None,
        )

        assert payload["product_id"] == "BTC-USD"
        assert payload["side"] == "BUY"
        assert "order_configuration" in payload
        assert "limit_limit_gtc" in payload["order_configuration"]
        assert payload["order_configuration"]["limit_limit_gtc"]["base_size"] == "0.1"
        assert payload["order_configuration"]["limit_limit_gtc"]["limit_price"] == "50000.00"
        assert payload["client_order_id"] == "test_client_123"

    def test_build_order_payload_market_order(self) -> None:
        """Test building market order payload."""
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=None,
            stop_price=None,
            tif=TimeInForce.IOC,
            client_id=None,
            reduce_only=False,
            leverage=None,
        )

        assert payload["product_id"] == "BTC-USD"
        assert payload["side"] == "SELL"
        assert "order_configuration" in payload
        assert "market_market_ioc" in payload["order_configuration"]
        assert payload["order_configuration"]["market_market_ioc"]["base_size"] == "0.1"
        assert "client_order_id" in payload
        assert payload["client_order_id"].startswith("perps_")

    def test_build_order_payload_stop_limit(self) -> None:
        """Test building stop limit order payload."""
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.STOP_LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=Decimal("49000.00"),
            tif=TimeInForce.GTC,
            client_id=None,
            reduce_only=False,
            leverage=None,
        )

        assert payload["product_id"] == "BTC-USD"
        assert "order_configuration" in payload
        assert "stop_limit_stop_limit_gtc" in payload["order_configuration"]
        config = payload["order_configuration"]["stop_limit_stop_limit_gtc"]
        assert config["base_size"] == "0.1"
        assert config["limit_price"] == "50000.00"
        assert config["stop_price"] == "49000.00"

    def test_build_order_payload_with_reduce_only(self) -> None:
        """Test building order payload with reduce_only flag."""
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=None,
            tif=TimeInForce.GTC,
            client_id=None,
            reduce_only=True,
            leverage=None,
        )

        assert payload["reduce_only"] is True

    def test_build_order_payload_with_leverage(self) -> None:
        """Test building order payload with leverage."""
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=None,
            tif=TimeInForce.GTC,
            client_id=None,
            reduce_only=False,
            leverage=5,
        )

        assert payload["leverage"] == 5

    def test_build_order_payload_post_only(self) -> None:
        """Test building order payload with post_only flag."""
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=None,
            tif=TimeInForce.GTC,
            client_id=None,
            reduce_only=False,
            leverage=None,
            post_only=True,
        )

        assert payload["post_only"] is True

    def test_build_order_payload_exclude_client_id(self) -> None:
        """Test building order payload without client ID."""
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=None,
            tif=TimeInForce.GTC,
            client_id=None,
            reduce_only=False,
            leverage=None,
            include_client_id=False,
        )

        assert "client_order_id" not in payload

    def test_build_order_payload_quantize_values(self) -> None:
        """Test that values are properly quantized."""
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.123456789"),  # Will be quantized to 8 decimal places
            price=Decimal("50000.123456"),  # Will be quantized to 2 decimal places
            stop_price=None,
            tif=TimeInForce.GTC,
            client_id=None,
            reduce_only=False,
            leverage=None,
        )

        config = payload["order_configuration"]["limit_limit_gtc"]
        assert config["base_size"] == "0.12345678"  # Rounded down to step_size
        assert config["limit_price"] == "50000.12"  # Rounded to price_increment

    def test_build_order_payload_quantity_below_min_size(self) -> None:
        """Test building order payload with quantity below minimum size."""
        self.product_catalog.get.return_value = self.mock_product

        with pytest.raises(InvalidRequestError) as exc:
            self.service._build_order_payload(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.0001"),  # Below min_size of 0.001
                price=Decimal("50000.00"),
                stop_price=None,
                tif=TimeInForce.GTC,
                client_id=None,
                reduce_only=False,
                leverage=None,
            )
        assert "Order quantity 0.00010000 is below minimum size 0.001" in str(exc.value)

    def test_build_order_payload_notional_below_min_notional(self) -> None:
        """Test building order payload with notional below minimum."""
        self.product_catalog.get.return_value = self.mock_product
        # Mock quote price
        mock_quote = Mock()
        mock_quote.last = Decimal("50.00")  # Low price makes notional too small
        self.service.get_rest_quote = Mock(return_value=mock_quote)

        with pytest.raises(InvalidRequestError) as exc:
            self.service._build_order_payload(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),  # Min size
                price=Decimal("50.00"),  # Low price
                stop_price=None,
                tif=TimeInForce.GTC,
                client_id=None,
                reduce_only=False,
                leverage=None,
            )
        assert "Order notional 0.0500000000 is below minimum 10" in str(exc.value)

    def test_build_order_payload_limit_order_requires_price(self) -> None:
        """Test that limit orders require a price."""
        self.product_catalog.get.return_value = self.mock_product

        with pytest.raises(ValidationError, match="price is required for limit orders"):
            self.service._build_order_payload(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
                price=None,
                stop_price=None,
                tif=TimeInForce.GTC,
                client_id=None,
                reduce_only=False,
                leverage=None,
            )

    def test_build_order_payload_product_not_found(self) -> None:
        """Test building order payload when product is not found."""
        self.product_catalog.get.side_effect = NotFoundError("Product not found")
        self.service.get_product = Mock(return_value=None)

        with pytest.raises(NotFoundError):
            self.service._build_order_payload(
                symbol="UNKNOWN-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
                price=Decimal("50000.00"),
                stop_price=None,
                tif=TimeInForce.GTC,
                client_id=None,
                reduce_only=False,
                leverage=None,
            )

    def test_build_order_payload_enum_coercion(self) -> None:
        """Test enum coercion for order parameters."""
        self.product_catalog.get.return_value = self.mock_product

        # Test string values
        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side="BUY",  # String instead of enum
            order_type="LIMIT",  # String instead of enum
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=None,
            tif="GTC",  # String instead of enum
            client_id=None,
            reduce_only=False,
            leverage=None,
        )

        assert payload["side"] == "BUY"
        assert payload["type"] == "LIMIT"
        assert payload["time_in_force"] == "GTC"

    def test_build_order_payload_invalid_side_passthrough(self) -> None:
        """Test payload preserves side string when enum coercion fails."""
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side="LONG",
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=None,
            stop_price=None,
            tif=TimeInForce.GTC,
            client_id=None,
            reduce_only=False,
            leverage=None,
        )

        assert payload["side"] == "LONG"

    def test_build_order_payload_gtd_conversion(self) -> None:
        """Test GTD time in force conversion to GTC."""
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=None,
            tif="GTD",  # Should be converted to GTC
            client_id=None,
            reduce_only=False,
            leverage=None,
        )

        assert payload["time_in_force"] == "GTC"

    def test_build_order_payload_invalid_tif_passthrough(self) -> None:
        """Test payload preserves time-in-force string when enum coercion fails."""
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=None,
            stop_price=None,
            tif="DAY",
            client_id=None,
            reduce_only=False,
            leverage=None,
        )

        assert payload["time_in_force"] == "DAY"

    def test_execute_order_payload_success(self) -> None:
        """Test successful order execution."""
        payload = {"product_id": "BTC-USD", "side": "BUY"}
        mock_order = Mock(spec=Order)
        mock_order.id = "order_123"
        self.client.place_order.return_value = {"order_id": "order_123"}

        with patch(
            "gpt_trader.features.brokerages.coinbase.rest.base.to_order", return_value=mock_order
        ):
            result = self.service._execute_order_payload("BTC-USD", payload, "client_123")

        assert result == mock_order
        self.client.place_order.assert_called_once_with(payload)

    def test_execute_order_payload_with_preview(self) -> None:
        """Test order execution with preview enabled."""
        payload = {"product_id": "BTC-USD", "side": "BUY"}
        mock_order = Mock(spec=Order)
        self.client.preview_order.return_value = {"success": True}
        self.client.place_order.return_value = {"order_id": "order_123"}

        # Enable order preview via bot_config (previously was os.environ["ORDER_PREVIEW_ENABLED"])
        mock_bot_config = Mock()
        mock_bot_config.enable_order_preview = True
        self.service.bot_config = mock_bot_config

        with patch(
            "gpt_trader.features.brokerages.coinbase.rest.base.to_order", return_value=mock_order
        ):
            result = self.service._execute_order_payload("BTC-USD", payload, "client_123")

        assert result == mock_order
        self.client.preview_order.assert_called_once_with(payload)
        self.client.place_order.assert_called_once_with(payload)

    def test_execute_order_payload_preview_failure_still_places(self) -> None:
        """Test order execution continues when preview fails."""
        payload = {"product_id": "BTC-USD", "side": "BUY"}
        mock_order = Mock(spec=Order)
        self.client.preview_order.side_effect = Exception("preview failed")
        self.client.place_order.return_value = {"order_id": "order_123"}

        mock_bot_config = Mock()
        mock_bot_config.enable_order_preview = True
        self.service.bot_config = mock_bot_config

        with patch(
            "gpt_trader.features.brokerages.coinbase.rest.base.to_order", return_value=mock_order
        ):
            result = self.service._execute_order_payload("BTC-USD", payload, "client_123")

        assert result == mock_order
        self.client.preview_order.assert_called_once_with(payload)
        self.client.place_order.assert_called_once_with(payload)

    def test_execute_order_payload_insufficient_funds(self) -> None:
        """Test order execution with insufficient funds."""
        payload = {"product_id": "BTC-USD", "side": "BUY"}
        self.client.place_order.side_effect = InsufficientFunds("Insufficient balance")

        with pytest.raises(InsufficientFunds):
            self.service._execute_order_payload("BTC-USD", payload, "client_123")

    def test_execute_order_payload_validation_error(self) -> None:
        """Test order execution with validation error."""
        payload = {"product_id": "BTC-USD", "side": "BUY"}
        self.client.place_order.side_effect = ValidationError("Invalid order")

        with pytest.raises(ValidationError):
            self.service._execute_order_payload("BTC-USD", payload, "client_123")

    def test_execute_order_payload_duplicate_client_id(self) -> None:
        """Test order execution with duplicate client ID."""
        payload = {"product_id": "BTC-USD", "side": "BUY"}
        mock_order = Mock(spec=Order)
        mock_order.id = "existing_order_123"

        # First call fails with duplicate error, second call finds existing order
        self.client.place_order.side_effect = InvalidRequestError("duplicate client_order_id")
        self.service._find_existing_order_by_client_id = Mock(return_value=mock_order)

        result = self.service._execute_order_payload("BTC-USD", payload, "client_123")

        assert result == mock_order
        self.service._find_existing_order_by_client_id.assert_called_once_with(
            "BTC-USD", "client_123"
        )

    def test_execute_order_payload_duplicate_client_id_retry_failure(self) -> None:
        """Test duplicate client id retry failure re-raises original error."""
        payload = {"product_id": "BTC-USD", "side": "BUY"}
        error = InvalidRequestError("duplicate client_order_id")
        self.client.place_order.side_effect = [error, Exception("retry failed")]
        self.service._find_existing_order_by_client_id = Mock(return_value=None)

        with pytest.raises(InvalidRequestError) as exc:
            self.service._execute_order_payload("BTC-USD", payload, "client_123")

        assert "duplicate client_order_id" in str(exc.value)
        assert self.client.place_order.call_count == 2
        self.service._find_existing_order_by_client_id.assert_called_once_with(
            "BTC-USD", "client_123"
        )

    def test_execute_order_payload_network_error(self) -> None:
        """Test order execution with network errors."""
        payload = {"product_id": "BTC-USD", "side": "BUY"}
        self.client.place_order.side_effect = ConnectionError("network down")

        with pytest.raises(ConnectionError):
            self.service._execute_order_payload("BTC-USD", payload, "client_123")

    def test_execute_order_payload_unexpected_error(self) -> None:
        """Test order execution with unexpected error."""
        payload = {"product_id": "BTC-USD", "side": "BUY"}
        self.client.place_order.side_effect = Exception("Unexpected error")

        with pytest.raises(Exception, match="Unexpected error"):
            self.service._execute_order_payload("BTC-USD", payload, "client_123")

    def test_find_existing_order_by_client_id_success(self) -> None:
        """Test finding existing order by client ID."""
        mock_order_data = {
            "order_id": "order_123",
            "client_order_id": "client_123",
            "created_at": "2024-01-01T00:00:00Z",
        }
        self.client.list_orders.return_value = {"orders": [mock_order_data]}
        mock_order = Mock(spec=Order)
        mock_order.id = "order_123"
        mock_order.client_id = "client_123"
        mock_order.created_at = datetime(2024, 1, 1)

        with patch(
            "gpt_trader.features.brokerages.coinbase.rest.base.to_order", return_value=mock_order
        ):
            result = self.service._find_existing_order_by_client_id("BTC-USD", "client_123")

        assert result == mock_order
        self.client.list_orders.assert_called_once_with(product_id="BTC-USD")

    def test_find_existing_order_by_client_id_no_client_id(self) -> None:
        """Test finding existing order with no client ID."""
        result = self.service._find_existing_order_by_client_id("BTC-USD", "")

        assert result is None
        self.client.list_orders.assert_not_called()

    def test_find_existing_order_by_client_id_no_matches(self) -> None:
        """Test finding existing order with no matches."""
        self.client.list_orders.return_value = {"orders": []}

        result = self.service._find_existing_order_by_client_id("BTC-USD", "client_123")

        assert result is None

    def test_find_existing_order_by_client_id_multiple_matches(self) -> None:
        """Test finding existing order with multiple matches (returns newest)."""
        order1 = {
            "order_id": "order_1",
            "client_order_id": "client_123",
            "created_at": "2024-01-01T00:00:00Z",
        }
        order2 = {
            "order_id": "order_2",
            "client_order_id": "client_123",
            "created_at": "2024-01-01T01:00:00Z",  # Newer
        }
        self.client.list_orders.return_value = {"orders": [order1, order2]}
        mock_order = Mock(spec=Order)
        mock_order.id = "order_2"
        mock_order.client_id = "client_123"
        mock_order.created_at = datetime(2024, 1, 1, 1)

        with patch(
            "gpt_trader.features.brokerages.coinbase.rest.base.to_order", return_value=mock_order
        ):
            result = self.service._find_existing_order_by_client_id("BTC-USD", "client_123")

        assert result == mock_order

    def test_find_existing_order_by_client_id_api_error(self) -> None:
        """Test finding existing order when API fails."""
        self.client.list_orders.side_effect = Exception("API error")

        result = self.service._find_existing_order_by_client_id("BTC-USD", "client_123")

        assert result is None

    def test_find_existing_order_by_client_id_network_error(self) -> None:
        """Test finding existing order when network errors occur."""
        self.client.list_orders.side_effect = ConnectionError("network down")

        result = self.service._find_existing_order_by_client_id("BTC-USD", "client_123")

        assert result is None

    def test_find_existing_order_by_client_id_value_error(self) -> None:
        """Test finding existing order when payload parsing fails."""
        self.client.list_orders.side_effect = ValueError("bad payload")

        result = self.service._find_existing_order_by_client_id("BTC-USD", "client_123")

        assert result is None

    def test_update_position_metrics_no_position(self) -> None:
        """Test updating position metrics when no position exists."""
        self.service.update_position_metrics("BTC-USD")

        # Should not raise any errors
        self.market_data.get_mark.assert_not_called()

    def test_update_position_metrics_no_mark_price(self) -> None:
        """Test updating position metrics when no mark price is available."""
        from gpt_trader.features.brokerages.coinbase.utilities import PositionState

        position = PositionState(
            symbol="BTC-USD", side="LONG", quantity=Decimal("0.1"), entry_price=Decimal("50000")
        )
        self.position_store.set("BTC-USD", position)
        self.market_data.get_mark.return_value = None

        self.service.update_position_metrics("BTC-USD")

        # Should not raise any errors
        self.event_store.append_position.assert_not_called()

    def test_update_position_metrics_missing_position_entry(self) -> None:
        """Test updating metrics when store contains missing position entry."""
        from gpt_trader.features.brokerages.coinbase.utilities import PositionState

        position = PositionState(
            symbol="BTC-USD", side="LONG", quantity=Decimal("0.1"), entry_price=Decimal("50000")
        )
        self.position_store.set("BTC-USD", position)

        with patch.object(self.position_store, "get", return_value=None):
            self.service.update_position_metrics("BTC-USD")

        self.market_data.get_mark.assert_not_called()

    def test_update_position_metrics_success(self) -> None:
        """Test successful position metrics update."""
        from gpt_trader.features.brokerages.coinbase.utilities import PositionState

        position = PositionState(
            symbol="BTC-USD", side="LONG", quantity=Decimal("0.1"), entry_price=Decimal("50000")
        )
        self.position_store.set("BTC-USD", position)
        self.market_data.get_mark.return_value = Decimal("51000")
        self.product_catalog.get_funding.return_value = (
            Decimal("0.01"),
            datetime(2024, 1, 1, 12, 0, 0),
        )

        self.service.update_position_metrics("BTC-USD")

        # Should call event store methods
        self.event_store.append_metric.assert_called()
        self.event_store.append_position.assert_called()

        # Check the position update call
        position_call = self.event_store.append_position.call_args
        assert position_call[1]["bot_id"] == "coinbase_perps"
        assert position_call[1]["position"]["symbol"] == "BTC-USD"
        assert position_call[1]["position"]["mark_price"] == "51000"

    def test_update_position_metrics_with_funding(self) -> None:
        """Test position metrics update with funding accrual."""
        from gpt_trader.features.brokerages.coinbase.utilities import PositionState

        position = PositionState(
            symbol="BTC-USD", side="LONG", quantity=Decimal("0.1"), entry_price=Decimal("50000")
        )
        self.position_store.set("BTC-USD", position)
        self.market_data.get_mark.return_value = Decimal("51000")
        self.product_catalog.get_funding.return_value = (
            Decimal("0.01"),
            datetime(2024, 1, 1, 12, 0, 0),
        )

        # Mock funding calculator to return a non-zero delta
        with patch.object(
            self.service._funding_calculator, "accrue_if_due", return_value=Decimal("5.0")
        ):
            self.service.update_position_metrics("BTC-USD")

        # Should have recorded funding metric
        # First call should be funding, second call is position state
        assert self.event_store.append_metric.call_count >= 1
        funding_call = self.event_store.append_metric.call_args_list[0]
        assert funding_call.kwargs["metrics"]["event_type"] == "funding_accrual"
        assert funding_call.kwargs["metrics"]["funding_amount"] == "5.0"

        # Position should have updated realized PnL
        assert position.realized_pnl == Decimal("5.0")

    def test_update_position_metrics_skips_funding_when_none(self) -> None:
        """Test position metrics update when funding rate is unavailable."""
        from gpt_trader.features.brokerages.coinbase.utilities import PositionState

        position = PositionState(
            symbol="BTC-USD", side="LONG", quantity=Decimal("0.1"), entry_price=Decimal("50000")
        )
        self.position_store.set("BTC-USD", position)
        self.market_data.get_mark.return_value = Decimal("51000")
        self.product_catalog.get_funding.return_value = (None, None)

        self.service.update_position_metrics("BTC-USD")

        self.event_store.append_metric.assert_called_once()
        self.event_store.append_position.assert_called_once()

    def test_positions_property(self) -> None:
        """Test positions property access."""
        from gpt_trader.features.brokerages.coinbase.utilities import PositionState

        position = PositionState(
            symbol="BTC-USD", side="LONG", quantity=Decimal("0.1"), entry_price=Decimal("50000")
        )
        self.position_store.set("BTC-USD", position)

        positions = self.service.positions

        assert isinstance(positions, dict)
        assert "BTC-USD" in positions
        assert positions["BTC-USD"] == position

    def test_update_position_metrics_public_method(self) -> None:
        """Test public update_position_metrics method."""
        from gpt_trader.features.brokerages.coinbase.utilities import PositionState

        position = PositionState(
            symbol="BTC-USD", side="LONG", quantity=Decimal("0.1"), entry_price=Decimal("50000")
        )
        self.position_store.set("BTC-USD", position)
        self.market_data.get_mark.return_value = Decimal("51000")
        self.product_catalog.get_funding.return_value = (Decimal("0"), None)

        self.service.update_position_metrics("BTC-USD")

        self.market_data.get_mark.assert_called_once_with("BTC-USD")

    def test_build_order_payload_ioc_fok_limit_orders(self) -> None:
        """Test building limit orders with IOC and FOK time in force."""
        self.product_catalog.get.return_value = self.mock_product

        # Test IOC
        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=None,
            tif=TimeInForce.IOC,
            client_id=None,
            reduce_only=False,
            leverage=None,
        )

        assert "limit_limit_ioc" in payload["order_configuration"]

        # Test FOK
        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=None,
            tif=TimeInForce.FOK,
            client_id=None,
            reduce_only=False,
            leverage=None,
        )

        assert "limit_limit_fok" in payload["order_configuration"]

    def test_build_order_payload_fallback_configuration(self) -> None:
        """Test building order payload with fallback configuration for unsupported types."""
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.STOP,  # Not in special handling
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=Decimal("49000.00"),
            tif=TimeInForce.GTC,
            client_id=None,
            reduce_only=False,
            leverage=None,
        )

        # Should use fallback configuration
        assert payload["type"] == "STOP"
        assert payload["size"] == "0.1"
        assert payload["time_in_force"] == "GTC"
        assert payload["price"] == "50000.00"
        assert payload["stop_price"] == "49000.00"
