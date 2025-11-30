"""Contract tests for Coinbase REST service components."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from gpt_trader.errors import ValidationError
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
from gpt_trader.features.brokerages.coinbase.errors import (
    OrderCancellationError,
    OrderQueryError,
)
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.models import APIConfig, Product
from gpt_trader.features.brokerages.coinbase.rest.base import CoinbaseRestServiceCore
from gpt_trader.features.brokerages.coinbase.rest.order_service import OrderService
from gpt_trader.features.brokerages.coinbase.rest.pnl_service import PnLService
from gpt_trader.features.brokerages.coinbase.rest.portfolio_service import PortfolioService
from gpt_trader.features.brokerages.coinbase.rest.position_state_store import PositionStateStore
from gpt_trader.features.brokerages.coinbase.utilities import PositionState
from gpt_trader.features.brokerages.core.interfaces import (
    Balance,
    InsufficientFunds,
    InvalidRequestError,
    Order,
    OrderSide,
    OrderType,
    Position,
)
from gpt_trader.persistence.event_store import EventStore


class TestCoinbaseRestContractSuite:
    """Contract tests for Coinbase REST service components."""

    @pytest.fixture
    def mock_client(self) -> Mock:
        return Mock(spec=CoinbaseClient)

    @pytest.fixture
    def mock_endpoints(self) -> Mock:
        return Mock(spec=CoinbaseEndpoints)

    @pytest.fixture
    def mock_config(self) -> Mock:
        return Mock(spec=APIConfig)

    @pytest.fixture
    def mock_product_catalog(self) -> Mock:
        return Mock()

    @pytest.fixture
    def mock_market_data(self) -> Mock:
        return Mock(spec=MarketDataService)

    @pytest.fixture
    def mock_event_store(self) -> Mock:
        return Mock(spec=EventStore)

    @pytest.fixture
    def mock_product(self) -> Mock:
        product = Mock(spec=Product)
        product.product_id = "BTC-USD"
        product.step_size = Decimal("0.00000001")
        product.price_increment = Decimal("0.01")
        product.min_size = Decimal("0.001")
        product.min_notional = Decimal("10")
        return product

    @pytest.fixture
    def position_store(self) -> PositionStateStore:
        return PositionStateStore()

    @pytest.fixture
    def service_core(
        self,
        mock_client,
        mock_endpoints,
        mock_config,
        mock_product_catalog,
        mock_market_data,
        mock_event_store,
        position_store,
    ) -> CoinbaseRestServiceCore:
        return CoinbaseRestServiceCore(
            client=mock_client,
            endpoints=mock_endpoints,
            config=mock_config,
            product_catalog=mock_product_catalog,
            market_data=mock_market_data,
            event_store=mock_event_store,
            position_store=position_store,
        )

    @pytest.fixture
    def portfolio_service(self, mock_client, mock_endpoints, mock_event_store) -> PortfolioService:
        """Create a PortfolioService."""
        return PortfolioService(
            client=mock_client,
            endpoints=mock_endpoints,
            event_store=mock_event_store,
        )

    @pytest.fixture
    def order_service(self, mock_client, service_core, portfolio_service) -> OrderService:
        """Create an OrderService with Core and Portfolio services as collaborators."""
        return OrderService(
            client=mock_client,
            payload_builder=service_core,
            payload_executor=service_core,
            position_provider=portfolio_service,
        )

    @pytest.fixture
    def pnl_service(self, position_store, mock_market_data) -> PnLService:
        """Create a PnLService."""
        return PnLService(
            position_store=position_store,
            market_data=mock_market_data,
        )

    # ------------------------------------------------------------------
    # OrderService Contract Tests
    # ------------------------------------------------------------------

    def test_place_order_quantity_resolution_success(
        self, order_service, service_core, mock_product_catalog, mock_product, mock_client
    ):
        """Test successful order placement with quantity resolution."""
        mock_product_catalog.get.return_value = mock_product
        mock_client.place_order.return_value = {"order_id": "test_123"}

        # Spy on execute_order_payload on the core service
        with patch.object(
            service_core, "execute_order_payload", side_effect=service_core.execute_order_payload
        ) as mock_execute:
            # Mock to_order to return a valid Order
            with patch(
                "gpt_trader.features.brokerages.coinbase.rest.base.to_order",
                return_value=Mock(spec=Order),
            ):
                order = order_service.place_order(
                    symbol="BTC-USD",
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=Decimal("0.123456789"),
                    price=Decimal("50000.00"),
                )

            assert order is not None
            # Verify quantity was quantized to step_size
            call_args = mock_execute.call_args
            payload = call_args[0][1]  # Second argument is payload
            assert payload["order_configuration"]["limit_limit_gtc"]["base_size"] == "0.12345678"

    def test_place_order_quantity_resolution_error_branch(
        self, order_service, mock_product_catalog, mock_product
    ):
        """Test order placement with quantity below minimum."""
        mock_product_catalog.get.return_value = mock_product

        # Quantity < min_size
        with pytest.raises(InvalidRequestError, match="quantity .* is below minimum size"):
            order_service.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.0001"),  # Below min_size
                price=Decimal("50000.00"),
            )

    def test_place_order_error_branch_insufficient_funds(
        self, order_service, mock_product_catalog, mock_product, mock_client
    ):
        """Test order placement with insufficient funds error."""
        mock_product_catalog.get.return_value = mock_product
        mock_client.place_order.side_effect = InsufficientFunds("Insufficient balance")

        with pytest.raises(InsufficientFunds):
            order_service.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
                price=Decimal("50000.00"),
            )

    def test_place_order_error_branch_validation_error(
        self, order_service, mock_product_catalog, mock_product, mock_client
    ):
        """Test order placement with validation error."""
        mock_product_catalog.get.return_value = mock_product
        mock_client.place_order.side_effect = ValidationError("Invalid order parameters")

        with pytest.raises(ValidationError):
            order_service.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
                price=Decimal("50000.00"),
            )

    def test_cancel_order_success(self, order_service, mock_client):
        """Test successful order cancellation."""
        mock_client.cancel_orders.return_value = {
            "results": [{"order_id": "test_123", "success": True}]
        }

        result = order_service.cancel_order("test_123")

        assert result is True

    def test_cancel_order_failure(self, order_service, mock_client):
        """Test order cancellation failure raises OrderCancellationError."""
        mock_client.cancel_orders.return_value = {
            "results": [{"order_id": "test_123", "success": False}]
        }

        with pytest.raises(OrderCancellationError, match="Cancellation rejected"):
            order_service.cancel_order("test_123")

    def test_list_orders_with_pagination(self, order_service, mock_client):
        """Test order listing with pagination handling."""
        # Mock paginated responses
        mock_client.list_orders.side_effect = [
            {"orders": [{"order_id": "1"}, {"order_id": "2"}], "cursor": "next_page"},
            {"orders": [{"order_id": "3"}]},  # No cursor = last page
        ]

        with patch(
            "gpt_trader.features.brokerages.coinbase.rest.order_service.to_order"
        ) as mock_to_order:
            mock_to_order.return_value = Mock(spec=Order)
            orders = order_service.list_orders()

        assert len(orders) == 3
        # Verify pagination was handled (called twice)
        assert mock_client.list_orders.call_count == 2

    def test_list_orders_error_handling(self, order_service, mock_client):
        """Test order listing raises OrderQueryError on error."""
        mock_client.list_orders.side_effect = Exception("API error")

        with pytest.raises(OrderQueryError, match="Failed to list orders"):
            order_service.list_orders()

    def test_get_order_success(self, order_service, mock_client):
        """Test successful order retrieval."""
        mock_order_data = {"order_id": "test_123", "status": "filled"}
        mock_client.get_order_historical.return_value = {"order": mock_order_data}

        with patch(
            "gpt_trader.features.brokerages.coinbase.rest.order_service.to_order"
        ) as mock_to_order:
            mock_to_order.return_value = Mock(spec=Order)
            order = order_service.get_order("test_123")

        assert order is not None

    def test_get_order_not_found(self, order_service, mock_client):
        """Test order retrieval raises OrderQueryError when exception occurs."""
        mock_client.get_order_historical.side_effect = Exception("Order not found")

        with pytest.raises(OrderQueryError, match="Failed to get order"):
            order_service.get_order("test_123")

    def test_list_fills_with_pagination(self, order_service, mock_client):
        """Test fills listing with pagination."""
        mock_client.list_fills.side_effect = [
            {"fills": [{"fill_id": "1"}, {"fill_id": "2"}], "cursor": "next"},
            {"fills": [{"fill_id": "3"}]},
        ]

        fills = order_service.list_fills()

        assert len(fills) == 3

    def test_list_fills_error_handling(self, order_service, mock_client):
        """Test fills listing raises OrderQueryError on error."""
        mock_client.list_fills.side_effect = Exception("API error")

        with pytest.raises(OrderQueryError, match="Failed to list fills"):
            order_service.list_fills()

    def test_close_position_success(self, portfolio_service, order_service, mock_client):
        """Test successful position closing."""
        # Mock positions on the portfolio service (which acts as position provider)
        mock_position = Mock(spec=Position)
        mock_position.symbol = "BTC-USD"
        mock_position.quantity = Decimal("1.0")
        # We need to patch list_positions on the portfolio_service instance
        with patch.object(portfolio_service, "list_positions", return_value=[mock_position]):
            # Mock close_position API
            mock_client.close_position.return_value = {"order": {"order_id": "close_123"}}

            with patch(
                "gpt_trader.features.brokerages.coinbase.rest.order_service.to_order"
            ) as mock_to_order:
                mock_to_order.return_value = Mock(spec=Order)
                order = order_service.close_position("BTC-USD")

            assert order is not None

    def test_close_position_no_position(self, portfolio_service, order_service):
        """Test position closing when no position exists."""
        with patch.object(portfolio_service, "list_positions", return_value=[]):
            with pytest.raises(ValidationError, match="No open position"):
                order_service.close_position("BTC-USD")

    def test_close_position_fallback(
        self, portfolio_service, order_service, mock_product_catalog, mock_product, mock_client
    ):
        """Test position closing with fallback when API fails."""
        mock_product_catalog.get.return_value = mock_product

        # Mock positions
        mock_position = Mock(spec=Position)
        mock_position.symbol = "BTC-USD"
        mock_position.quantity = Decimal("1.0")

        with patch.object(portfolio_service, "list_positions", return_value=[mock_position]):
            # Mock API failure and fallback success
            mock_client.close_position.side_effect = Exception("API failed")
            fallback_order = Mock(spec=Order)
            fallback_func = Mock(return_value=fallback_order)

            order = order_service.close_position("BTC-USD", fallback=fallback_func)

            assert order == fallback_order
            fallback_func.assert_called_once()

    # ------------------------------------------------------------------
    # PortfolioService Contract Tests
    # ------------------------------------------------------------------

    def test_list_balances_success(self, portfolio_service, mock_client):
        """Test successful balance listing."""
        mock_accounts = [
            {
                "currency": "BTC",
                "available_balance": {"value": "1.5"},
                "hold": {"value": "0.1"},
                "balance": {"value": "1.6"},
            },
            {
                "currency": "USD",
                "available_balance": {"value": "1000.00"},
                "hold": {"value": "0.00"},
                "balance": {"value": "1000.00"},
            },
        ]
        mock_client.get_accounts.return_value = {"accounts": mock_accounts}

        balances = portfolio_service.list_balances()

        assert len(balances) == 2
        btc_balance = next(b for b in balances if b.asset == "BTC")
        assert btc_balance.total == Decimal("1.6")
        assert btc_balance.available == Decimal("1.5")
        assert btc_balance.hold == Decimal("0.1")

    def test_list_balances_error_handling(self, portfolio_service, mock_client):
        """Test balance listing with malformed data."""
        mock_accounts = [
            {
                "currency": "BTC",
                "available_balance": "invalid",  # Invalid format
                "hold": {"value": "0.1"},
                "balance": {"value": "1.6"},
            }
        ]
        mock_client.get_accounts.return_value = {"accounts": mock_accounts}

        balances = portfolio_service.list_balances()

        # Should skip malformed entries but not crash
        assert len(balances) == 0

    def test_get_portfolio_balances_fallback(self, portfolio_service, mock_client):
        """Test portfolio balances fallback to account balances."""
        mock_client.get_accounts.return_value = {"accounts": []}

        # Mock fallback behavior
        with patch.object(portfolio_service, "list_balances") as mock_list_balances:
            mock_list_balances.return_value = [Mock(spec=Balance)]
            _ = portfolio_service.get_portfolio_balances()

        mock_list_balances.assert_called_once()

    def test_list_positions_success(self, portfolio_service, mock_endpoints, mock_client):
        """Test successful position listing."""
        mock_endpoints.supports_derivatives.return_value = True
        mock_positions = [
            {
                "product_id": "BTC-USD",
                "side": "long",
                "size": "1.5",
                "entry_price": "50000.00",
                "mark_price": "51000.00",
                "unrealized_pnl": "750.00",
                "leverage": 5,
            }
        ]
        mock_client.list_positions.return_value = {"positions": mock_positions}

        positions = portfolio_service.list_positions()

        assert len(positions) == 1
        pos = positions[0]
        assert pos.symbol == "BTC-USD"
        assert pos.quantity == Decimal("1.5")
        assert pos.side == "long"

    def test_list_positions_error_handling(self, portfolio_service, mock_endpoints, mock_client):
        """Test position listing error handling."""
        mock_endpoints.supports_derivatives.return_value = True
        mock_client.list_positions.side_effect = Exception("API error")

        positions = portfolio_service.list_positions()

        assert positions == []

    def test_get_position_success(self, portfolio_service, mock_endpoints, mock_client):
        """Test successful position retrieval."""
        mock_endpoints.supports_derivatives.return_value = True
        mock_position = {
            "product_id": "BTC-USD",
            "side": "long",
            "size": "1.0",
            "entry_price": "50000.00",
        }
        mock_client.get_position.return_value = mock_position

        position = portfolio_service.get_position("BTC-USD")

        assert position is not None
        assert position.symbol == "BTC-USD"

    def test_get_position_not_found(self, portfolio_service, mock_endpoints, mock_client):
        """Test position retrieval when not found."""
        mock_endpoints.supports_derivatives.return_value = True
        mock_client.get_position.side_effect = Exception("Position not found")

        position = portfolio_service.get_position("BTC-USD")

        assert position is None

    def test_intx_allocate_success(self, portfolio_service, mock_endpoints, mock_client):
        """Test successful INTX allocation."""
        mock_endpoints.mode = "advanced"
        mock_response = {"allocation_id": "alloc_123", "status": "confirmed"}
        mock_client.intx_allocate.return_value = mock_response

        result = portfolio_service.intx_allocate({"amount": "1000", "currency": "USD"})

        assert result["allocation_id"] == "alloc_123"
        # Verify telemetry emission
        portfolio_service._event_store.append_metric.assert_called()

    def test_intx_allocate_error_handling(self, portfolio_service, mock_endpoints, mock_client):
        """Test INTX allocation error handling."""
        mock_endpoints.mode = "advanced"
        mock_client.intx_allocate.side_effect = Exception("Allocation failed")

        with pytest.raises(Exception, match="Allocation failed"):
            portfolio_service.intx_allocate({"amount": "1000"})

    def test_cfm_balance_summary_telemetry(self, portfolio_service, mock_endpoints, mock_client):
        """Test CFM balance summary with telemetry."""
        mock_endpoints.supports_derivatives.return_value = True
        mock_summary = {
            "total_balance": "10000.00",
            "available_balance": "9500.00",
            "timestamp": "2024-01-01T12:00:00Z",
        }
        mock_client.cfm_balance_summary.return_value = {"balance_summary": mock_summary}

        result = portfolio_service.get_cfm_balance_summary()

        assert result["total_balance"] == Decimal("10000.00")
        # Verify telemetry emission
        portfolio_service._event_store.append_metric.assert_called()

    def test_cfm_sweeps_telemetry(self, portfolio_service, mock_endpoints, mock_client):
        """Test CFM sweeps with telemetry."""
        mock_endpoints.supports_derivatives.return_value = True
        mock_sweeps = [
            {"sweep_id": "sweep_1", "amount": "100.00", "status": "completed"},
            {"sweep_id": "sweep_2", "amount": "200.00", "status": "pending"},
        ]
        mock_client.cfm_sweeps.return_value = {"sweeps": mock_sweeps}

        result = portfolio_service.list_cfm_sweeps()

        assert len(result) == 2
        assert result[0]["amount"] == Decimal("100.00")
        # Verify telemetry emission
        portfolio_service._event_store.append_metric.assert_called()

    def test_update_cfm_margin_window_success(self, portfolio_service, mock_endpoints, mock_client):
        """Test successful CFM margin window update."""
        mock_endpoints.supports_derivatives.return_value = True
        mock_response = {"margin_window": "maintenance", "effective_time": "2024-01-01T12:00:00Z"}
        mock_client.cfm_intraday_margin_setting.return_value = mock_response

        result = portfolio_service.update_cfm_margin_window("maintenance")

        assert result["margin_window"] == "maintenance"
        # Verify telemetry emission
        portfolio_service._event_store.append_metric.assert_called()

    # ------------------------------------------------------------------
    # PnLRestMixin Contract Tests
    # ------------------------------------------------------------------

    def test_process_fill_for_pnl_new_position(self, pnl_service, mock_market_data):
        """Test PnL processing for new position creation."""
        mock_market_data.get_mark.return_value = Decimal("51000")

        fill = {
            "product_id": "BTC-USD",
            "size": "1.0",
            "price": "50000.00",
            "side": "buy",
        }

        pnl_service.process_fill_for_pnl(fill)

        # Should create new position
        assert "BTC-USD" in pnl_service._position_store.all()
        position = pnl_service._position_store.get("BTC-USD")
        assert position.quantity == Decimal("1.0")
        assert position.entry_price == Decimal("50000.00")

    def test_process_fill_for_pnl_existing_position(self, pnl_service, mock_market_data):
        """Test PnL processing for existing position update."""
        # Setup existing position
        pnl_service._position_store.set(
            "BTC-USD",
            PositionState(
                symbol="BTC-USD",
                side="long",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000.00"),
            ),
        )
        mock_market_data.get_mark.return_value = Decimal("51000")

        fill = {
            "product_id": "BTC-USD",
            "size": "0.5",
            "price": "51000.00",
            "side": "sell",  # Partial close
        }

        pnl_service.process_fill_for_pnl(fill)

        # Should update position and calculate realized PnL
        position = pnl_service._position_store.get("BTC-USD")
        assert position.quantity == Decimal("0.5")  # Reduced by fill
        assert position.realized_pnl == Decimal("500.00")  # (51000 - 50000) * 0.5

    def test_process_fill_for_pnl_invalid_data(self, pnl_service):
        """Test PnL processing with invalid fill data."""
        # Missing required fields
        fill = {"product_id": "BTC-USD"}

        # Should not crash, just return early
        pnl_service.process_fill_for_pnl(fill)

        assert not pnl_service._position_store.contains("BTC-USD")

    def test_get_position_pnl_no_position(self, pnl_service):
        """Test position PnL retrieval for non-existent position."""
        pnl = pnl_service.get_position_pnl("BTC-USD")

        assert pnl["symbol"] == "BTC-USD"
        assert pnl["quantity"] == Decimal("0")
        assert pnl["unrealized_pnl"] == Decimal("0")
        assert pnl["realized_pnl"] == Decimal("0")

    def test_get_position_pnl_with_position(self, pnl_service, mock_market_data):
        """Test position PnL retrieval for existing position."""
        # Setup position
        pnl_service._position_store.set(
            "BTC-USD",
            PositionState(
                symbol="BTC-USD",
                side="long",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000.00"),
                realized_pnl=Decimal("1000.00"),
            ),
        )
        mock_market_data.get_mark.return_value = Decimal("51000")

        pnl = pnl_service.get_position_pnl("BTC-USD")

        assert pnl["quantity"] == Decimal("1.0")
        assert pnl["entry"] == Decimal("50000.00")
        assert pnl["mark"] == Decimal("51000")
        assert pnl["unrealized_pnl"] == Decimal("1000.00")  # (51000 - 50000) * 1.0
        assert pnl["realized_pnl"] == Decimal("1000.00")

    def test_get_portfolio_pnl_aggregation(self, pnl_service, mock_market_data):
        """Test portfolio PnL aggregation across multiple positions."""
        # Setup multiple positions
        pnl_service._position_store.set(
            "BTC-USD",
            PositionState(
                symbol="BTC-USD",
                side="long",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000.00"),
                realized_pnl=Decimal("1000.00"),
            ),
        )
        pnl_service._position_store.set(
            "ETH-USD",
            PositionState(
                symbol="ETH-USD",
                side="short",
                quantity=Decimal("10.0"),
                entry_price=Decimal("3000.00"),
                realized_pnl=Decimal("500.00"),
            ),
        )

        def get_mark_side_effect(symbol):
            if symbol == "BTC-USD":
                return Decimal("51000")
            if symbol == "ETH-USD":
                return Decimal("3000")
            return Decimal("0")

        mock_market_data.get_mark.side_effect = get_mark_side_effect

        portfolio_pnl = pnl_service.get_portfolio_pnl()

        assert portfolio_pnl["total_realized_pnl"] == Decimal("1500.00")
        assert portfolio_pnl["total_unrealized_pnl"] == Decimal("1000.00")  # BTC unrealized
        assert portfolio_pnl["total_pnl"] == Decimal("2500.00")
        assert len(portfolio_pnl["positions"]) == 2

    # ------------------------------------------------------------------
    # Retry and Telemetry Logging Tests
    # ------------------------------------------------------------------

    def test_order_retry_on_duplicate_client_id(
        self, order_service, mock_product_catalog, mock_product, mock_client
    ):
        """Test order retry logic on duplicate client ID error."""
        mock_product_catalog.get.return_value = mock_product

        # First call fails with duplicate, second succeeds
        mock_client.place_order.side_effect = [
            InvalidRequestError("duplicate client_order_id"),
            {"order_id": "retry_success_123"},
        ]

        # Mock list_orders via mock_client to return empty so find_existing returns None
        mock_client.list_orders.return_value = {"orders": []}

        with patch("gpt_trader.features.brokerages.coinbase.models.to_order") as mock_to_order:
            mock_to_order.return_value = Mock(spec=Order)
            order = order_service.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
                price=Decimal("50000.00"),
                client_id="duplicate_id",
            )

        assert order is not None
        assert mock_client.place_order.call_count == 2

    def test_telemetry_logging_on_api_calls(self, portfolio_service, mock_endpoints, mock_client):
        """Test telemetry logging on various API calls."""
        mock_endpoints.mode = "advanced"

        # Test INTX allocate telemetry
        mock_client.intx_allocate.return_value = {"status": "success"}
        portfolio_service.intx_allocate({"amount": "1000"})

        # Test CFM telemetry
        mock_endpoints.supports_derivatives.return_value = True
        mock_client.cfm_balance_summary.return_value = {"balance_summary": {}}
        portfolio_service.get_cfm_balance_summary()

        # Verify telemetry was emitted for both calls
        assert portfolio_service._event_store.append_metric.call_count >= 2

    def test_pagination_cursor_handling(self, order_service, mock_client):
        """Test pagination cursor handling in list operations."""
        # Mock paginated fills response
        mock_client.list_fills.side_effect = [
            {"fills": [{"id": "1"}], "cursor": "page2"},
            {"fills": [{"id": "2"}]},  # No cursor = end
        ]

        fills = order_service.list_fills(limit=1)

        assert len(fills) == 2
        # Verify cursor was used in second call
        second_call = mock_client.list_fills.call_args_list[1]
        assert second_call[1]["cursor"] == "page2"
