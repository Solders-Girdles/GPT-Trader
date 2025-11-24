"""Contract tests for Coinbase REST service mixins."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from gpt_trader.errors import ValidationError
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.models import APIConfig, Product
from gpt_trader.features.brokerages.coinbase.rest.base import CoinbaseRestServiceBase
from gpt_trader.features.brokerages.coinbase.rest.orders import OrderRestMixin
from gpt_trader.features.brokerages.coinbase.rest.pnl import PnLRestMixin
from gpt_trader.features.brokerages.coinbase.rest.portfolio import PortfolioRestMixin
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
    """Contract tests for Coinbase REST service mixins."""

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
    def service_base(
        self,
        mock_client,
        mock_endpoints,
        mock_config,
        mock_product_catalog,
        mock_market_data,
        mock_event_store,
    ) -> CoinbaseRestServiceBase:
        return CoinbaseRestServiceBase(
            client=mock_client,
            endpoints=mock_endpoints,
            config=mock_config,
            product_catalog=mock_product_catalog,
            market_data=mock_market_data,
            event_store=mock_event_store,
        )

    @pytest.fixture
    def order_service(self, service_base) -> OrderRestMixin:
        """Create a service with OrderRestMixin."""
        service_base.__class__ = type(
            "TestOrderService",
            (CoinbaseRestServiceBase, OrderRestMixin),
            {},
        )
        return service_base

    @pytest.fixture
    def portfolio_service(self, service_base) -> PortfolioRestMixin:
        """Create a service with PortfolioRestMixin."""
        service_base.__class__ = type(
            "TestPortfolioService",
            (CoinbaseRestServiceBase, PortfolioRestMixin),
            {},
        )
        return service_base

    @pytest.fixture
    def pnl_service(self, service_base) -> PnLRestMixin:
        """Create a service with PnLRestMixin."""
        service_base.__class__ = type(
            "TestPnLService",
            (CoinbaseRestServiceBase, PnLRestMixin),
            {},
        )
        return service_base

    # ------------------------------------------------------------------
    # OrderRestMixin Contract Tests
    # ------------------------------------------------------------------

    def test_place_order_quantity_resolution_success(
        self, order_service, mock_product_catalog, mock_product
    ):
        """Test successful order placement with quantity resolution."""
        mock_product_catalog.get.return_value = mock_product
        order_service.client.place_order.return_value = {"order_id": "test_123"}
        order_service._execute_order_payload = Mock(return_value=Mock(spec=Order))

        order = order_service.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.123456789"),
            price=Decimal("50000.00"),
        )

        assert order is not None
        # Verify quantity was quantized to step_size
        call_args = order_service._execute_order_payload.call_args
        payload = call_args[0][1]  # Second argument is payload
        assert payload["order_configuration"]["limit_limit_gtc"]["base_size"] == "0.12345678"

    def test_place_order_quantity_resolution_error_branch(
        self, order_service, mock_product_catalog, mock_product
    ):
        """Test order placement with quantity below minimum."""
        mock_product_catalog.get.return_value = mock_product

        with pytest.raises(ValidationError, match="quantity .* is below min_size"):
            order_service.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.0001"),  # Below min_size
                price=Decimal("50000.00"),
            )

    def test_place_order_error_branch_insufficient_funds(
        self, order_service, mock_product_catalog, mock_product
    ):
        """Test order placement with insufficient funds error."""
        mock_product_catalog.get.return_value = mock_product
        order_service.client.place_order.side_effect = InsufficientFunds("Insufficient balance")

        with pytest.raises(InsufficientFunds):
            order_service.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
                price=Decimal("50000.00"),
            )

    def test_place_order_error_branch_validation_error(
        self, order_service, mock_product_catalog, mock_product
    ):
        """Test order placement with validation error."""
        mock_product_catalog.get.return_value = mock_product
        order_service.client.place_order.side_effect = ValidationError("Invalid order parameters")

        with pytest.raises(ValidationError):
            order_service.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
                price=Decimal("50000.00"),
            )

    def test_cancel_order_success(self, order_service):
        """Test successful order cancellation."""
        order_service.client.cancel_orders.return_value = {
            "results": [{"order_id": "test_123", "success": True}]
        }

        result = order_service.cancel_order("test_123")

        assert result is True

    def test_cancel_order_failure(self, order_service):
        """Test order cancellation failure."""
        order_service.client.cancel_orders.return_value = {
            "results": [{"order_id": "test_123", "success": False}]
        }

        result = order_service.cancel_order("test_123")

        assert result is False

    def test_list_orders_with_pagination(self, order_service):
        """Test order listing with pagination handling."""
        # Mock paginated responses
        order_service.client.list_orders.side_effect = [
            {"orders": [{"order_id": "1"}, {"order_id": "2"}], "cursor": "next_page"},
            {"orders": [{"order_id": "3"}]},  # No cursor = last page
        ]

        with patch("gpt_trader.features.brokerages.coinbase.rest.orders.to_order") as mock_to_order:
            mock_to_order.return_value = Mock(spec=Order)
            orders = order_service.list_orders()

        assert len(orders) == 3
        # Verify pagination was handled (called twice)
        assert order_service.client.list_orders.call_count == 2

    def test_list_orders_error_handling(self, order_service):
        """Test order listing with error handling."""
        order_service.client.list_orders.side_effect = Exception("API error")

        orders = order_service.list_orders()

        assert orders == []

    def test_get_order_success(self, order_service):
        """Test successful order retrieval."""
        mock_order_data = {"order_id": "test_123", "status": "filled"}
        order_service.client.get_order_historical.return_value = {"order": mock_order_data}

        with patch("gpt_trader.features.brokerages.coinbase.rest.orders.to_order") as mock_to_order:
            mock_to_order.return_value = Mock(spec=Order)
            order = order_service.get_order("test_123")

        assert order is not None

    def test_get_order_not_found(self, order_service):
        """Test order retrieval when not found."""
        order_service.client.get_order_historical.side_effect = Exception("Order not found")

        order = order_service.get_order("test_123")

        assert order is None

    def test_list_fills_with_pagination(self, order_service):
        """Test fills listing with pagination."""
        order_service.client.list_fills.side_effect = [
            {"fills": [{"fill_id": "1"}, {"fill_id": "2"}], "cursor": "next"},
            {"fills": [{"fill_id": "3"}]},
        ]

        fills = order_service.list_fills()

        assert len(fills) == 3

    def test_list_fills_error_handling(self, order_service):
        """Test fills listing error handling."""
        order_service.client.list_fills.side_effect = Exception("API error")

        fills = order_service.list_fills()

        assert fills == []

    def test_close_position_success(self, portfolio_service, order_service):
        """Test successful position closing."""
        # Mock positions
        mock_position = Mock(spec=Position)
        mock_position.symbol = "BTC-USD"
        mock_position.quantity = Decimal("1.0")
        portfolio_service.list_positions.return_value = [mock_position]

        # Mock close_position API
        order_service.client.close_position.return_value = {"order": {"order_id": "close_123"}}

        with patch("gpt_trader.features.brokerages.coinbase.rest.orders.to_order") as mock_to_order:
            mock_to_order.return_value = Mock(spec=Order)
            order = order_service.close_position("BTC-USD")

        assert order is not None

    def test_close_position_no_position(self, portfolio_service, order_service):
        """Test position closing when no position exists."""
        portfolio_service.list_positions.return_value = []

        with pytest.raises(ValidationError, match="No open position"):
            order_service.close_position("BTC-USD")

    def test_close_position_fallback(
        self, portfolio_service, order_service, mock_product_catalog, mock_product
    ):
        """Test position closing with fallback when API fails."""
        mock_product_catalog.get.return_value = mock_product

        # Mock positions
        mock_position = Mock(spec=Position)
        mock_position.symbol = "BTC-USD"
        mock_position.quantity = Decimal("1.0")
        portfolio_service.list_positions.return_value = [mock_position]

        # Mock API failure and fallback success
        order_service.client.close_position.side_effect = Exception("API failed")
        fallback_order = Mock(spec=Order)
        fallback_func = Mock(return_value=fallback_order)

        order = order_service.close_position("BTC-USD", fallback=fallback_func)

        assert order == fallback_order
        fallback_func.assert_called_once()

    # ------------------------------------------------------------------
    # PortfolioRestMixin Contract Tests
    # ------------------------------------------------------------------

    def test_list_balances_success(self, portfolio_service):
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
        portfolio_service.client.get_accounts.return_value = {"accounts": mock_accounts}

        balances = portfolio_service.list_balances()

        assert len(balances) == 2
        btc_balance = next(b for b in balances if b.asset == "BTC")
        assert btc_balance.total == Decimal("1.6")
        assert btc_balance.available == Decimal("1.5")
        assert btc_balance.hold == Decimal("0.1")

    def test_list_balances_error_handling(self, portfolio_service):
        """Test balance listing with malformed data."""
        mock_accounts = [
            {
                "currency": "BTC",
                "available_balance": "invalid",  # Invalid format
                "hold": {"value": "0.1"},
                "balance": {"value": "1.6"},
            }
        ]
        portfolio_service.client.get_accounts.return_value = {"accounts": mock_accounts}

        balances = portfolio_service.list_balances()

        # Should skip malformed entries but not crash
        assert len(balances) == 0

    def test_get_portfolio_balances_fallback(self, portfolio_service):
        """Test portfolio balances fallback to account balances."""
        portfolio_service.client.get_accounts.return_value = {"accounts": []}

        # Mock fallback behavior
        with patch.object(portfolio_service, "list_balances") as mock_list_balances:
            mock_list_balances.return_value = [Mock(spec=Balance)]
            _ = portfolio_service.get_portfolio_balances()

        mock_list_balances.assert_called_once()

    def test_list_positions_success(self, portfolio_service):
        """Test successful position listing."""
        portfolio_service.endpoints.supports_derivatives.return_value = True
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
        portfolio_service.client.list_positions.return_value = {"positions": mock_positions}

        positions = portfolio_service.list_positions()

        assert len(positions) == 1
        pos = positions[0]
        assert pos.symbol == "BTC-USD"
        assert pos.quantity == Decimal("1.5")
        assert pos.side == "long"

    def test_list_positions_error_handling(self, portfolio_service):
        """Test position listing error handling."""
        portfolio_service.endpoints.supports_derivatives.return_value = True
        portfolio_service.client.list_positions.side_effect = Exception("API error")

        positions = portfolio_service.list_positions()

        assert positions == []

    def test_get_position_success(self, portfolio_service):
        """Test successful position retrieval."""
        portfolio_service.endpoints.supports_derivatives.return_value = True
        mock_position = {
            "product_id": "BTC-USD",
            "side": "long",
            "size": "1.0",
            "entry_price": "50000.00",
        }
        portfolio_service.client.get_position.return_value = mock_position

        position = portfolio_service.get_position("BTC-USD")

        assert position is not None
        assert position.symbol == "BTC-USD"

    def test_get_position_not_found(self, portfolio_service):
        """Test position retrieval when not found."""
        portfolio_service.endpoints.supports_derivatives.return_value = True
        portfolio_service.client.get_position.side_effect = Exception("Position not found")

        position = portfolio_service.get_position("BTC-USD")

        assert position is None

    def test_intx_allocate_success(self, portfolio_service):
        """Test successful INTX allocation."""
        portfolio_service.endpoints.mode = "advanced"
        mock_response = {"allocation_id": "alloc_123", "status": "confirmed"}
        portfolio_service.client.intx_allocate.return_value = mock_response

        result = portfolio_service.intx_allocate({"amount": "1000", "currency": "USD"})

        assert result["allocation_id"] == "alloc_123"
        # Verify telemetry emission
        portfolio_service._event_store.append.assert_called()

    def test_intx_allocate_error_handling(self, portfolio_service):
        """Test INTX allocation error handling."""
        portfolio_service.endpoints.mode = "advanced"
        portfolio_service.client.intx_allocate.side_effect = Exception("Allocation failed")

        with pytest.raises(Exception, match="Allocation failed"):
            portfolio_service.intx_allocate({"amount": "1000"})

    def test_cfm_balance_summary_telemetry(self, portfolio_service):
        """Test CFM balance summary with telemetry."""
        portfolio_service.endpoints.supports_derivatives.return_value = True
        mock_summary = {
            "total_balance": "10000.00",
            "available_balance": "9500.00",
            "timestamp": "2024-01-01T12:00:00Z",
        }
        portfolio_service.client.cfm_balance_summary.return_value = {
            "balance_summary": mock_summary
        }

        result = portfolio_service.get_cfm_balance_summary()

        assert result["total_balance"] == Decimal("10000.00")
        # Verify telemetry emission
        portfolio_service._event_store.append.assert_called()

    def test_cfm_sweeps_telemetry(self, portfolio_service):
        """Test CFM sweeps with telemetry."""
        portfolio_service.endpoints.supports_derivatives.return_value = True
        mock_sweeps = [
            {"sweep_id": "sweep_1", "amount": "100.00", "status": "completed"},
            {"sweep_id": "sweep_2", "amount": "200.00", "status": "pending"},
        ]
        portfolio_service.client.cfm_sweeps.return_value = {"sweeps": mock_sweeps}

        result = portfolio_service.list_cfm_sweeps()

        assert len(result) == 2
        assert result[0]["amount"] == Decimal("100.00")
        # Verify telemetry emission
        portfolio_service._event_store.append.assert_called()

    def test_update_cfm_margin_window_success(self, portfolio_service):
        """Test successful CFM margin window update."""
        portfolio_service.endpoints.supports_derivatives.return_value = True
        mock_response = {"margin_window": "maintenance", "effective_time": "2024-01-01T12:00:00Z"}
        portfolio_service.client.cfm_intraday_margin_setting.return_value = mock_response

        result = portfolio_service.update_cfm_margin_window("maintenance")

        assert result["margin_window"] == "maintenance"
        # Verify telemetry emission
        portfolio_service._event_store.append.assert_called()

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
        assert "BTC-USD" in pnl_service.positions
        position = pnl_service.positions["BTC-USD"]
        assert position.quantity == Decimal("1.0")
        assert position.entry_price == Decimal("50000.00")

    def test_process_fill_for_pnl_existing_position(self, pnl_service, mock_market_data):
        """Test PnL processing for existing position update."""
        # Setup existing position
        pnl_service.positions["BTC-USD"] = PositionState(
            symbol="BTC-USD",
            side="long",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
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
        position = pnl_service.positions["BTC-USD"]
        assert position.quantity == Decimal("0.5")  # Reduced by fill
        assert position.realized_pnl == Decimal("500.00")  # (51000 - 50000) * 0.5

    def test_process_fill_for_pnl_invalid_data(self, pnl_service):
        """Test PnL processing with invalid fill data."""
        # Missing required fields
        fill = {"product_id": "BTC-USD"}

        # Should not crash, just return early
        pnl_service.process_fill_for_pnl(fill)

        assert "BTC-USD" not in pnl_service.positions

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
        pnl_service.positions["BTC-USD"] = PositionState(
            symbol="BTC-USD",
            side="long",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            realized_pnl=Decimal("1000.00"),
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
        pnl_service.positions["BTC-USD"] = PositionState(
            symbol="BTC-USD",
            side="long",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            realized_pnl=Decimal("1000.00"),
        )
        pnl_service.positions["ETH-USD"] = PositionState(
            symbol="ETH-USD",
            side="short",
            quantity=Decimal("10.0"),
            entry_price=Decimal("3000.00"),
            realized_pnl=Decimal("500.00"),
        )
        mock_market_data.get_mark.return_value = Decimal("51000")  # For BTC

        portfolio_pnl = pnl_service.get_portfolio_pnl()

        assert portfolio_pnl["total_realized_pnl"] == Decimal("1500.00")
        assert portfolio_pnl["total_unrealized_pnl"] == Decimal("1000.00")  # BTC unrealized
        assert portfolio_pnl["total_pnl"] == Decimal("2500.00")
        assert len(portfolio_pnl["positions"]) == 2

    # ------------------------------------------------------------------
    # Retry and Telemetry Logging Tests
    # ------------------------------------------------------------------

    def test_order_retry_on_duplicate_client_id(
        self, order_service, mock_product_catalog, mock_product
    ):
        """Test order retry logic on duplicate client ID error."""
        mock_product_catalog.get.return_value = mock_product

        # First call fails with duplicate, second succeeds
        order_service.client.place_order.side_effect = [
            InvalidRequestError("duplicate client_order_id"),
            {"order_id": "retry_success_123"},
        ]
        order_service._find_existing_order_by_client_id = Mock(return_value=None)

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
        assert order_service.client.place_order.call_count == 2

    def test_telemetry_logging_on_api_calls(self, portfolio_service):
        """Test telemetry logging on various API calls."""
        portfolio_service.endpoints.mode = "advanced"

        # Test INTX allocate telemetry
        portfolio_service.client.intx_allocate.return_value = {"status": "success"}
        portfolio_service.intx_allocate({"amount": "1000"})

        # Test CFM telemetry
        portfolio_service.endpoints.supports_derivatives.return_value = True
        portfolio_service.client.cfm_balance_summary.return_value = {"balance_summary": {}}
        portfolio_service.get_cfm_balance_summary()

        # Verify telemetry was emitted for both calls
        assert portfolio_service._event_store.append.call_count >= 2

    def test_pagination_cursor_handling(self, order_service):
        """Test pagination cursor handling in list operations."""
        # Mock paginated fills response
        order_service.client.list_fills.side_effect = [
            {"fills": [{"id": "1"}], "cursor": "page2"},
            {"fills": [{"id": "2"}]},  # No cursor = end
        ]

        fills = order_service.list_fills(limit=1)

        assert len(fills) == 2
        # Verify cursor was used in second call
        second_call = order_service.client.list_fills.call_args_list[1]
        assert second_call[1]["cursor"] == "page2"
