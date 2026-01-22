"""Shared fixtures for Coinbase REST contract suite tests."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

import gpt_trader.features.brokerages.coinbase.models as models_module
import gpt_trader.features.brokerages.coinbase.rest.base as rest_base_module
import gpt_trader.features.brokerages.coinbase.rest.order_service as order_service_module
from gpt_trader.core import (
    Balance,
    InsufficientFunds,
    InvalidRequestError,
    Order,
    OrderSide,
    OrderType,
    Position,
)
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
from gpt_trader.persistence.event_store import EventStore

BTC_SYMBOL = "BTC-USD"
ETH_SYMBOL = "ETH-USD"
TIMESTAMP = "2024-01-01T12:00:00Z"
WEIGHTED_ENTRY = (
    (Decimal("1.0") * Decimal("50000.00")) + (Decimal("0.5") * Decimal("51000.00"))
) / Decimal("1.5")


def fill(symbol: str, size: str, price: str, side: str) -> dict:
    return {"product_id": symbol, "size": size, "price": price, "side": side}


def state(symbol: str, side: str, quantity: str, entry: str, realized: str = "0") -> PositionState:
    return PositionState(
        symbol=symbol,
        side=side,
        quantity=Decimal(quantity),
        entry_price=Decimal(entry),
        realized_pnl=Decimal(realized),
    )


def account(
    currency: str,
    available: str,
    hold: str,
    balance: str,
    *,
    raw_available: bool = False,
) -> dict:
    available_value = available if raw_available else {"value": available}
    return {
        "currency": currency,
        "available_balance": available_value,
        "hold": {"value": hold},
        "balance": {"value": balance},
    }


def position(
    symbol: str,
    side: str,
    quantity: str,
    entry: str,
    mark: str,
    unrealized: str,
    realized: str,
    leverage: int,
) -> Position:
    return Position(
        symbol=symbol,
        side=side,
        quantity=Decimal(quantity),
        entry_price=Decimal(entry),
        mark_price=Decimal(mark),
        unrealized_pnl=Decimal(unrealized),
        realized_pnl=Decimal(realized),
        leverage=leverage,
    )


def balance_summary(total: str, available: str, timestamp: str = TIMESTAMP) -> dict:
    return {
        "balance_summary": {
            "total_balance": total,
            "available_balance": available,
            "timestamp": timestamp,
        }
    }


def sweep(sweep_id: str, amount: str, status: str) -> dict:
    return {"sweep_id": sweep_id, "amount": amount, "status": status}


class CoinbaseRestContractSuiteBase:
    """Contract fixtures for Coinbase REST service components."""

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
        return PortfolioService(
            client=mock_client,
            endpoints=mock_endpoints,
            event_store=mock_event_store,
        )

    @pytest.fixture
    def order_service(self, mock_client, service_core, portfolio_service) -> OrderService:
        return OrderService(
            client=mock_client,
            payload_builder=service_core,
            payload_executor=service_core,
            position_provider=portfolio_service,
        )

    @pytest.fixture
    def pnl_service(self, position_store, mock_market_data) -> PnLService:
        return PnLService(
            position_store=position_store,
            market_data=mock_market_data,
        )


def assert_order_service_contracts(
    order_service,
    service_core,
    portfolio_service,
    mock_product_catalog,
    mock_product,
    mock_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_order = {
        "symbol": BTC_SYMBOL,
        "side": OrderSide.BUY,
        "order_type": OrderType.LIMIT,
        "price": Decimal("50000.00"),
    }
    mock_product_catalog.get.return_value, mock_client.place_order.return_value = (
        mock_product,
        {"order_id": "test_123"},
    )
    mock_execute = Mock(side_effect=service_core.execute_order_payload)
    monkeypatch.setattr(service_core, "execute_order_payload", mock_execute)
    monkeypatch.setattr(rest_base_module, "to_order", Mock(return_value=Mock(spec=Order)))
    monkeypatch.setattr(order_service_module, "to_order", Mock(return_value=Mock(spec=Order)))
    assert order_service.place_order(**base_order, quantity=Decimal("0.123456789")) is not None
    assert (
        mock_execute.call_args[0][1]["order_configuration"]["limit_limit_gtc"]["base_size"]
        == "0.12345678"
    )
    with pytest.raises(InvalidRequestError, match="quantity .* is below minimum size"):
        order_service.place_order(**base_order, quantity=Decimal("0.0001"))
    for exc in (InsufficientFunds("Insufficient balance"), ValidationError("Invalid order")):
        mock_client.place_order.side_effect = exc
        with pytest.raises(type(exc)):
            order_service.place_order(**base_order, quantity=Decimal("0.1"))
    for success in (True, False):
        mock_client.cancel_orders.return_value = {
            "results": [{"order_id": "test_123", "success": success}]
        }
        if success:
            assert order_service.cancel_order("test_123") is True
        else:
            with pytest.raises(OrderCancellationError, match="Cancellation rejected"):
                order_service.cancel_order("test_123")
    for method, key, error in (
        ("list_orders", "orders", "Failed to list orders"),
        ("list_fills", "fills", "Failed to list fills"),
    ):
        getattr(mock_client, method).side_effect = [
            {key: [{"id": "1"}, {"id": "2"}], "cursor": "next"},
            {key: [{"id": "3"}]},
        ]
        if method == "list_fills":
            result = getattr(order_service, method)(limit=1)
            assert getattr(mock_client, method).call_args_list[1][1]["cursor"] == "next"
        else:
            result = getattr(order_service, method)()
        assert len(result) == 3
        assert getattr(mock_client, method).call_count == 2
        getattr(mock_client, method).side_effect = Exception("API error")
        with pytest.raises(OrderQueryError, match=error):
            getattr(order_service, method)()
        if method == "list_orders":
            getattr(mock_client, method).side_effect = None
    mock_client.get_order_historical.return_value = {"order": {"order_id": "test_123"}}
    assert order_service.get_order("test_123") is not None
    mock_client.get_order_historical.side_effect = Exception("Order not found")
    with pytest.raises(OrderQueryError, match="Failed to get order"):
        order_service.get_order("test_123")
    mock_position = Mock(spec=Position, symbol=BTC_SYMBOL, quantity=Decimal("1.0"))
    monkeypatch.setattr(portfolio_service, "list_positions", lambda: [mock_position])
    mock_client.close_position.return_value = {"order": {"order_id": "close_123"}}
    assert order_service.close_position(BTC_SYMBOL) is not None
    monkeypatch.setattr(portfolio_service, "list_positions", lambda: [])
    with pytest.raises(ValidationError, match="No open position"):
        order_service.close_position(BTC_SYMBOL)
    monkeypatch.setattr(portfolio_service, "list_positions", lambda: [mock_position])
    mock_client.close_position.side_effect = Exception("API failed")
    fallback = Mock(return_value=Mock(spec=Order))
    assert order_service.close_position(BTC_SYMBOL, fallback=fallback) == fallback.return_value
    fallback.assert_called_once()
    mock_client.place_order.reset_mock()
    mock_client.place_order.side_effect = [
        InvalidRequestError("duplicate client_order_id"),
        {"order_id": "retry_success_123"},
    ]
    mock_client.list_orders.return_value = {"orders": []}
    monkeypatch.setattr(models_module, "to_order", Mock(return_value=Mock(spec=Order)))
    assert (
        order_service.place_order(**base_order, quantity=Decimal("0.1"), client_id="duplicate_id")
        is not None
    )
    assert mock_client.place_order.call_count == 2


def assert_pnl_service_contracts(pnl_service, mock_market_data) -> None:
    cases = [
        (None, (BTC_SYMBOL, "1.0", "50000.00", "buy"), "51000", ("1.0", "50000.00", None)),
        (
            (BTC_SYMBOL, "long", "1.0", "50000.00"),
            (BTC_SYMBOL, "0.5", "51000.00", "sell"),
            "51000",
            ("0.5", None, "500.00"),
        ),
        (
            (BTC_SYMBOL, "long", "1.0", "50000.00"),
            (BTC_SYMBOL, "0.5", "51000.00", "buy"),
            None,
            ("1.5", "weighted", None),
        ),
        (
            (ETH_SYMBOL, "short", "2.0", "100.00"),
            (ETH_SYMBOL, "2.0", "90.00", "buy"),
            None,
            ("0", None, "20.00"),
        ),
    ]
    store = pnl_service._position_store
    for pre, fill_data, mark, expected in cases:
        store.clear()
        if pre:
            store.set(pre[0], state(*pre))
        if mark is not None:
            mock_market_data.get_mark.return_value = Decimal(mark)
        pnl_service.process_fill_for_pnl(fill(*fill_data))
        position = store.get(fill_data[0])
        quantity, entry, realized = expected
        assert position.quantity == Decimal(quantity)
        if entry is not None:
            entry_value = WEIGHTED_ENTRY if entry == "weighted" else Decimal(entry)
            assert position.entry_price == entry_value
        if realized is not None:
            assert position.realized_pnl == Decimal(realized)
    store_stub = Mock()
    store_stub.contains.return_value, store_stub.get.return_value = True, None
    PnLService(position_store=store_stub, market_data=mock_market_data).process_fill_for_pnl(
        fill(BTC_SYMBOL, "0.5", "51000.00", "sell")
    )
    store_stub.set.assert_not_called()
    store.clear()
    pnl_service.process_fill_for_pnl({"product_id": BTC_SYMBOL})
    assert not store.contains(BTC_SYMBOL)
    store.clear()
    pnl = pnl_service.get_position_pnl(BTC_SYMBOL)
    assert (
        pnl["quantity"] == Decimal("0")
        and pnl["unrealized_pnl"] == Decimal("0")
        and pnl["realized_pnl"] == Decimal("0")
    )
    store.set(BTC_SYMBOL, state(BTC_SYMBOL, "long", "1.0", "50000.00", "1000.00"))
    mock_market_data.get_mark.return_value = Decimal("51000")
    pnl = pnl_service.get_position_pnl(BTC_SYMBOL)
    assert (
        pnl["quantity"] == Decimal("1.0")
        and pnl["realized_pnl"] == Decimal("1000.00")
        and pnl["unrealized_pnl"] == Decimal("1000.00")
    )
    store.clear()
    store.set(BTC_SYMBOL, state(BTC_SYMBOL, "long", "1.0", "50000.00", "1000.00"))
    store.set(ETH_SYMBOL, state(ETH_SYMBOL, "short", "10.0", "3000.00", "500.00"))
    mock_market_data.get_mark.side_effect = lambda symbol: Decimal(
        "51000" if symbol == BTC_SYMBOL else "3000"
    )
    portfolio_pnl = pnl_service.get_portfolio_pnl()
    assert (
        portfolio_pnl["total_realized_pnl"] == Decimal("1500.00")
        and portfolio_pnl["total_unrealized_pnl"] == Decimal("1000.00")
        and portfolio_pnl["total_pnl"] == Decimal("2500.00")
    )
    assert len(portfolio_pnl["positions"]) == 2


def assert_portfolio_service_contracts(
    portfolio_service,
    mock_client,
    mock_endpoints,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_client.get_accounts.return_value = {
        "accounts": [
            account("BTC", "1.5", "0.1", "1.6"),
            account("USD", "1000.00", "0.00", "1000.00"),
        ]
    }
    balances = portfolio_service.list_balances()
    btc_balance = next(balance for balance in balances if balance.asset == "BTC")
    assert (
        btc_balance.total == Decimal("1.6")
        and btc_balance.available == Decimal("1.5")
        and btc_balance.hold == Decimal("0.1")
    )
    mock_client.get_accounts.return_value = {
        "accounts": [account("BTC", "invalid", "0.1", "1.6", raw_available=True)]
    }
    assert portfolio_service.list_balances() == []
    mock_client.get_accounts.return_value = {"accounts": []}
    list_balances_mock = Mock(return_value=[Mock(spec=Balance)])
    monkeypatch.setattr(portfolio_service, "list_balances", list_balances_mock)
    portfolio_service.get_portfolio_balances()
    list_balances_mock.assert_called_once()
    mock_endpoints.supports_derivatives.return_value = True
    mock_client.list_positions.return_value = [
        position(BTC_SYMBOL, "long", "1.5", "50000.00", "51000.00", "750.00", "0.00", 5)
    ]
    assert portfolio_service.list_positions()[0].symbol == BTC_SYMBOL
    mock_client.list_positions.side_effect = Exception("API error")
    assert portfolio_service.list_positions() == []
    mock_client.get_cfm_position.return_value = {
        "product_id": BTC_SYMBOL,
        "side": "long",
        "size": "1.0",
        "entry_price": "50000.00",
    }
    assert portfolio_service.get_position(BTC_SYMBOL) is not None
    mock_client.get_cfm_position.side_effect = Exception("Position not found")
    assert portfolio_service.get_position(BTC_SYMBOL) is None
    mock_endpoints.mode = "advanced"
    mock_client.intx_allocate.return_value = {"allocation_id": "alloc_123"}
    assert (
        portfolio_service.intx_allocate({"amount": "1000", "currency": "USD"})["allocation_id"]
        == "alloc_123"
    )
    portfolio_service._event_store.append_metric.assert_called()
    mock_client.intx_allocate.side_effect = Exception("Allocation failed")
    with pytest.raises(Exception, match="Allocation failed"):
        portfolio_service.intx_allocate({"amount": "1000"})
    mock_client.cfm_balance_summary.return_value = balance_summary("10000.00", "9500.00")
    summary = portfolio_service.get_cfm_balance_summary()
    assert summary["total_balance"] == Decimal("10000.00")
    mock_client.cfm_sweeps.return_value = {
        "sweeps": [
            sweep("sweep_1", "100.00", "completed"),
            sweep("sweep_2", "200.00", "pending"),
        ]
    }
    sweeps = portfolio_service.list_cfm_sweeps()
    assert sweeps[0]["amount"] == Decimal("100.00")
    mock_client.cfm_intraday_margin_setting.return_value = {
        "margin_window": "maintenance",
        "effective_time": TIMESTAMP,
    }
    result = portfolio_service.update_cfm_margin_window("maintenance")
    assert result["margin_window"] == "maintenance"
    assert portfolio_service._event_store.append_metric.call_count >= 3
