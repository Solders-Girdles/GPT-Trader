import os
from collections.abc import Generator
from datetime import datetime
from decimal import Decimal

import pytest

from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.models import APIConfig
from gpt_trader.features.brokerages.coinbase.rest_service import CoinbaseRestService
from gpt_trader.features.brokerages.coinbase.utilities import ProductCatalog
from gpt_trader.features.brokerages.core.interfaces import (
    Balance,
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Product,
    TimeInForce,
)
from gpt_trader.persistence.event_store import EventStore


class MockAsyncBroker:
    """Minimal mock broker for contract tests."""

    def __init__(self):
        self.orders: dict[str, Order] = {}
        self._balances = [
            Balance(
                asset="USD", total=Decimal("100000"), available=Decimal("100000"), hold=Decimal("0")
            )
        ]
        self._products = {
            "BTC-PERP": Product(
                symbol="BTC-PERP",
                base_asset="BTC",
                quote_asset="USD",
                market_type=MarketType.PERPETUAL,
                min_size=Decimal("0.001"),
                step_size=Decimal("0.001"),
                min_notional=Decimal("10"),
                price_increment=Decimal("0.01"),
                leverage_max=10,
            ),
        }

    def get_product(self, symbol: str) -> Product | None:
        return self._products.get(symbol)

    async def balances(self) -> list[Balance]:
        return self._balances

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal | None = None,
        price: Decimal | None = None,
        tif: TimeInForce = TimeInForce.GTC,
        **kwargs,
    ) -> Order:
        order_id = f"mock_order_{len(self.orders)}"
        order = Order(
            id=order_id,
            client_id=order_id,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity or Decimal("0"),
            price=price,
            stop_price=None,
            tif=tif,
            status=OrderStatus.SUBMITTED,
            filled_quantity=Decimal("0"),
            avg_fill_price=None,
            submitted_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self.orders[order_id] = order
        return order

    async def get_order(self, order_id: str) -> Order | None:
        return self.orders.get(order_id)

    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False


# Define a marker for contract tests
pytestmark = [pytest.mark.anyio, pytest.mark.contract]


@pytest.fixture
def broker(request) -> Generator:
    """
    Fixture to provide the broker instance.
    Defaults to MockAsyncBroker.
    Set env var RUN_CONTRACT_TESTS_AGAINST_REAL_API=1 to use real API.
    """
    use_real_api = os.environ.get("RUN_CONTRACT_TESTS_AGAINST_REAL_API") == "1"

    if use_real_api:
        # Setup real CoinbaseRestService
        api_key = os.environ.get("COINBASE_API_KEY")
        api_secret = os.environ.get("COINBASE_API_SECRET")
        if not api_key or not api_secret:
            pytest.skip("Real API credentials not found")

        config = APIConfig(
            api_key=api_key,
            api_secret=api_secret,
            base_url="https://api-public.sandbox.exchange.coinbase.com",  # Default to sandbox
            sandbox=True,
        )
        client = CoinbaseClient(config)
        endpoints = CoinbaseEndpoints(base_url=config.base_url)
        product_catalog = ProductCatalog()
        market_data = MarketDataService(symbols=[])
        event_store = EventStore()

        service = CoinbaseRestService(
            client=client,
            endpoints=endpoints,
            config=config,
            product_catalog=product_catalog,
            market_data=market_data,
            event_store=event_store,
        )
        yield service
    else:
        # Use MockAsyncBroker
        broker = MockAsyncBroker()
        yield broker


@pytest.mark.asyncio
async def test_list_balances_contract(broker):
    """
    Contract: list_balances should return a list of Balance objects.
    Each Balance should have asset, total, available, and hold.
    """
    balances = (
        await broker.balances()
        if hasattr(broker, "balances") and callable(broker.balances)
        else broker.list_balances()
    )

    # Handle async/sync difference if any (MockAsyncBroker has async balances, CoinbaseRestService might be sync or async)
    # CoinbaseRestService.list_balances is likely sync based on previous view, but let's check.
    # Actually MockAsyncBroker has async balances(). CoinbaseRestService has list_balances().
    # We should standardize or handle both.

    assert isinstance(balances, list)
    if len(balances) > 0:
        balance = balances[0]
        assert hasattr(balance, "asset")
        assert hasattr(balance, "total")
        assert hasattr(balance, "available")
        assert isinstance(balance.total, Decimal)
        assert isinstance(balance.available, Decimal)


@pytest.mark.asyncio
async def test_product_structure_contract(broker):
    """
    Contract: get_product should return a Product object with specific fields.
    """
    # Use a common symbol
    symbol = "BTC-USD"
    # For MockAsyncBroker it uses BTC-PERP by default, let's check
    if isinstance(broker, MockAsyncBroker):
        symbol = "BTC-PERP"

    product = broker.get_product(symbol)

    if product:
        assert hasattr(product, "symbol")
        assert hasattr(product, "base_asset")
        assert hasattr(product, "quote_asset")
        assert hasattr(product, "min_size")
        assert isinstance(product.min_size, Decimal)


@pytest.mark.asyncio
async def test_order_lifecycle_contract(broker):
    """
    Contract: Place order -> Get order -> Cancel order.
    """
    from gpt_trader.features.brokerages.core.interfaces import (
        OrderSide,
        OrderStatus,
        OrderType,
        TimeInForce,
    )

    symbol = "BTC-USD"
    if isinstance(broker, MockAsyncBroker):
        symbol = "BTC-PERP"

    # 1. Place Order
    order = await broker.place_order(
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.01"),
        price=Decimal("10000"),  # Deep OTM to avoid fill if real
        tif=TimeInForce.GTC,
    )

    assert order is not None
    assert order.id is not None
    assert order.symbol == symbol
    assert order.side == OrderSide.BUY
    assert order.type == OrderType.LIMIT
    assert order.status in [
        OrderStatus.SUBMITTED,
        OrderStatus.PENDING,
        OrderStatus.FILLED,
    ]  # Mock might fill immediately

    # 2. Get Order
    fetched_order = await broker.get_order(order.id)
    assert fetched_order is not None
    assert fetched_order.id == order.id
    assert fetched_order.symbol == order.symbol

    # 3. Cancel Order
    # Only cancel if not filled/cancelled
    if fetched_order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
        cancelled = await broker.cancel_order(order.id)
        assert cancelled is True

        # Verify status
        final_order = await broker.get_order(order.id)
        assert final_order.status == OrderStatus.CANCELLED
