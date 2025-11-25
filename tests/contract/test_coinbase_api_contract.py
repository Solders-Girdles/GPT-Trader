import os
import pytest
from decimal import Decimal
from typing import Generator

from gpt_trader.features.brokerages.coinbase.rest_service import CoinbaseRestService
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
from gpt_trader.features.brokerages.coinbase.models import APIConfig
from gpt_trader.features.brokerages.coinbase.utilities import ProductCatalog
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.persistence.event_store import EventStore
from tests.shared.mock_brokers import MockAsyncBroker

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
            base_url="https://api-public.sandbox.exchange.coinbase.com", # Default to sandbox
            sandbox=True
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
            event_store=event_store
        )
        yield service
    else:
        # Use MockAsyncBroker
        broker = MockAsyncBroker()
        yield broker

async def test_list_balances_contract(broker):
    """
    Contract: list_balances should return a list of Balance objects.
    Each Balance should have asset, total, available, and hold.
    """
    balances = await broker.balances() if hasattr(broker, "balances") and callable(broker.balances) else broker.list_balances()
    
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

async def test_order_lifecycle_contract(broker):
    """
    Contract: Place order -> Get order -> Cancel order.
    """
    from gpt_trader.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce, OrderStatus
    
    symbol = "BTC-USD"
    if isinstance(broker, MockAsyncBroker):
        symbol = "BTC-PERP"

    # 1. Place Order
    order = await broker.place_order(
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.01"),
        price=Decimal("10000"), # Deep OTM to avoid fill if real
        tif=TimeInForce.GTC
    )
    
    assert order is not None
    assert order.id is not None
    assert order.symbol == symbol
    assert order.side == OrderSide.BUY
    assert order.type == OrderType.LIMIT
    assert order.status in [OrderStatus.SUBMITTED, OrderStatus.PENDING, OrderStatus.FILLED] # Mock might fill immediately

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

