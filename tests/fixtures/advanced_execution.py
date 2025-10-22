"""Fixtures for advanced execution testing."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

import pytest

from bot_v2.features.brokerages.core.interfaces import (
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Product,
    Quote,
    TimeInForce,
)

__all__ = [
    "high_liquidity_market_data",
    "low_liquidity_market_data",
    "volatile_market_data",
    "market_order_fixture",
    "limit_order_fixture",
    "stop_order_fixture",
    "stop_limit_order_fixture",
    "ioc_order_fixture",
    "failing_broker_fixture",
    "network_error_broker_fixture",
    "rate_limited_broker_fixture",
]


@pytest.fixture
def high_liquidity_market_data() -> dict[str, Any]:
    """Market data with high liquidity and tight spreads."""
    return {
        "symbol": "BTC-PERP",
        "bid": Decimal("100.00"),
        "ask": Decimal("100.01"),
        "last": Decimal("100.005"),
        "ts": datetime.utcnow(),
        "depth_l1": Decimal("1000000"),  # $1M at best bid/ask
        "depth_l10": Decimal("10000000"),  # $10M within 10 levels
        "spread_bps": Decimal("1"),  # 0.01% spread
        "volume_24h": Decimal("100000000"),  # $100M daily volume
    }


@pytest.fixture
def low_liquidity_market_data() -> dict[str, Any]:
    """Market data with low liquidity and wide spreads."""
    return {
        "symbol": "ALT-PERP",
        "bid": Decimal("10.00"),
        "ask": Decimal("10.50"),
        "last": Decimal("10.25"),
        "ts": datetime.utcnow(),
        "depth_l1": Decimal("1000"),  # $1K at best bid/ask
        "depth_l10": Decimal("5000"),  # $5K within 10 levels
        "spread_bps": Decimal("500"),  # 5% spread
        "volume_24h": Decimal("100000"),  # $100K daily volume
    }


@pytest.fixture
def volatile_market_data() -> dict[str, Any]:
    """Market data with high volatility."""
    return {
        "symbol": "MEME-PERP",
        "bid": Decimal("1.00"),
        "ask": Decimal("1.10"),
        "last": Decimal("1.05"),
        "ts": datetime.utcnow(),
        "depth_l1": Decimal("50000"),  # $50K at best bid/ask
        "depth_l10": Decimal("200000"),  # $200K within 10 levels
        "spread_bps": Decimal("1000"),  # 10% spread
        "volume_24h": Decimal("5000000"),  # $5M daily volume
        "volatility_24h": Decimal("0.15"),  # 15% daily volatility
    }


@pytest.fixture
def market_order_fixture() -> dict[str, Any]:
    """Market order fixture."""
    return {
        "symbol": "BTC-PERP",
        "side": OrderSide.BUY,
        "quantity": Decimal("0.1"),
        "order_type": OrderType.MARKET,
        "time_in_force": TimeInForce.GTC,
        "reduce_only": False,
        "post_only": False,
    }


@pytest.fixture
def limit_order_fixture() -> dict[str, Any]:
    """Limit order fixture."""
    return {
        "symbol": "BTC-PERP",
        "side": OrderSide.BUY,
        "quantity": Decimal("0.1"),
        "order_type": OrderType.LIMIT,
        "limit_price": Decimal("99.50"),
        "time_in_force": TimeInForce.GTC,
        "reduce_only": False,
        "post_only": True,
    }


@pytest.fixture
def stop_order_fixture() -> dict[str, Any]:
    """Stop order fixture."""
    return {
        "symbol": "BTC-PERP",
        "side": OrderSide.SELL,
        "quantity": Decimal("0.1"),
        "order_type": OrderType.STOP,
        "stop_price": Decimal("95.00"),
        "time_in_force": TimeInForce.GTC,
        "reduce_only": True,
    }


@pytest.fixture
def stop_limit_order_fixture() -> dict[str, Any]:
    """Stop-limit order fixture."""
    return {
        "symbol": "BTC-PERP",
        "side": OrderSide.SELL,
        "quantity": Decimal("0.1"),
        "order_type": OrderType.STOP_LIMIT,
        "stop_price": Decimal("95.00"),
        "limit_price": Decimal("94.50"),
        "time_in_force": TimeInForce.GTC,
        "reduce_only": True,
    }


@pytest.fixture
def ioc_order_fixture() -> dict[str, Any]:
    """IOC (Immediate-or-Cancel) order fixture."""
    return {
        "symbol": "BTC-PERP",
        "side": OrderSide.BUY,
        "quantity": Decimal("0.1"),
        "order_type": OrderType.LIMIT,
        "limit_price": Decimal("100.00"),
        "time_in_force": TimeInForce.IOC,
        "reduce_only": False,
        "post_only": False,
    }


class FailingBroker:
    """Broker that simulates various failure scenarios."""

    def __init__(
        self,
        fail_order_types: set[str] | None = None,
        fail_after_count: int | None = None,
        network_error: bool = False,
        rate_limit: bool = False,
    ):
        self.fail_order_types = fail_order_types or set()
        self.fail_after_count = fail_after_count
        self.network_error = network_error
        self.rate_limit = rate_limit
        self.orders = []
        self.order_count = 0

    def get_product(self, symbol: str) -> Product:
        return Product(
            symbol=symbol,
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
            leverage_max=5,
            contract_size=Decimal("1"),
            funding_rate=Decimal("0.0001"),
            next_funding_time=None,
        )

    def get_quote(self, symbol: str) -> Quote:
        return Quote(
            symbol=symbol,
            bid=Decimal("100"),
            ask=Decimal("101"),
            last=Decimal("100.5"),
            ts=datetime.utcnow(),
        )

    def place_order(self, **kwargs) -> Order:
        self.order_count += 1

        # Check for failure conditions
        if self.fail_after_count and self.order_count > self.fail_after_count:
            if self.network_error:
                raise ConnectionError("Network timeout")
            elif self.rate_limit:
                raise RuntimeError("Rate limit exceeded")
            else:
                raise RuntimeError("Broker failure")

        order_type = kwargs.get("order_type")
        if isinstance(order_type, OrderType):
            order_type_key = order_type.value
        else:
            order_type_key = str(order_type)

        if order_type_key.lower() in self.fail_order_types:
            raise RuntimeError("Simulated broker failure")

        order_id = f"order-{len(self.orders) + 1}"
        now = datetime.utcnow()
        order_quantity = kwargs.get("quantity", Decimal("0.01"))
        order = Order(
            id=order_id,
            client_id=kwargs.get("client_id", order_id),
            symbol=kwargs.get("symbol", "BTC-PERP"),
            side=kwargs.get("side", OrderSide.BUY),
            type=kwargs.get("order_type", OrderType.MARKET),
            quantity=order_quantity,
            price=kwargs.get("price"),
            stop_price=kwargs.get("stop_price"),
            tif=kwargs.get("tif", TimeInForce.GTC),
            status=OrderStatus.SUBMITTED,
            filled_quantity=Decimal("0"),
            avg_fill_price=None,
            submitted_at=now,
            updated_at=now,
        )
        order.reduce_only = kwargs.get("reduce_only", False)
        self.orders.append(order)
        return order


@pytest.fixture
def failing_broker_fixture() -> FailingBroker:
    """Broker that fails on specific order types."""
    return FailingBroker(fail_order_types={"stop", "stop_limit"})


@pytest.fixture
def network_error_broker_fixture() -> FailingBroker:
    """Broker that simulates network errors after a few orders."""
    return FailingBroker(fail_after_count=2, network_error=True)


@pytest.fixture
def rate_limited_broker_fixture() -> FailingBroker:
    """Broker that simulates rate limiting."""
    return FailingBroker(rate_limit=True)
