"""
Shared mock broker implementations for standardized testing.

This module provides common mock broker implementations to be used across
the test suite, promoting consistency and reducing code duplication.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from gpt_trader.features.brokerages.core.interfaces import (
    Balance,
    Candle,
    IBrokerage,
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Product,
    Quote,
    TimeInForce,
)


class MockAsyncBroker(IBrokerage):
    """
    A mock broker implementation with async methods properly mocked using AsyncMock.

    This broker is designed for testing async functionality and provides
    realistic mock behavior for all async methods.
    """

    def __init__(self, failure_mode: str | None = None):
        self.failure_mode = failure_mode
        self.orders: dict[str, Order] = {}
        self.positions: dict[str, Position] = {}
        self.balances: list[Balance] = [
            Balance(
                asset="USD", total=Decimal("100000"), available=Decimal("100000"), hold=Decimal("0")
            )
        ]
        self.connected = False
        self.products: dict[str, Product] = {}
        self._setup_default_products()

    def _setup_default_products(self):
        """Setup default product catalog."""
        self.products = {
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
            "ETH-PERP": Product(
                symbol="ETH-PERP",
                base_asset="ETH",
                quote_asset="USD",
                market_type=MarketType.PERPETUAL,
                min_size=Decimal("0.001"),
                step_size=Decimal("0.001"),
                min_notional=Decimal("10"),
                price_increment=Decimal("0.01"),
                leverage_max=10,
            ),
        }

    # Connection methods
    def connect(self) -> bool:
        """Connect to the broker."""
        self.connected = True
        return True

    def disconnect(self) -> None:
        """Disconnect from the broker."""
        self.connected = False

    def validate_connection(self) -> bool:
        """Validate connection status."""
        return self.connected

    def get_account_id(self) -> str:
        """Get account ID."""
        return "MOCK_ASYNC_BROKER"

    # Account methods
    def list_balances(self) -> list[Balance]:
        """List account balances."""
        return self.balances

    # Product methods
    def list_products(self, market: MarketType | None = None) -> list[Product]:
        """List available products."""
        products = list(self.products.values())
        if market is None:
            return products
        return [p for p in products if p.market_type == market]

    def get_product(self, symbol: str) -> Product:
        """Get product details."""
        return self.products.get(symbol, self.products.get("BTC-PERP"))

    # Market data methods
    def get_quote(self, symbol: str) -> Quote:
        """Get market quote."""
        return Quote(
            symbol=symbol,
            bid=Decimal("50000"),
            ask=Decimal("50100"),
            last=Decimal("50050"),
            ts=datetime.utcnow(),
        )

    def get_candles(
        self,
        symbol: str,
        granularity: str,
        limit: int = 200,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Candle]:
        """Get candle data."""
        return []

    # Async order methods
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal | None = None,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        tif: TimeInForce = TimeInForce.GTC,
        client_id: str | None = None,
        reduce_only: bool | None = None,
        leverage: int | None = None,
    ) -> Order:
        """Place an order asynchronously."""
        if self.failure_mode == "place_order_failure":
            raise Exception("Failed to place order")

        order_id = client_id or f"mock_order_{len(self.orders)}"
        now = datetime.utcnow()

        order = Order(
            id=order_id,
            client_id=client_id,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity or Decimal("0"),
            price=price,
            stop_price=stop_price,
            tif=tif,
            status=OrderStatus.SUBMITTED,
            filled_quantity=Decimal("0"),
            avg_fill_price=None,
            submitted_at=now,
            updated_at=now,
        )

        self.orders[order_id] = order
        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order asynchronously."""
        if self.failure_mode == "cancel_order_failure":
            raise Exception("Failed to cancel order")

        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    async def get_order(self, order_id: str) -> Order | None:
        """Get order details asynchronously."""
        if self.failure_mode == "get_order_failure":
            raise Exception("Failed to get order")

        return self.orders.get(order_id)

    async def list_orders(
        self,
        status: OrderStatus | None = None,
        symbol: str | None = None,
    ) -> list[Order]:
        """List orders asynchronously."""
        if self.failure_mode == "list_orders_failure":
            raise Exception("Failed to list orders")

        orders = list(self.orders.values())
        if status is not None:
            orders = [o for o in orders if o.status == status]
        if symbol is not None:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    # Async position methods
    async def positions(self) -> list[Position]:
        """Get positions asynchronously."""
        if self.failure_mode == "positions_failure":
            raise Exception("Failed to get positions")

        return list(self.positions.values())

    async def position(self, symbol: str) -> Position | None:
        """Get position for a symbol asynchronously."""
        if self.failure_mode == "position_failure":
            raise Exception("Failed to get position")

        return self.positions.get(symbol)

    # Other async methods
    async def balances(self) -> list[Balance]:
        """Get balances asynchronously."""
        if self.failure_mode == "balances_failure":
            raise Exception("Failed to get balances")

        return self.balances

    async def balance(self, currency: str) -> Balance | None:
        """Get balance for a currency asynchronously."""
        if self.failure_mode == "balance_failure":
            raise Exception("Failed to get balance")

        for bal in self.balances:
            if bal.asset == currency:
                return bal
        return None

    async def equity(self) -> Decimal:
        """Get account equity asynchronously."""
        if self.failure_mode == "equity_failure":
            raise Exception("Failed to get equity")

        return sum(bal.total for bal in self.balances)

    async def margin_info(self) -> dict[str, Decimal]:
        """Get margin information asynchronously."""
        if self.failure_mode == "margin_info_failure":
            raise Exception("Failed to get margin info")

        return {
            "initial_margin": Decimal("0"),
            "maintenance_margin": Decimal("0"),
            "available_margin": Decimal("100000"),
        }

    async def fills(
        self,
        symbol: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """Get fills asynchronously."""
        if self.failure_mode == "fills_failure":
            raise Exception("Failed to get fills")

        return []

    # Streaming methods
    def stream_trades(self, symbols: list[str]) -> Any:
        """Stream trades."""
        return iter([])

    def stream_orderbook(self, symbols: list[str], level: int = 1) -> Any:
        """Stream orderbook."""
        return iter([])

    # Test helper methods
    def add_position(self, symbol: str, quantity: Decimal, side: str):
        """Add a test position."""
        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            leverage=None,
            side=side,
        )

    def add_balance(self, asset: str, total: Decimal, available: Decimal | None = None):
        """Add a test balance."""
        if available is None:
            available = total
        self.balances.append(
            Balance(asset=asset, total=total, available=available, hold=Decimal("0"))
        )

    def set_failure_mode(self, mode: str | None):
        """Set the failure mode for testing."""
        self.failure_mode = mode


class MockBrokerWithAsyncMethods:
    """
    A MagicMock-based mock broker with async methods properly configured.

    This is useful when you need more control over mock behavior
    and want to use MagicMock's features like side_effect, return_value, etc.
    """

    def __init__(self):
        # Sync methods
        self.connect = MagicMock(return_value=True)
        self.disconnect = MagicMock()
        self.validate_connection = MagicMock(return_value=True)
        self.get_account_id = MagicMock(return_value="MOCK_BROKER")
        self.list_balances = MagicMock(
            return_value=[
                Balance(
                    asset="USD",
                    total=Decimal("100000"),
                    available=Decimal("100000"),
                    hold=Decimal("0"),
                )
            ]
        )
        self.list_products = MagicMock(return_value=[])
        self.get_product = MagicMock()
        self.get_quote = MagicMock()
        self.get_candles = MagicMock(return_value=[])
        self.stream_trades = MagicMock(return_value=iter([]))
        self.stream_orderbook = MagicMock(return_value=iter([]))

        # Async methods - properly mocked with AsyncMock
        self.place_order = AsyncMock()
        self.cancel_order = AsyncMock()
        self.get_order = AsyncMock()
        self.list_orders = AsyncMock()
        self.positions = AsyncMock(return_value=[])
        self.position = AsyncMock()
        self.balances = AsyncMock(
            return_value=[
                Balance(
                    asset="USD",
                    total=Decimal("100000"),
                    available=Decimal("100000"),
                    hold=Decimal("0"),
                )
            ]
        )
        self.balance = AsyncMock()
        self.equity = AsyncMock(return_value=Decimal("100000"))
        self.margin_info = AsyncMock(
            return_value={
                "initial_margin": Decimal("0"),
                "maintenance_margin": Decimal("0"),
                "available_margin": Decimal("100000"),
            }
        )
        self.fills = AsyncMock(return_value=[])

    def configure_place_order_success(self, order_id: str = "test_order"):
        """Configure place_order to return a successful order."""
        from gpt_trader.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce

        test_order = Order(
            id=order_id,
            client_id=order_id,
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            stop_price=None,
            tif=TimeInForce.GTC,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("1.0"),
            avg_fill_price=Decimal("50000"),
            submitted_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self.place_order.return_value = test_order
        return test_order

    def configure_place_order_failure(self, error: Exception):
        """Configure place_order to raise an exception."""
        self.place_order.side_effect = error

    def configure_cancel_order_success(self):
        """Configure cancel_order to return True."""
        self.cancel_order.return_value = True

    def configure_cancel_order_failure(self, error: Exception):
        """Configure cancel_order to raise an exception."""
        self.cancel_order.side_effect = error

    def configure_positions(self, positions: list[Position]):
        """Configure positions to return a specific list."""
        self.positions.return_value = positions

    def configure_balances(self, balances: list[Balance]):
        """Configure balances to return a specific list."""
        self.balances.return_value = balances


def create_mock_async_broker(failure_mode: str | None = None) -> MockAsyncBroker:
    """
    Create a MockAsyncBroker instance.

    Args:
        failure_mode: Optional failure mode for testing error scenarios

    Returns:
        MockAsyncBroker instance
    """
    return MockAsyncBroker(failure_mode=failure_mode)


def create_mock_broker_with_async_methods() -> MockBrokerWithAsyncMethods:
    """
    Create a MockBrokerWithAsyncMethods instance.

    Returns:
        MockBrokerWithAsyncMethods instance
    """
    return MockBrokerWithAsyncMethods()


# Common mock broker configurations
def create_success_mock_broker() -> MockBrokerWithAsyncMethods:
    """
    Create a mock broker configured for successful operations.

    Returns:
        MockBrokerWithAsyncMethods configured for success
    """
    broker = MockBrokerWithAsyncMethods()
    broker.configure_place_order_success()
    broker.configure_cancel_order_success()
    return broker


def create_failure_mock_broker() -> MockBrokerWithAsyncMethods:
    """
    Create a mock broker configured for failed operations.

    Returns:
        MockBrokerWithAsyncMethods configured for failure
    """
    broker = MockBrokerWithAsyncMethods()
    broker.configure_place_order_failure(Exception("Order placement failed"))
    broker.configure_cancel_order_failure(Exception("Order cancellation failed"))
    return broker
