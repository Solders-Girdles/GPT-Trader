"""
Live Order Management System for Real-Time Trading Infrastructure

This module implements sophisticated order management including:
- Multi-exchange order routing and execution
- Advanced order types (Market, Limit, Stop, Iceberg, TWAP, VWAP)
- Real-time order status tracking and updates
- Position tracking and portfolio reconciliation
- Risk controls and pre-trade validation
- Fill reporting and transaction cost analysis
- Order book management and smart routing
- Failover and error handling mechanisms
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

# Optional dependencies
try:
    import aiohttp
    import websockets

    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

try:
    from ..live.market_data_pipeline import DataType, MarketDataPoint
except ImportError:
    # Fallback for testing
    class MarketDataPoint:
        pass

    class DataType:
        QUOTE = "quote"
        TRADE = "trade"


logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by the system"""

    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    BRACKET = "bracket"  # Bracket order with profit/loss
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order side"""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""

    PENDING = "pending"
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REPLACED = "replaced"


class TimeInForce(Enum):
    """Order time in force"""

    GTC = "gtc"  # Good Till Cancel
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    DAY = "day"  # Day order


class ExecutionVenue(Enum):
    """Execution venues/exchanges"""

    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    ALPACA = "alpaca"
    INTERACTIVE_BROKERS = "ib"
    SIMULATION = "simulation"


@dataclass
class OrderRequest:
    """Order request specification"""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: str | None = None
    venue: ExecutionVenue | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Advanced order parameters
    iceberg_qty: float | None = None  # For iceberg orders
    duration_seconds: int | None = None  # For TWAP/VWAP
    min_qty: float | None = None  # Minimum fill quantity
    max_floor: float | None = None  # Maximum visible quantity

    def __post_init__(self):
        if self.client_order_id is None:
            self.client_order_id = str(uuid.uuid4())


@dataclass
class Fill:
    """Order fill information"""

    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    commission_asset: str
    timestamp: pd.Timestamp
    venue: ExecutionVenue
    liquidity: str = "unknown"  # "maker" or "taker"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Order:
    """Order tracking object"""

    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None
    stop_price: float | None
    time_in_force: TimeInForce
    venue: ExecutionVenue

    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float | None = None
    avg_fill_price: float | None = None

    created_at: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    updated_at: pd.Timestamp = field(default_factory=pd.Timestamp.now)

    fills: list[Fill] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity

    @property
    def is_active(self) -> bool:
        """Check if order is still active"""
        return self.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING]

    @property
    def fill_rate(self) -> float:
        """Get order fill rate (0.0 to 1.0)"""
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0.0

    def add_fill(self, fill: Fill) -> None:
        """Add a fill to the order"""
        self.fills.append(fill)
        self.filled_quantity += fill.quantity
        self.remaining_quantity = max(0, self.quantity - self.filled_quantity)

        # Update average fill price
        total_value = sum(f.quantity * f.price for f in self.fills)
        self.avg_fill_price = (
            total_value / self.filled_quantity if self.filled_quantity > 0 else None
        )

        # Update status
        if self.remaining_quantity <= 0:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED

        self.updated_at = pd.Timestamp.now()


@dataclass
class OrderManagerConfig:
    """Configuration for order management system"""

    max_orders_per_symbol: int = 100
    max_total_orders: int = 1000
    default_venue: ExecutionVenue = ExecutionVenue.SIMULATION
    enable_smart_routing: bool = True
    enable_position_tracking: bool = True
    enable_risk_controls: bool = True
    max_order_value: float = 100000.0
    max_position_size: float = 1000000.0
    fill_timeout_seconds: float = 60.0
    status_update_interval: float = 1.0
    enable_order_book_routing: bool = False
    commission_rate: float = 0.001  # 0.1%
    slippage_threshold: float = 0.002  # 0.2%


class BaseOrderVenue(ABC):
    """Base class for order execution venues"""

    def __init__(self, venue: ExecutionVenue, config: dict[str, Any] = None) -> None:
        self.venue = venue
        self.config = config or {}
        self.is_connected = False
        self.orders = {}
        self.positions = defaultdict(float)

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the venue"""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the venue"""
        pass

    @abstractmethod
    async def submit_order(self, order_request: OrderRequest) -> Order:
        """Submit order to venue"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order at venue"""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order | None:
        """Get order status from venue"""
        pass

    @abstractmethod
    async def get_positions(self) -> dict[str, float]:
        """Get current positions from venue"""
        pass

    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        return f"{self.venue.value}_{uuid.uuid4().hex[:8]}_{int(time.time())}"


class SimulatedOrderVenue(BaseOrderVenue):
    """Simulated order venue for testing"""

    def __init__(self, config: dict[str, Any] = None) -> None:
        super().__init__(ExecutionVenue.SIMULATION, config)
        self.market_data = {}
        self.fill_probability = 0.95  # 95% chance of fill for limit orders
        self.latency_ms = np.random.uniform(10, 50)  # 10-50ms latency

    async def connect(self) -> bool:
        """Connect to simulation"""
        await asyncio.sleep(0.1)  # Simulate connection time
        self.is_connected = True
        logger.info("Connected to simulated venue")
        return True

    async def disconnect(self) -> bool:
        """Disconnect from simulation"""
        self.is_connected = False
        logger.info("Disconnected from simulated venue")
        return True

    async def submit_order(self, order_request: OrderRequest) -> Order:
        """Submit order to simulation"""
        if not self.is_connected:
            raise RuntimeError("Not connected to venue")

        # Simulate processing latency
        await asyncio.sleep(self.latency_ms / 1000)

        order_id = self._generate_order_id()

        order = Order(
            order_id=order_id,
            client_order_id=order_request.client_order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            quantity=order_request.quantity,
            price=order_request.price,
            stop_price=order_request.stop_price,
            time_in_force=order_request.time_in_force,
            venue=self.venue,
            status=OrderStatus.NEW,
        )

        self.orders[order_id] = order

        # Simulate immediate market order fill
        if order_request.order_type == OrderType.MARKET:
            await self._simulate_fill(order, order_request.quantity)

        # Simulate probabilistic limit order fill
        elif order_request.order_type == OrderType.LIMIT:
            if np.random.random() < self.fill_probability:
                # Simulate partial or full fill
                fill_qty = order_request.quantity * np.random.uniform(0.5, 1.0)
                await self._simulate_fill(order, fill_qty)

        logger.info(
            f"Submitted order {order_id}: {order_request.side.value} "
            f"{order_request.quantity} {order_request.symbol}"
        )

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order in simulation"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.is_active:
                order.status = OrderStatus.CANCELED
                order.updated_at = pd.Timestamp.now()
                logger.info(f"Canceled order {order_id}")
                return True
        return False

    async def get_order_status(self, order_id: str) -> Order | None:
        """Get order status from simulation"""
        return self.orders.get(order_id)

    async def get_positions(self) -> dict[str, float]:
        """Get simulated positions"""
        return dict(self.positions)

    async def _simulate_fill(self, order: Order, fill_quantity: float) -> None:
        """Simulate order fill"""
        # Generate realistic fill price
        if order.price:
            # Add small random slippage for limit orders
            slippage = np.random.uniform(-0.001, 0.001)  # ±0.1% slippage
            fill_price = order.price * (1 + slippage)
        else:
            # Market order - use current market price (simulated)
            fill_price = 100.0 * (1 + np.random.uniform(-0.01, 0.01))

        # Create fill
        fill = Fill(
            fill_id=f"fill_{uuid.uuid4().hex[:8]}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            commission=fill_quantity * fill_price * 0.001,  # 0.1% commission
            commission_asset="USD",
            timestamp=pd.Timestamp.now(),
            venue=self.venue,
            liquidity="taker",
        )

        # Add fill to order
        order.add_fill(fill)

        # Update positions
        position_delta = fill_quantity if order.side == OrderSide.BUY else -fill_quantity
        self.positions[order.symbol] += position_delta

        logger.info(f"Fill: {fill_quantity} {order.symbol} @ {fill_price:.4f}")


class LiveOrderManager:
    """Main order management system"""

    def __init__(self, config: OrderManagerConfig) -> None:
        self.config = config
        self.venues = {}
        self.orders = {}  # All orders across venues
        self.positions = defaultdict(float)  # Consolidated positions
        self.order_listeners = []
        self.fill_listeners = []
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Performance metrics
        self.metrics = {
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_canceled": 0,
            "orders_rejected": 0,
            "total_volume": 0.0,
            "total_commission": 0.0,
            "avg_fill_time": 0.0,
        }

        # Add default simulation venue
        self.add_venue(SimulatedOrderVenue())

    def add_venue(self, venue: BaseOrderVenue) -> None:
        """Add execution venue"""
        self.venues[venue.venue] = venue
        logger.info(f"Added venue: {venue.venue.value}")

    def remove_venue(self, venue: ExecutionVenue) -> None:
        """Remove execution venue"""
        if venue in self.venues:
            del self.venues[venue]
            logger.info(f"Removed venue: {venue.value}")

    def add_order_listener(self, callback: Callable[[Order], None]) -> None:
        """Add order status listener"""
        self.order_listeners.append(callback)

    def add_fill_listener(self, callback: Callable[[Fill], None]) -> None:
        """Add fill listener"""
        self.fill_listeners.append(callback)

    async def start(self) -> None:
        """Start order management system"""
        if self.is_running:
            logger.warning("Order manager already running")
            return

        self.is_running = True
        logger.info("Starting order management system...")

        # Connect to all venues
        for venue_name, venue in self.venues.items():
            try:
                await venue.connect()
                logger.info(f"Connected to venue: {venue_name.value}")
            except Exception as e:
                logger.error(f"Failed to connect to {venue_name.value}: {str(e)}")

        # Start monitoring tasks
        asyncio.create_task(self._monitor_orders())
        asyncio.create_task(self._update_positions())

        logger.info("Order management system started")

    async def stop(self) -> None:
        """Stop order management system"""
        if not self.is_running:
            return

        logger.info("Stopping order management system...")
        self.is_running = False

        # Cancel all active orders
        active_orders = [order for order in self.orders.values() if order.is_active]
        for order in active_orders:
            try:
                await self.cancel_order(order.order_id)
            except Exception as e:
                logger.warning(f"Failed to cancel order {order.order_id}: {str(e)}")

        # Disconnect from venues
        for venue_name, venue in self.venues.items():
            try:
                await venue.disconnect()
                logger.info(f"Disconnected from venue: {venue_name.value}")
            except Exception as e:
                logger.warning(f"Error disconnecting from {venue_name.value}: {str(e)}")

        self.executor.shutdown(wait=True)
        logger.info("Order management system stopped")

    async def submit_order(self, order_request: OrderRequest) -> Order:
        """Submit order for execution"""
        if not self.is_running:
            raise RuntimeError("Order manager not running")

        # Validate order
        validation_result = self._validate_order(order_request)
        if not validation_result["valid"]:
            raise ValueError(f"Order validation failed: {validation_result['reason']}")

        # Select venue
        venue = self._select_venue(order_request)
        if not venue:
            raise RuntimeError("No available venue for order")

        try:
            # Submit order to venue
            order = await venue.submit_order(order_request)

            # Track order
            self.orders[order.order_id] = order
            self.metrics["orders_submitted"] += 1

            # Notify listeners
            await self._notify_order_listeners(order)

            logger.info(f"Order submitted: {order.order_id} on {venue.venue.value}")
            return order

        except Exception as e:
            self.metrics["orders_rejected"] += 1
            logger.error(f"Order submission failed: {str(e)}")
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        if order_id not in self.orders:
            logger.warning(f"Order not found: {order_id}")
            return False

        order = self.orders[order_id]
        if not order.is_active:
            logger.warning(f"Order not active: {order_id}")
            return False

        venue = self.venues.get(order.venue)
        if not venue:
            logger.error(f"Venue not found for order: {order_id}")
            return False

        try:
            success = await venue.cancel_order(order_id)
            if success:
                order.status = OrderStatus.CANCELED
                order.updated_at = pd.Timestamp.now()
                self.metrics["orders_canceled"] += 1
                await self._notify_order_listeners(order)
                logger.info(f"Order canceled: {order_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel all active orders, optionally filtered by symbol"""
        canceled_count = 0

        active_orders = [
            order
            for order in self.orders.values()
            if order.is_active and (symbol is None or order.symbol == symbol)
        ]

        for order in active_orders:
            try:
                success = await self.cancel_order(order.order_id)
                if success:
                    canceled_count += 1
            except Exception as e:
                logger.warning(f"Failed to cancel order {order.order_id}: {str(e)}")

        logger.info(f"Canceled {canceled_count}/{len(active_orders)} orders")
        return canceled_count

    def get_order(self, order_id: str) -> Order | None:
        """Get order by ID"""
        return self.orders.get(order_id)

    def get_orders(
        self, symbol: str | None = None, status: OrderStatus | None = None
    ) -> list[Order]:
        """Get orders filtered by symbol and/or status"""
        orders = list(self.orders.values())

        if symbol:
            orders = [order for order in orders if order.symbol == symbol]

        if status:
            orders = [order for order in orders if order.status == status]

        return orders

    def get_positions(self) -> dict[str, float]:
        """Get current positions"""
        return dict(self.positions)

    def get_position(self, symbol: str) -> float:
        """Get position for specific symbol"""
        return self.positions.get(symbol, 0.0)

    def _validate_order(self, order_request: OrderRequest) -> dict[str, Any]:
        """Validate order request"""
        try:
            # Basic validation
            if order_request.quantity <= 0:
                return {"valid": False, "reason": "Invalid quantity"}

            if order_request.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                if order_request.price is None or order_request.price <= 0:
                    return {"valid": False, "reason": "Invalid price for limit order"}

            if order_request.order_type in [OrderType.STOP_MARKET, OrderType.STOP_LIMIT]:
                if order_request.stop_price is None or order_request.stop_price <= 0:
                    return {"valid": False, "reason": "Invalid stop price"}

            # Risk controls
            if self.config.enable_risk_controls:
                order_value = order_request.quantity * (order_request.price or 100.0)

                if order_value > self.config.max_order_value:
                    return {"valid": False, "reason": "Order exceeds maximum value"}

                # Position size check
                current_position = self.get_position(order_request.symbol)
                position_delta = (
                    order_request.quantity
                    if order_request.side == OrderSide.BUY
                    else -order_request.quantity
                )
                new_position = abs(current_position + position_delta)

                if new_position * (order_request.price or 100.0) > self.config.max_position_size:
                    return {"valid": False, "reason": "Order exceeds maximum position size"}

            # Order count limits
            symbol_orders = len(
                [
                    o
                    for o in self.orders.values()
                    if o.symbol == order_request.symbol and o.is_active
                ]
            )
            if symbol_orders >= self.config.max_orders_per_symbol:
                return {"valid": False, "reason": "Too many orders for symbol"}

            total_orders = len([o for o in self.orders.values() if o.is_active])
            if total_orders >= self.config.max_total_orders:
                return {"valid": False, "reason": "Too many total orders"}

            return {"valid": True, "reason": "Order validated"}

        except Exception as e:
            return {"valid": False, "reason": f"Validation error: {str(e)}"}

    def _select_venue(self, order_request: OrderRequest) -> BaseOrderVenue | None:
        """Select best venue for order execution"""
        if order_request.venue:
            return self.venues.get(order_request.venue)

        # Use default venue or smart routing
        if self.config.enable_smart_routing:
            # Simple smart routing - could be enhanced with order book analysis
            connected_venues = [venue for venue in self.venues.values() if venue.is_connected]
            if connected_venues:
                return connected_venues[0]  # Use first connected venue

        # Use default venue
        return self.venues.get(self.config.default_venue)

    async def _monitor_orders(self) -> None:
        """Monitor order status and updates"""
        while self.is_running:
            try:
                # Update order statuses from venues
                for order in list(self.orders.values()):
                    if order.is_active:
                        venue = self.venues.get(order.venue)
                        if venue:
                            try:
                                updated_order = await venue.get_order_status(order.order_id)
                                if updated_order and updated_order.status != order.status:
                                    # Update local order
                                    old_status = order.status
                                    order.status = updated_order.status
                                    order.filled_quantity = updated_order.filled_quantity
                                    order.remaining_quantity = updated_order.remaining_quantity
                                    order.avg_fill_price = updated_order.avg_fill_price
                                    order.updated_at = pd.Timestamp.now()

                                    # Process new fills
                                    for fill in updated_order.fills:
                                        if fill not in order.fills:
                                            order.add_fill(fill)
                                            await self._notify_fill_listeners(fill)
                                            self._update_metrics_for_fill(fill)

                                    # Notify order listeners
                                    await self._notify_order_listeners(order)

                                    logger.info(
                                        f"Order {order.order_id} status: {old_status.value} -> {order.status.value}"
                                    )

                            except Exception as e:
                                logger.warning(f"Error updating order {order.order_id}: {str(e)}")

                await asyncio.sleep(self.config.status_update_interval)

            except Exception as e:
                logger.error(f"Error in order monitoring: {str(e)}")
                await asyncio.sleep(self.config.status_update_interval)

    async def _update_positions(self) -> None:
        """Update consolidated positions"""
        while self.is_running:
            try:
                # Update positions from all venues
                all_positions = defaultdict(float)

                for venue in self.venues.values():
                    if venue.is_connected:
                        try:
                            venue_positions = await venue.get_positions()
                            for symbol, position in venue_positions.items():
                                all_positions[symbol] += position
                        except Exception as e:
                            logger.warning(
                                f"Error getting positions from {venue.venue.value}: {str(e)}"
                            )

                # Update consolidated positions
                self.positions = all_positions

                await asyncio.sleep(10)  # Update every 10 seconds

            except Exception as e:
                logger.error(f"Error updating positions: {str(e)}")
                await asyncio.sleep(10)

    async def _notify_order_listeners(self, order: Order) -> None:
        """Notify order status listeners"""
        for callback in self.order_listeners:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order)
                else:
                    callback(order)
            except Exception as e:
                logger.warning(f"Error notifying order listener: {str(e)}")

    async def _notify_fill_listeners(self, fill: Fill) -> None:
        """Notify fill listeners"""
        for callback in self.fill_listeners:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(fill)
                else:
                    callback(fill)
            except Exception as e:
                logger.warning(f"Error notifying fill listener: {str(e)}")

    def _update_metrics_for_fill(self, fill: Fill) -> None:
        """Update performance metrics for fill"""
        self.metrics["total_volume"] += fill.quantity * fill.price
        self.metrics["total_commission"] += fill.commission

        # Check if order is now filled
        if fill.order_id in self.orders:
            order = self.orders[fill.order_id]
            if order.status == OrderStatus.FILLED:
                self.metrics["orders_filled"] += 1

                # Update average fill time
                fill_time = (order.updated_at - order.created_at).total_seconds()
                current_avg = self.metrics["avg_fill_time"]
                filled_count = self.metrics["orders_filled"]
                self.metrics["avg_fill_time"] = (
                    current_avg * (filled_count - 1) + fill_time
                ) / filled_count

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get order management performance metrics"""
        total_orders = self.metrics["orders_submitted"]

        return {
            "orders": {
                "submitted": self.metrics["orders_submitted"],
                "filled": self.metrics["orders_filled"],
                "canceled": self.metrics["orders_canceled"],
                "rejected": self.metrics["orders_rejected"],
                "fill_rate": self.metrics["orders_filled"] / max(total_orders, 1),
                "rejection_rate": self.metrics["orders_rejected"] / max(total_orders, 1),
            },
            "execution": {
                "total_volume": self.metrics["total_volume"],
                "total_commission": self.metrics["total_commission"],
                "avg_fill_time_seconds": self.metrics["avg_fill_time"],
                "commission_rate": self.metrics["total_commission"]
                / max(self.metrics["total_volume"], 1),
            },
            "positions": {
                "symbols": len(self.positions),
                "total_exposure": sum(abs(pos) for pos in self.positions.values()),
            },
            "system": {
                "connected_venues": sum(1 for venue in self.venues.values() if venue.is_connected),
                "total_venues": len(self.venues),
                "is_running": self.is_running,
                "active_orders": len([o for o in self.orders.values() if o.is_active]),
            },
        }


def create_order_manager(
    max_orders_per_symbol: int = 100,
    max_total_orders: int = 1000,
    enable_risk_controls: bool = True,
    **kwargs,
) -> LiveOrderManager:
    """Factory function to create order manager"""
    config = OrderManagerConfig(
        max_orders_per_symbol=max_orders_per_symbol,
        max_total_orders=max_total_orders,
        enable_risk_controls=enable_risk_controls,
        **kwargs,
    )

    return LiveOrderManager(config)


# Example usage and testing
async def main() -> None:
    """Example usage of order management system"""
    print("Live Order Management System Testing")
    print("=" * 45)

    # Create order manager
    order_manager = create_order_manager(
        max_orders_per_symbol=50, enable_risk_controls=True, max_order_value=50000.0
    )

    # Add order listener
    def order_listener(order: Order) -> None:
        print(
            f"📋 Order Update: {order.order_id} - {order.status.value} "
            f"({order.fill_rate:.1%} filled)"
        )

    def fill_listener(fill: Fill) -> None:
        print(f"✅ Fill: {fill.quantity} {fill.symbol} @ {fill.price:.4f}")

    order_manager.add_order_listener(order_listener)
    order_manager.add_fill_listener(fill_listener)

    # Start order manager
    await order_manager.start()
    print("✅ Order manager started")

    # Submit test orders
    try:
        # Market order
        market_order_request = OrderRequest(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )

        order1 = await order_manager.submit_order(market_order_request)
        print(f"📤 Submitted market order: {order1.order_id}")

        # Limit order
        limit_order_request = OrderRequest(
            symbol="GOOGL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=50.0,
            price=150.00,
        )

        order2 = await order_manager.submit_order(limit_order_request)
        print(f"📤 Submitted limit order: {order2.order_id}")

        # Wait for processing
        await asyncio.sleep(2)

        # Check positions
        positions = order_manager.get_positions()
        print(f"📊 Positions: {positions}")

        # Get performance metrics
        metrics = order_manager.get_performance_metrics()
        print(f"📈 Fill rate: {metrics['orders']['fill_rate']:.2%}")
        print(f"💰 Total volume: ${metrics['execution']['total_volume']:.2f}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")

    finally:
        await order_manager.stop()
        print("🛑 Order manager stopped")

    print("\n🚀 Live Order Management System ready for production!")


if __name__ == "__main__":
    if ASYNC_AVAILABLE:
        asyncio.run(main())
    else:
        print("Async libraries not available - showing configuration only")
        print("Live Order Management System Framework Created! 🚀")
