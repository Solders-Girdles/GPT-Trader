"""
Mock broker for development and testing.

DEPRECATION (tests): Prefer deterministic test doubles or real API-backed
integration tests. MockBroker is still available for DEV profile and manual
validation, but unit/integration tests should migrate to stable fixtures.
"""

from decimal import Decimal
from datetime import datetime
from typing import List, Dict, Any, Optional
import random
import logging
import os

from ..features.brokerages.core.interfaces import (
    Product,
    MarketType,
    Balance,
    Quote,
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
    OrderStatus,
    Position,
)

logger = logging.getLogger(__name__)


class MockBroker:
    """Mock broker implementation for testing."""
    
    def __init__(self):
        """Initialize mock broker with test data."""
        try:
            if os.getenv('PERPS_SILENCE_MOCK_BROKER_WARNING', '') not in ('1', 'true', 'TRUE', 'True'):
                logger.warning(
                    "MockBroker is deprecated for automated tests; use DeterministicBroker "
                    "(tests/utils/deterministic_broker.py) for unit tests or real API integration tests."
                )
        except Exception:
            pass
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.equity = Decimal("100000")
        self.connected = False

        # Mock market prices
        self.marks: Dict[str, Decimal] = {}

        # Mock products
        self._products: List[Product] = []
        self._product_map: Dict[str, Product] = {}
        self._init_products()
        
        # Mock WebSocket data
        self._market_data = {}
        self._ws_client = None
        self._streaming = False
        
    def _init_products(self):
        """Initialize mock products for both spot and perpetual markets."""
        spot_specs: Dict[str, Dict[str, Any]] = {
            "BTC-USD": {
                "mark": Decimal("50000"),
                "base": "BTC",
                "quote": "USD",
                "min_size": Decimal("0.0001"),
                "step_size": Decimal("0.0001"),
                "min_notional": Decimal("10"),
                "price_increment": Decimal("0.01"),
            },
            "ETH-USD": {
                "mark": Decimal("3000"),
                "base": "ETH",
                "quote": "USD",
                "min_size": Decimal("0.001"),
                "step_size": Decimal("0.001"),
                "min_notional": Decimal("5"),
                "price_increment": Decimal("0.01"),
            },
            "SOL-USD": {
                "mark": Decimal("100"),
                "base": "SOL",
                "quote": "USD",
                "min_size": Decimal("0.01"),
                "step_size": Decimal("0.01"),
                "min_notional": Decimal("5"),
                "price_increment": Decimal("0.01"),
            },
            "XRP-USD": {
                "mark": Decimal("0.5"),
                "base": "XRP",
                "quote": "USD",
                "min_size": Decimal("1"),
                "step_size": Decimal("0.1"),
                "min_notional": Decimal("5"),
                "price_increment": Decimal("0.0001"),
            },
            "LTC-USD": {
                "mark": Decimal("80"),
                "base": "LTC",
                "quote": "USD",
                "min_size": Decimal("0.01"),
                "step_size": Decimal("0.01"),
                "min_notional": Decimal("5"),
                "price_increment": Decimal("0.01"),
            },
            "ADA-USD": {
                "mark": Decimal("0.45"),
                "base": "ADA",
                "quote": "USD",
                "min_size": Decimal("1"),
                "step_size": Decimal("1"),
                "min_notional": Decimal("5"),
                "price_increment": Decimal("0.0001"),
            },
            "DOGE-USD": {
                "mark": Decimal("0.12"),
                "base": "DOGE",
                "quote": "USD",
                "min_size": Decimal("1"),
                "step_size": Decimal("1"),
                "min_notional": Decimal("5"),
                "price_increment": Decimal("0.0001"),
            },
            "BCH-USD": {
                "mark": Decimal("400"),
                "base": "BCH",
                "quote": "USD",
                "min_size": Decimal("0.001"),
                "step_size": Decimal("0.001"),
                "min_notional": Decimal("5"),
                "price_increment": Decimal("0.01"),
            },
            "AVAX-USD": {
                "mark": Decimal("35"),
                "base": "AVAX",
                "quote": "USD",
                "min_size": Decimal("0.01"),
                "step_size": Decimal("0.01"),
                "min_notional": Decimal("5"),
                "price_increment": Decimal("0.01"),
            },
            "LINK-USD": {
                "mark": Decimal("15"),
                "base": "LINK",
                "quote": "USD",
                "min_size": Decimal("0.01"),
                "step_size": Decimal("0.01"),
                "min_notional": Decimal("5"),
                "price_increment": Decimal("0.01"),
            },
        }

        perp_specs: Dict[str, Dict[str, Any]] = {
            "BTC-PERP": {
                "mark": Decimal("50000"),
                "base": "BTC",
                "quote": "USD",
                "min_size": Decimal("0.001"),
                "step_size": Decimal("0.001"),
                "min_notional": Decimal("10"),
                "price_increment": Decimal("0.01"),
                "leverage_max": 3,
                "contract_size": Decimal("1"),
                "funding_rate": Decimal("0.0001"),
            },
            "ETH-PERP": {
                "mark": Decimal("3000"),
                "base": "ETH",
                "quote": "USD",
                "min_size": Decimal("0.01"),
                "step_size": Decimal("0.01"),
                "min_notional": Decimal("5"),
                "price_increment": Decimal("0.01"),
                "leverage_max": 3,
                "contract_size": Decimal("1"),
                "funding_rate": Decimal("0.0001"),
            },
            "SOL-PERP": {
                "mark": Decimal("100"),
                "base": "SOL",
                "quote": "USD",
                "min_size": Decimal("0.1"),
                "step_size": Decimal("0.1"),
                "min_notional": Decimal("5"),
                "price_increment": Decimal("0.01"),
                "leverage_max": 3,
                "contract_size": Decimal("1"),
                "funding_rate": Decimal("0.0001"),
            },
            "XRP-PERP": {
                "mark": Decimal("0.5"),
                "base": "XRP",
                "quote": "USD",
                "min_size": Decimal("10"),
                "step_size": Decimal("10"),
                "min_notional": Decimal("5"),
                "price_increment": Decimal("0.0001"),
                "leverage_max": 3,
                "contract_size": Decimal("1"),
                "funding_rate": Decimal("0.0001"),
            },
        }

        self.marks.clear()
        self._products.clear()
        self._product_map.clear()

        for symbol, spec in {**spot_specs, **perp_specs}.items():
            self.marks[symbol] = spec["mark"]
            market_type = MarketType.SPOT if symbol.endswith("-USD") else MarketType.PERPETUAL
            product = Product(
                symbol=symbol,
                base_asset=spec["base"],
                quote_asset=spec["quote"],
                market_type=market_type,
                min_size=spec["min_size"],
                step_size=spec["step_size"],
                min_notional=spec["min_notional"],
                price_increment=spec["price_increment"],
                leverage_max=spec.get("leverage_max") if market_type != MarketType.SPOT else 1,
                contract_size=spec.get("contract_size"),
                funding_rate=spec.get("funding_rate"),
                next_funding_time=None,
            )
            self._products.append(product)
            self._product_map[product.symbol] = product

    def connect(self) -> bool:
        """Simulate broker connection."""
        self.connected = True
        logger.info("Mock broker connected")
        return True
    
    def disconnect(self):
        """Simulate broker disconnection."""
        self.connected = False
        logger.info("Mock broker disconnected")
    
    def get_product(self, symbol: str):
        """Get product by symbol."""
        return self._product_map.get(symbol.upper())
    
    def list_products(self, market: Optional[MarketType] = None) -> List[Product]:
        """List available products."""
        if market is None:
            return list(self._products)
        return [p for p in self._products if p.market_type == market]
    
    def list_balances(self):
        """Get mock balances compatible with IBrokerage interface."""
        return [Balance(asset='USD', available=self.equity, total=self.equity, hold=Decimal('0'))]
    
    def get_account(self):
        """Get mock account info for backward compatibility."""
        class _Account:
            def __init__(self, eq: Decimal):
                self.equity = eq
                self.balance = eq
                self.cash = eq * Decimal('0.5')
                self.buying_power = eq * Decimal('2')
                self.portfolio_value = eq
        return _Account(self.equity)
    
    def list_positions(self):
        """Get mock positions compatible with IBrokerage interface."""
        return list(self.positions.values())
    
    def get_positions(self):
        """Get mock positions for backward compatibility."""
        return list(self.positions.values())

    def list_orders(self, status: Optional[str] = None, symbol: Optional[str] = None) -> List[Order]:
        """Get mock orders, optionally filtered."""
        orders: List[Order] = self.orders
        if status:
            if isinstance(status, OrderStatus):
                orders = [o for o in orders if o.status == status]
            else:
                orders = [o for o in orders if o.status.value == str(status)]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return list(orders)
    
    def get_quote(self, symbol: str):
        """Get mock quote with realistic bid/ask spread."""
        price = self.marks.get(symbol, Decimal("1000"))
        spread = price * Decimal("0.0001")
        return Quote(symbol=symbol, bid=price - spread, ask=price + spread, last=price, ts=datetime.utcnow())
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        qty: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        tif: TimeInForce = TimeInForce.GTC,
        client_id: Optional[str] = None,
        reduce_only: Optional[bool] = None,
        leverage: Optional[int] = None,
    ) -> Order:
        """Place mock order following IBrokerage contract."""
        order_id = client_id or f"mock_{len(self.orders)}"
        now = datetime.utcnow()
        status = OrderStatus.FILLED if order_type == OrderType.MARKET else OrderStatus.SUBMITTED
        avg_fill = self.marks.get(symbol, Decimal("1000")) if status == OrderStatus.FILLED else None

        order = Order(
            id=order_id,
            client_id=order_id,
            symbol=symbol,
            side=side,
            type=order_type,
            qty=qty,
            price=price,
            stop_price=stop_price,
            tif=tif,
            status=status,
            filled_qty=qty if status == OrderStatus.FILLED else Decimal('0'),
            avg_fill_price=avg_fill,
            submitted_at=now,
            updated_at=now,
        )

        self.orders.append(order)

        if status == OrderStatus.FILLED:
            self._apply_fill_to_position(symbol, side.value.lower(), qty, avg_fill or Decimal('0'))

        logger.info(f"Mock order placed: {order}")
        return order
    
    def _apply_fill_to_position(self, symbol: str, side: str, quantity: Decimal, price: Decimal):
        """Update mock position after fill."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            if side == 'buy':
                if pos.side == 'long':
                    new_qty = pos.qty + quantity
                    pos.entry_price = ((pos.entry_price * pos.qty) + (price * quantity)) / new_qty
                    pos.qty = new_qty
                else:
                    reduce_qty = min(pos.qty, quantity)
                    pos.qty -= reduce_qty
                    if pos.qty == 0:
                        pos.side = 'long'
                        pos.qty = quantity - reduce_qty
                        pos.entry_price = price
                pos.mark_price = price
            else:
                if pos.side == 'short':
                    new_qty = pos.qty + quantity
                    pos.entry_price = ((pos.entry_price * pos.qty) + (price * quantity)) / new_qty
                    pos.qty = new_qty
                else:
                    reduce_qty = min(pos.qty, quantity)
                    pos.qty -= reduce_qty
                    if pos.qty == 0:
                        pos.side = 'short'
                        pos.qty = quantity - reduce_qty
                        pos.entry_price = price
                pos.mark_price = price
        else:
            if side == 'buy':
                self.positions[symbol] = Position(
                    symbol=symbol,
                    qty=quantity,
                    entry_price=price,
                    mark_price=price,
                    unrealized_pnl=Decimal('0'),
                    realized_pnl=Decimal('0'),
                    leverage=None,
                    side='long',
                )
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    qty=quantity,
                    entry_price=price,
                    mark_price=price,
                    unrealized_pnl=Decimal('0'),
                    realized_pnl=Decimal('0'),
                    leverage=None,
                    side='short',
                )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel mock order."""
        for idx, order in enumerate(self.orders):
            if order.id == order_id and order.status == OrderStatus.SUBMITTED:
                self.orders[idx] = Order(
                    **{**order.__dict__, 'status': OrderStatus.CANCELLED, 'updated_at': datetime.utcnow()}
                )
                logger.info(f"Mock order cancelled: {order_id}")
                return True
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        for order in self.orders:
            if order.id == order_id:
                return order
        logger.debug("MockBroker.get_order: unknown order_id %s", order_id)
        return None

    def list_fills(self, symbol: Optional[str] = None, limit: int = 200) -> List[Dict]:
        return []

    # Optional helper to mirror real adapter risk info
    def get_position_risk(self, symbol: str) -> Dict[str, Any]:
        """Return mock risk info (no real liquidation price)."""
        # Could approximate a liquidation price far from mark to avoid false trips
        try:
            mark = self.marks.get(symbol)
            if mark is None:
                return {}
            # Place a very distant mock liquidation to avoid tripping tests
            return {"liquidation_price": mark * Decimal('0.5')}
        except Exception:
            return {}
    
    def start_market_data(self, symbols: List[str]):
        """Start mock WebSocket market data."""
        self._ws_client = "mock_ws"
        self._streaming = True
        
        for symbol in symbols:
            price = self.marks.get(symbol, Decimal("1000"))
            
            # Create realistic market snapshot
            self._market_data[symbol] = {
                'mid': price,
                'spread_bps': 5 + random.uniform(0, 5),  # 5-10 bps
                'depth_l1': price * Decimal(str(random.uniform(100, 500))),  # $100k-500k
                'depth_l10': price * Decimal(str(random.uniform(1000, 5000))),  # $1M-5M
                'vol_1m': price * Decimal(str(random.uniform(50, 500))),  # $50k-500k
                'vol_5m': price * Decimal(str(random.uniform(500, 2500))),  # $500k-2.5M
                'last_update': datetime.now()
            }
        
        logger.info(f"Mock WebSocket started for {symbols}")

    def stream_trades(self, symbols: List[str]):
        """Yield fake trade ticks for the requested symbols.

        This provides a minimal iterable so orchestrators can update mark windows
        during development without a real WS. Prices follow a tiny random walk.
        """
        import time
        rng = random.Random(42)
        syms = list(symbols)
        if not syms:
            syms = list(self.marks.keys())
        while self._streaming:
            for sym in syms:
                base = self.marks.get(sym, Decimal("1000"))
                # Random walk within +/- 10 bps
                drift = Decimal(str(rng.uniform(-0.001, 0.001)))
                new_price = max(Decimal("0.01"), base * (Decimal("1") + drift))
                # Persist for get_quote consumers
                self.marks[sym] = new_price
                yield {
                    'type': 'trade',
                    'product_id': sym,
                    'price': str(new_price),
                    'time': datetime.utcnow().isoformat(),
                }
                time.sleep(0.05)
    
    def get_market_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Get mock market snapshot."""
        if symbol not in self._market_data:
            # Generate on demand if not initialized
            price = self.marks.get(symbol, Decimal("1000"))
            self._market_data[symbol] = {
                'mid': price,
                'spread_bps': 5 + random.uniform(0, 5),
                'depth_l1': price * Decimal(str(random.uniform(100, 500))),
                'depth_l10': price * Decimal(str(random.uniform(1000, 5000))),
                'vol_1m': price * Decimal(str(random.uniform(50, 500))),
                'vol_5m': price * Decimal(str(random.uniform(500, 2500))),
                'last_update': datetime.now()
            }
        
        # Update timestamp
        self._market_data[symbol]['last_update'] = datetime.now()
        
        return self._market_data[symbol].copy()
    
    def is_stale(self, symbol: str, threshold_seconds: int = 10) -> bool:
        """Check if market data is stale."""
        if symbol not in self._market_data:
            return True
        
        last_update = self._market_data[symbol].get('last_update')
        if not last_update:
            return True
        
        age = (datetime.now() - last_update).total_seconds()
        return age > threshold_seconds

    # ===== Test helpers (deterministic) =====
    def set_mark(self, symbol: str, price: Decimal) -> None:
        """Set the current mark and revalue any open position's mark price.

        Intended for deterministic tests and verification scripts.
        """
        self.marks[symbol] = price
        if symbol in self.positions:
            pos = self.positions[symbol]
            self.positions[symbol] = Position(
                symbol=pos.symbol,
                qty=pos.qty,
                entry_price=pos.entry_price,
                mark_price=price,
                unrealized_pnl=pos.unrealized_pnl,
                realized_pnl=pos.realized_pnl,
                leverage=pos.leverage,
                side=pos.side,
            )
