from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict

from bot_v2.features.brokerages.core.interfaces import (
    IBrokerage,
    Product,
    MarketType,
    Quote,
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
    OrderStatus,
    Balance,
    Position,
)
from bot_v2.features.live_trade.risk import LiveRiskManager, RiskConfig
from bot_v2.orchestration.live_execution import LiveExecutionEngine


class StubPartialFillBroker(IBrokerage):
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}

    def connect(self) -> bool:  # pragma: no cover
        return True

    def disconnect(self) -> None:  # pragma: no cover
        pass

    def validate_connection(self) -> bool:  # pragma: no cover
        return True

    def get_account_id(self) -> str:  # pragma: no cover
        return "TEST"

    def list_balances(self) -> List[Balance]:
        return [Balance(asset='USD', total=Decimal('100000'), available=Decimal('100000'), hold=Decimal('0'))]

    def list_products(self, market: Optional[MarketType] = None) -> List[Product]:  # pragma: no cover
        return [self.get_product('BTC-PERP')]

    def get_product(self, symbol: str) -> Product:
        return Product(
            symbol=symbol,
            base_asset=symbol.split('-')[0],
            quote_asset='USD',
            market_type=MarketType.PERPETUAL,
            min_size=Decimal('0.001'),
            step_size=Decimal('0.001'),
            min_notional=Decimal('10'),
            price_increment=Decimal('0.01'),
            leverage_max=10,
        )

    def get_quote(self, symbol: str) -> Quote:
        return Quote(symbol=symbol, bid=Decimal('99.5'), ask=Decimal('100.5'), last=Decimal('100'), ts=datetime.utcnow())

    def get_candles(
        self,
        symbol: str,
        granularity: str,
        limit: int = 200,
        start=None,
        end=None,
    ):  # pragma: no cover
        return []

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
        now = datetime.utcnow()
        oid = client_id or f"ord_{len(self.orders)+1}"
        filled = qty / 2 if order_type == OrderType.LIMIT else qty
        status = OrderStatus.PARTIALLY_FILLED if order_type == OrderType.LIMIT else OrderStatus.FILLED
        avg = price if order_type == OrderType.LIMIT else Decimal('100')

        # Update position with filled portion only
        if filled > 0:
            pos = self.positions.get(symbol)
            if side == OrderSide.BUY:
                if pos and pos.side == 'long':
                    new_qty = pos.qty + filled
                    new_entry = ((pos.entry_price * pos.qty) + (avg * filled)) / new_qty
                    self.positions[symbol] = Position(symbol, new_qty, new_entry, avg, Decimal('0'), Decimal('0'), None, 'long')
                elif pos and pos.side == 'short':
                    # Reduce short; do not flip
                    reduce_qty = min(pos.qty, filled)
                    remaining = pos.qty - reduce_qty
                    side_new = 'short' if remaining > 0 else 'long'
                    qty_new = remaining if remaining > 0 else Decimal('0')
                    self.positions[symbol] = Position(symbol, qty_new, pos.entry_price if remaining > 0 else avg, avg, Decimal('0'), Decimal('0'), None, side_new)
                else:
                    self.positions[symbol] = Position(symbol, filled, avg, avg, Decimal('0'), Decimal('0'), None, 'long')
            else:  # SELL
                if pos and pos.side == 'short':
                    new_qty = pos.qty + filled
                    new_entry = ((pos.entry_price * pos.qty) + (avg * filled)) / new_qty
                    self.positions[symbol] = Position(symbol, new_qty, new_entry, avg, Decimal('0'), Decimal('0'), None, 'short')
                elif pos and pos.side == 'long':
                    reduce_qty = min(pos.qty, filled)
                    remaining = pos.qty - reduce_qty
                    side_new = 'long' if remaining > 0 else 'short'
                    qty_new = remaining if remaining > 0 else Decimal('0')
                    self.positions[symbol] = Position(symbol, qty_new, pos.entry_price if remaining > 0 else avg, avg, Decimal('0'), Decimal('0'), None, side_new)
                else:
                    self.positions[symbol] = Position(symbol, filled, avg, avg, Decimal('0'), Decimal('0'), None, 'short')

        order = Order(
            id=oid,
            client_id=oid,
            symbol=symbol,
            side=side,
            type=order_type,
            qty=qty,
            price=price,
            stop_price=stop_price,
            tif=tif,
            status=status,
            filled_qty=filled,
            avg_fill_price=avg,
            submitted_at=now,
            updated_at=now,
        )

        self.orders[oid] = order
        return order

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            o = self.orders[order_id]
            self.orders[order_id] = Order(
                **{**o.__dict__, 'status': OrderStatus.CANCELLED, 'updated_at': datetime.utcnow()}
            )
            return True
        return False

    def get_order(self, order_id: str) -> Order:
        return self.orders[order_id]

    def list_orders(self, status: Optional[OrderStatus] = None, symbol: Optional[str] = None) -> List[Order]:  # pragma: no cover
        return list(self.orders.values())

    def list_positions(self) -> List[Position]:
        return [p for p in self.positions.values() if p.qty > 0]

    def list_fills(self, symbol: Optional[str] = None, limit: int = 200):  # pragma: no cover
        return []

    def stream_trades(self, symbols):  # pragma: no cover
        return []

    def stream_orderbook(self, symbols, level: int = 1):  # pragma: no cover
        return []


def test_partial_fill_limit_order_updates_position():
    broker = StubPartialFillBroker()
    risk = LiveRiskManager(config=RiskConfig(max_position_pct_per_symbol=0.9, max_exposure_pct=0.9))
    engine = LiveExecutionEngine(broker, risk_manager=risk)

    # Place a limit BUY for 2 units; broker fills half
    oid = engine.place_order(
        symbol='BTC-PERP',
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        qty=Decimal('2'),
        price=Decimal('100.00')
    )
    assert oid is not None
    ord_obj = broker.get_order(oid)
    assert ord_obj.status == OrderStatus.PARTIALLY_FILLED
    # Position should reflect only filled half
    pos = broker.list_positions()[0]
    assert pos.symbol == 'BTC-PERP'
    assert pos.side == 'long'
    assert pos.qty == Decimal('1')
    assert pos.entry_price == Decimal('100.00')
