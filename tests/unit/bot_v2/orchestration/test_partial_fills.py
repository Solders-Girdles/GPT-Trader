from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from bot_v2.features.brokerages.core.interfaces import (
    Balance,
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
from bot_v2.features.live_trade.risk import LiveRiskManager, RiskConfig
from bot_v2.orchestration.live_execution import LiveExecutionEngine


class StubPartialFillBroker(IBrokerage):
    def __init__(self):
        self.orders: dict[str, Order] = {}
        self.positions: dict[str, Position] = {}

    def connect(self) -> bool:  # pragma: no cover
        return True

    def disconnect(self) -> None:  # pragma: no cover
        pass

    def validate_connection(self) -> bool:  # pragma: no cover
        return True

    def get_account_id(self) -> str:  # pragma: no cover
        return "TEST"

    def list_balances(self) -> list[Balance]:
        return [
            Balance(
                asset="USD", total=Decimal("100000"), available=Decimal("100000"), hold=Decimal("0")
            )
        ]

    def list_products(self, market: MarketType | None = None) -> list[Product]:  # pragma: no cover
        return [self.get_product("BTC-PERP")]

    def get_product(self, symbol: str) -> Product:
        return Product(
            symbol=symbol,
            base_asset=symbol.split("-")[0],
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
            leverage_max=10,
        )

    def get_quote(self, symbol: str) -> Quote:
        return Quote(
            symbol=symbol,
            bid=Decimal("99.5"),
            ask=Decimal("100.5"),
            last=Decimal("100"),
            ts=datetime.utcnow(),
        )

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
        quantity: Decimal,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        tif: TimeInForce = TimeInForce.GTC,
        client_id: str | None = None,
        reduce_only: bool | None = None,
        leverage: int | None = None,
    ) -> Order:
        now = datetime.utcnow()
        oid = client_id or f"ord_{len(self.orders)+1}"
        filled = quantity / 2 if order_type == OrderType.LIMIT else quantity
        status = (
            OrderStatus.PARTIALLY_FILLED if order_type == OrderType.LIMIT else OrderStatus.FILLED
        )
        avg = price if order_type == OrderType.LIMIT else Decimal("100")

        # Update position with filled portion only
        if filled > 0:
            pos = self.positions.get(symbol)
            if side == OrderSide.BUY:
                if pos and pos.side == "long":
                    new_quantity = pos.quantity + filled
                    new_entry = ((pos.entry_price * pos.quantity) + (avg * filled)) / new_quantity
                    self.positions[symbol] = Position(
                        symbol,
                        new_quantity,
                        new_entry,
                        avg,
                        Decimal("0"),
                        Decimal("0"),
                        None,
                        "long",
                    )
                elif pos and pos.side == "short":
                    # Reduce short; do not flip
                    reduce_quantity = min(pos.quantity, filled)
                    remaining = pos.quantity - reduce_quantity
                    side_new = "short" if remaining > 0 else "long"
                    quantity_new = remaining if remaining > 0 else Decimal("0")
                    self.positions[symbol] = Position(
                        symbol,
                        quantity_new,
                        pos.entry_price if remaining > 0 else avg,
                        avg,
                        Decimal("0"),
                        Decimal("0"),
                        None,
                        side_new,
                    )
                else:
                    self.positions[symbol] = Position(
                        symbol, filled, avg, avg, Decimal("0"), Decimal("0"), None, "long"
                    )
            else:  # SELL
                if pos and pos.side == "short":
                    new_quantity = pos.quantity + filled
                    new_entry = ((pos.entry_price * pos.quantity) + (avg * filled)) / new_quantity
                    self.positions[symbol] = Position(
                        symbol,
                        new_quantity,
                        new_entry,
                        avg,
                        Decimal("0"),
                        Decimal("0"),
                        None,
                        "short",
                    )
                elif pos and pos.side == "long":
                    reduce_quantity = min(pos.quantity, filled)
                    remaining = pos.quantity - reduce_quantity
                    side_new = "long" if remaining > 0 else "short"
                    quantity_new = remaining if remaining > 0 else Decimal("0")
                    self.positions[symbol] = Position(
                        symbol,
                        quantity_new,
                        pos.entry_price if remaining > 0 else avg,
                        avg,
                        Decimal("0"),
                        Decimal("0"),
                        None,
                        side_new,
                    )
                else:
                    self.positions[symbol] = Position(
                        symbol, filled, avg, avg, Decimal("0"), Decimal("0"), None, "short"
                    )

        order = Order(
            id=oid,
            client_id=oid,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            tif=tif,
            status=status,
            filled_quantity=filled,
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
                **{**o.__dict__, "status": OrderStatus.CANCELLED, "updated_at": datetime.utcnow()}
            )
            return True
        return False

    def get_order(self, order_id: str) -> Order:
        return self.orders[order_id]

    def list_orders(
        self, status: OrderStatus | None = None, symbol: str | None = None
    ) -> list[Order]:  # pragma: no cover
        return list(self.orders.values())

    def list_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if p.quantity > 0]

    def list_fills(self, symbol: str | None = None, limit: int = 200):  # pragma: no cover
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
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("2"),
        price=Decimal("100.00"),
    )
    assert oid is not None
    ord_obj = broker.get_order(oid)
    assert ord_obj.status == OrderStatus.PARTIALLY_FILLED
    # Position should reflect only filled half
    pos = broker.list_positions()[0]
    assert pos.symbol == "BTC-PERP"
    assert pos.side == "long"
    assert pos.quantity == Decimal("1")
    assert pos.entry_price == Decimal("100.00")
