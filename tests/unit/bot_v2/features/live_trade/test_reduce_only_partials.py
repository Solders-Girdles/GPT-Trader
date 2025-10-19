from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

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
from bot_v2.features.live_trade.risk import LiveRiskManager, RiskConfig, ValidationError
from bot_v2.orchestration.live_execution import LiveExecutionEngine


class ReduceOnlyStubBroker(IBrokerage):
    def __init__(self):
        self.orders: dict[str, Order] = {}
        self.positions: dict[str, Position] = {}

    def list_balances(self) -> list[Balance]:
        return [
            Balance(
                asset="USD", total=Decimal("100000"), available=Decimal("100000"), hold=Decimal("0")
            )
        ]

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
            leverage_max=10,
        )

    def get_quote(self, symbol: str) -> Quote:
        return Quote(
            symbol=symbol,
            bid=Decimal("99"),
            ask=Decimal("101"),
            last=Decimal("100"),
            ts=datetime.utcnow(),
        )

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
        oid = client_id or f"ord_{len(self.orders) + 1}"
        pos = self.positions.get(symbol)

        filled = quantity
        # Enforce exchange-side reduce_only clamp to avoid flips
        if reduce_only:
            if side == OrderSide.SELL and pos and pos.side == "long":
                filled = min(quantity, pos.quantity)
            elif side == OrderSide.BUY and pos and pos.side == "short":
                filled = min(quantity, pos.quantity)

        avg = price or Decimal("100")

        # Apply filled to positions without allowing flips when reduce_only
        if filled > 0:
            if side == OrderSide.SELL:
                if pos and pos.side == "long":
                    remaining = pos.quantity - min(filled, pos.quantity)
                    self.positions[symbol] = Position(
                        symbol,
                        remaining,
                        pos.entry_price,
                        avg,
                        Decimal("0"),
                        Decimal("0"),
                        None,
                        "long" if remaining > 0 else "long",
                    )
                else:
                    # Opening short allowed only if not reduce_only
                    if not reduce_only:
                        self.positions[symbol] = Position(
                            symbol, filled, avg, avg, Decimal("0"), Decimal("0"), None, "short"
                        )
            else:  # BUY
                if pos and pos.side == "short":
                    remaining = pos.quantity - min(filled, pos.quantity)
                    self.positions[symbol] = Position(
                        symbol,
                        remaining,
                        pos.entry_price,
                        avg,
                        Decimal("0"),
                        Decimal("0"),
                        None,
                        "short" if remaining > 0 else "short",
                    )
                else:
                    if not reduce_only:
                        self.positions[symbol] = Position(
                            symbol, filled, avg, avg, Decimal("0"), Decimal("0"), None, "long"
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
            status=OrderStatus.FILLED,
            filled_quantity=filled,
            avg_fill_price=avg,
            submitted_at=now,
            updated_at=now,
        )
        self.orders[oid] = order
        return order

    # Unused interface methods for this test
    def connect(self) -> bool:  # pragma: no cover
        return True

    def disconnect(self) -> None:  # pragma: no cover
        pass

    def validate_connection(self) -> bool:  # pragma: no cover
        return True

    def list_products(self, market: MarketType | None = None):  # pragma: no cover
        return []

    def cancel_order(self, order_id: str) -> bool:  # pragma: no cover
        return True

    def get_order(self, order_id: str) -> Order:  # pragma: no cover
        return self.orders[order_id]

    def list_orders(
        self, status: OrderStatus | None = None, symbol: str | None = None
    ):  # pragma: no cover
        return []

    def list_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if p.quantity > 0]

    def list_fills(self, symbol: str | None = None, limit: int = 200):  # pragma: no cover
        return []

    def stream_trades(self, symbols):  # pragma: no cover
        return []

    def stream_orderbook(self, symbols, level: int = 1):  # pragma: no cover
        return []


def test_reduce_only_rejects_opening_trades():
    broker = ReduceOnlyStubBroker()
    risk = LiveRiskManager(
        config=RiskConfig(
            reduce_only_mode=True, max_position_pct_per_symbol=0.9, max_exposure_pct=0.9
        )
    )
    engine = LiveExecutionEngine(broker, risk_manager=risk)

    with pytest.raises(ValidationError):
        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
        )


def test_reduce_only_sell_clamped_to_position_size():
    broker = ReduceOnlyStubBroker()
    # Seed a long position of 1
    broker.positions["BTC-PERP"] = Position(
        "BTC-PERP",
        Decimal("1"),
        Decimal("100"),
        Decimal("100"),
        Decimal("0"),
        Decimal("0"),
        None,
        "long",
    )
    risk = LiveRiskManager(
        config=RiskConfig(
            reduce_only_mode=True, max_position_pct_per_symbol=0.9, max_exposure_pct=0.9
        )
    )
    engine = LiveExecutionEngine(broker, risk_manager=risk)

    oid = engine.place_order(
        symbol="BTC-PERP",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=Decimal("2"),  # greater than position size
        reduce_only=True,
    )
    assert oid is not None
    # Position should not flip; should be reduced to zero
    positions = broker.list_positions()
    assert positions == []  # No remaining position
