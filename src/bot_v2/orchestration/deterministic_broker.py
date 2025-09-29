"""Deterministic IBrokerage implementation for development and tests."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Literal

from bot_v2.features.brokerages.core.interfaces import (
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


class DeterministicBroker(IBrokerage):
    """A simple, predictable test broker implementing IBrokerage.

    - Static quotes derived from an internal `marks` map.
    - No random behavior; no side-effects beyond local state.
    - Suitable for unit tests that need a concrete broker object.
    """

    def __init__(self, equity: Decimal = Decimal("100000")) -> None:
        self._connected = False
        self.equity = Decimal(str(equity))
        # Minimal product catalog sufficient for tests
        self._products: dict[str, Product] = {
            "BTC-PERP": Product(
                symbol="BTC-PERP",
                base_asset="BTC",
                quote_asset="USD",
                market_type=MarketType.PERPETUAL,
                min_size=Decimal("0.001"),
                step_size=Decimal("0.001"),
                min_notional=Decimal("10"),
                price_increment=Decimal("0.01"),
                leverage_max=3,
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
                leverage_max=3,
            ),
            "XRP-PERP": Product(
                symbol="XRP-PERP",
                base_asset="XRP",
                quote_asset="USD",
                market_type=MarketType.PERPETUAL,
                min_size=Decimal("10"),
                step_size=Decimal("10"),
                min_notional=Decimal("10"),
                price_increment=Decimal("0.0001"),
                leverage_max=3,
            ),
        }
        # Deterministic marks
        self.marks: dict[str, Decimal] = {
            "BTC-PERP": Decimal("50000"),
            "ETH-PERP": Decimal("3000"),
            "XRP-PERP": Decimal("0.50"),
        }
        self._orders: list[Order] = []
        self._positions: dict[str, Position] = {}

    # ---- Connectivity ----
    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def validate_connection(self) -> bool:
        return self._connected

    def get_account_id(self) -> str:
        return "DETERMINISTIC"

    # ---- Accounts ----
    def list_balances(self) -> list[Balance]:
        return [Balance(asset="USD", total=self.equity, available=self.equity, hold=Decimal("0"))]

    # ---- Products/Market Data ----
    def list_products(self, market: MarketType | None = None) -> list[Product]:
        products = list(self._products.values())
        if market is None:
            return products
        return [p for p in products if p.market_type == market]

    def get_product(self, symbol: str) -> Product:
        return self._products.get(symbol) or Product(
            symbol=symbol,
            base_asset=(symbol.split("-")[0] if "-" in symbol else symbol),
            quote_asset=(symbol.split("-")[-1] if "-" in symbol else "USD"),
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
        )

    def get_quote(self, symbol: str) -> Quote:
        last = self.marks.get(symbol, Decimal("1000"))
        # Tight, deterministic spread: 2 bps total width
        half_spread = last * Decimal("0.0001")
        return Quote(
            symbol=symbol,
            bid=last - half_spread,
            ask=last + half_spread,
            last=last,
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
        return []

    # ---- Orders ----
    @staticmethod
    def _require_quantity(quantity: Decimal | None, *, context: str) -> Decimal:
        if quantity is None:
            raise ValueError(f"{context} requires a quantity")
        return quantity if isinstance(quantity, Decimal) else Decimal(str(quantity))

    def place_order(
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
        order_quantity = self._require_quantity(quantity, context="place_order")
        oid = client_id or f"det_{len(self._orders)}"
        now = datetime.utcnow()
        status = OrderStatus.SUBMITTED if order_type == OrderType.LIMIT else OrderStatus.FILLED
        avg_fill = None if status != OrderStatus.FILLED else self.marks.get(symbol, Decimal("1000"))
        order = Order(
            id=oid,
            client_id=oid,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=order_quantity,
            price=price,
            stop_price=stop_price,
            tif=tif,
            status=status,
            filled_quantity=(order_quantity if status == OrderStatus.FILLED else Decimal("0")),
            avg_fill_price=avg_fill,
            submitted_at=now,
            updated_at=now,
        )
        self._orders.append(order)
        # Apply simple position effect if filled
        if status == OrderStatus.FILLED:
            self._apply_fill(symbol, side, order_quantity, avg_fill or Decimal("0"))
        return order

    def cancel_order(self, order_id: str) -> bool:
        for idx, existing in enumerate(self._orders):
            if existing.id == order_id and existing.status == OrderStatus.SUBMITTED:
                self._orders[idx] = Order(
                    id=existing.id,
                    client_id=existing.client_id,
                    symbol=existing.symbol,
                    side=existing.side,
                    type=existing.type,
                    price=existing.price,
                    stop_price=existing.stop_price,
                    tif=existing.tif,
                    status=OrderStatus.CANCELLED,
                    avg_fill_price=existing.avg_fill_price,
                    submitted_at=existing.submitted_at,
                    updated_at=datetime.utcnow(),
                    quantity=existing.quantity,
                    filled_quantity=existing.filled_quantity,
                )
                return True
        return False

    def get_order(self, order_id: str) -> Order | None:
        for o in self._orders:
            if o.id == order_id:
                return o
        return None

    def list_orders(
        self, status: OrderStatus | None = None, symbol: str | None = None
    ) -> list[Order]:
        res = self._orders
        if status is not None:
            res = [o for o in res if o.status == status]
        if symbol is not None:
            res = [o for o in res if o.symbol == symbol]
        return list(res)

    # ---- Positions and fills ----
    def list_positions(self) -> list[Position]:
        return list(self._positions.values())

    def list_fills(self, symbol: str | None = None, limit: int = 200) -> list[dict]:
        return []

    # ---- Streaming (not used in these unit tests) ----
    def stream_trades(self, symbols: Sequence[str]) -> Iterable[dict]:
        return iter(())

    def stream_orderbook(self, symbols: Sequence[str], level: int = 1) -> Iterable[dict]:
        return iter(())

    # ---- Test helpers ----
    def seed_position(
        self,
        symbol: str,
        side: Literal["long", "short"],
        quantity: Decimal | None = None,
        price: Decimal = Decimal("0"),
    ) -> None:
        """Create or replace a position deterministically for tests."""
        position_quantity = self._require_quantity(quantity, context="seed_position")
        self._positions[symbol] = Position(
            symbol=symbol,
            quantity=position_quantity,
            entry_price=Decimal(str(price)),
            mark_price=Decimal(str(price)),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            leverage=None,
            side=side,
        )

    def set_mark(self, symbol: str, price: Decimal) -> None:
        self.marks[symbol] = Decimal(str(price))
        if symbol in self._positions:
            p = self._positions[symbol]
            self._positions[symbol] = Position(
                symbol=p.symbol,
                quantity=p.quantity,
                entry_price=p.entry_price,
                mark_price=Decimal(str(price)),
                unrealized_pnl=p.unrealized_pnl,
                realized_pnl=p.realized_pnl,
                leverage=p.leverage,
                side=p.side,
            )

    # Internal: apply a filled order to position state
    def _apply_fill(self, symbol: str, side: OrderSide, quantity: Decimal, price: Decimal) -> None:
        position_quantity = quantity if isinstance(quantity, Decimal) else Decimal(str(quantity))
        existing = self._positions.get(symbol)
        if existing is None:
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=position_quantity,
                entry_price=price,
                mark_price=price,
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                leverage=None,
                side=("long" if side == OrderSide.BUY else "short"),
            )
            return
        base_quantity = existing.quantity
        if (existing.side == "long" and side == OrderSide.BUY) or (
            existing.side == "short" and side == OrderSide.SELL
        ):
            new_quantity = base_quantity + position_quantity
            new_entry = (
                (existing.entry_price * base_quantity) + (price * position_quantity)
            ) / new_quantity
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=new_quantity,
                entry_price=new_entry,
                mark_price=price,
                unrealized_pnl=Decimal("0"),
                realized_pnl=existing.realized_pnl,
                leverage=None,
                side=existing.side,
            )
        else:
            reduce_quantity = min(base_quantity, position_quantity)
            remaining = base_quantity - reduce_quantity
            if remaining > 0:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=remaining,
                    entry_price=existing.entry_price,
                    mark_price=price,
                    unrealized_pnl=Decimal("0"),
                    realized_pnl=existing.realized_pnl,
                    leverage=None,
                    side=existing.side,
                )
            else:
                leftover = position_quantity - reduce_quantity
                if leftover > 0:
                    new_side: Literal["long", "short"] = (
                        "short" if existing.side == "long" else "long"
                    )
                    self._positions[symbol] = Position(
                        symbol=symbol,
                        quantity=leftover,
                        entry_price=price,
                        mark_price=price,
                        unrealized_pnl=Decimal("0"),
                        realized_pnl=existing.realized_pnl,
                        leverage=None,
                        side=new_side,
                    )
                else:
                    del self._positions[symbol]
