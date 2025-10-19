"""
DeterministicBroker: a minimal, predictable brokerage for tests.

Purpose:
- Provide a stable IBrokerage-compatible stub without market simulation.
- Avoid false positives caused by complex mock behavior.

Usage:
- Prefer this deterministic stub in unit tests.
- Exposes simple hooks to seed positions and marks deterministically.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from decimal import Decimal

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
from bot_v2.features.brokerages.fixtures import (
    create_product as load_fixture_product,
)
from bot_v2.features.brokerages.fixtures import (
    default_marks as load_default_marks,
)
from bot_v2.features.brokerages.fixtures import (
    list_perpetual_symbols,
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

        # Load products and marks from structured fixtures
        self._products: dict[str, Product] = self._load_products_from_fixtures()
        self.marks: dict[str, Decimal] = self._load_marks_from_fixtures()
        self._orders: list[Order] = []
        self._positions: dict[str, Position] = {}
        self.order_books: dict[
            str, tuple[list[tuple[Decimal, Decimal]], list[tuple[Decimal, Decimal]]]
        ] = {}
        for symbol, mark in self.marks.items():
            self.order_books[symbol] = self._build_default_order_book(mark)

    def _load_products_from_fixtures(self) -> dict[str, Product]:
        """Load products from structured fixtures."""
        try:
            products: dict[str, Product] = {}
            for symbol in list_perpetual_symbols():
                products[symbol] = load_fixture_product(symbol, MarketType.PERPETUAL)

            if products:
                return products

        except Exception:
            pass  # Fall through to hardcoded defaults

        # Fall back to hardcoded defaults (kept for backward compatibility)
        return self._get_hardcoded_products()

    @staticmethod
    def _get_hardcoded_products() -> dict[str, Product]:
        """Return hardcoded product definitions for fallback."""
        return {
            "BTC-PERP": Product(
                symbol="BTC-PERP",
                base_asset="BTC",
                quote_asset="USD",
                market_type=MarketType.PERPETUAL,
                min_size=Decimal("0.001"),
                step_size=Decimal("0.001"),
                min_notional=Decimal("10"),
                price_increment=Decimal("0.01"),
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
            ),
        }

    def _load_marks_from_fixtures(self) -> dict[str, Decimal]:
        """Load mark prices from structured fixtures."""
        try:
            marks_data = load_default_marks()

            if isinstance(marks_data, dict) and marks_data:
                return dict(marks_data)

        except Exception:
            pass  # Fall through to hardcoded defaults

        # Fall back to hardcoded defaults (kept for backward compatibility)
        return self._get_hardcoded_marks()

    @staticmethod
    def _get_hardcoded_marks() -> dict[str, Decimal]:
        """Return hardcoded mark prices for fallback."""
        return {
            "BTC-PERP": Decimal("50000"),
            "ETH-PERP": Decimal("3000"),
            "XRP-PERP": Decimal("0.50"),
        }

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

    def seed_order_book(
        self,
        symbol: str,
        *,
        bids: Sequence[tuple[Decimal | float, Decimal | float]],
        asks: Sequence[tuple[Decimal | float, Decimal | float]],
    ) -> None:
        """Seed a deterministic order book for impact-cost tests."""

        def _normalise(
            levels: Sequence[tuple[Decimal | float, Decimal | float]],
        ) -> list[tuple[Decimal, Decimal]]:
            return [(Decimal(str(price)), Decimal(str(size))) for price, size in levels]

        self.order_books[symbol] = (_normalise(bids), _normalise(asks))

    def _build_default_order_book(
        self, mark: Decimal
    ) -> tuple[list[tuple[Decimal, Decimal]], list[tuple[Decimal, Decimal]]]:
        tick = mark * Decimal("0.0005")
        depth = mark * Decimal("2")
        bids = [(mark - tick * Decimal(i + 1), depth) for i in range(5)]
        asks = [(mark + tick * Decimal(i + 1), depth) for i in range(5)]
        return bids, asks

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
    def _require_quantity(
        quantity: Decimal | None,
        *,
        context: str,
    ) -> Decimal:
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
        oid = client_id or f"det_{len(self._orders)}"
        now = datetime.utcnow()
        status = OrderStatus.SUBMITTED if order_type == OrderType.LIMIT else OrderStatus.FILLED
        avg_fill = None if status != OrderStatus.FILLED else self.marks.get(symbol, Decimal("1000"))
        order_quantity = self._require_quantity(quantity, context="place_order")
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
        for idx, o in enumerate(self._orders):
            if o.id == order_id and o.status == OrderStatus.SUBMITTED:
                self._orders[idx] = Order(
                    **{
                        **o.__dict__,
                        "status": OrderStatus.CANCELLED,
                        "updated_at": datetime.utcnow(),
                    }
                )
                return True
        return False

    def get_order(self, order_id: str) -> Order:
        for o in self._orders:
            if o.id == order_id:
                return o
        # Fallback dummy
        now = datetime.utcnow()
        return Order(
            id=order_id,
            client_id=order_id,
            symbol="UNKNOWN",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=Decimal("0"),
            price=None,
            stop_price=None,
            tif=TimeInForce.GTC,
            status=OrderStatus.CANCELLED,
            filled_quantity=Decimal("0"),
            avg_fill_price=None,
            submitted_at=now,
            updated_at=now,
        )

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
        side: str,
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
        existing = self._positions.get(symbol)
        if existing is None:
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                mark_price=price,
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                leverage=None,
                side=("long" if side == OrderSide.BUY else "short"),
            )
            return
        base_quantity = existing.quantity
        # Simplified netting: if same side, increase quantity with avg price; else reduce
        if (existing.side == "long" and side == OrderSide.BUY) or (
            existing.side == "short" and side == OrderSide.SELL
        ):
            new_quantity = base_quantity + quantity
            new_entry = ((existing.entry_price * base_quantity) + (price * quantity)) / new_quantity
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
            # Reduce position
            reduce_quantity = min(base_quantity, quantity)
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
                # Flip to the other side with leftover quantity
                leftover = quantity - reduce_quantity
                if leftover > 0:
                    self._positions[symbol] = Position(
                        symbol=symbol,
                        quantity=leftover,
                        entry_price=price,
                        mark_price=price,
                        unrealized_pnl=Decimal("0"),
                        realized_pnl=existing.realized_pnl,
                        leverage=None,
                        side=("short" if existing.side == "long" else "long"),
                    )
                else:
                    del self._positions[symbol]
