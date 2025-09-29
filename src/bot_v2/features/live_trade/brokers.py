"""
Legacy broker stubs retained for the live_trade slice.

The active production broker path is Coinbase (handled in
``bot_v2.orchestration``).  This module now only exposes a simulated
broker that keeps the legacy live_trade entry points and unit tests
working for educational/demo purposes.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any

from ...errors import NetworkError, ValidationError
from ...validation import PositiveNumberValidator, SymbolValidator
from ..brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderType,
    Position,
    Quote,
)
from ..brokerages.core.interfaces import OrderStatus as CoreOrderStatus
from .types import AccountInfo, MarketHours, position_to_trading_position
from bot_v2.types.trading import AccountSnapshot, TradingPosition

logger = logging.getLogger(__name__)


class BrokerInterface(ABC):
    """Minimal interface implemented by the simulated broker."""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the broker."""

    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate the broker connection is active."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the broker."""

    @abstractmethod
    def get_account_id(self) -> str:
        """Return account identifier."""

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Return account information."""

    def get_account_snapshot(self) -> AccountSnapshot:
        """Return shared account snapshot."""
        return self.get_account().to_account_snapshot()

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Return open positions."""

    def get_positions_trading(self) -> list[TradingPosition]:
        """Return shared trading positions."""
        return [position_to_trading_position(pos) for pos in self.get_positions()]

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal | int,
        order_type: OrderType,
        *,
        limit_price: Decimal | float | None = None,
        stop_price: Decimal | float | None = None,
        time_in_force: str = "day",
        **kwargs: Any,
    ) -> Order:
        """Place an order with optional price parameters."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order if still open."""

    @abstractmethod
    def get_orders(self, status: str = "open") -> list[Order]:
        """Return orders filtered by ``status``."""

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Return a quote for ``symbol``."""

    @abstractmethod
    def get_market_hours(self) -> MarketHours:
        """Return current market hours state."""


class SimulatedBroker(BrokerInterface):
    """Lightweight broker used for unit tests and walkthroughs."""

    def __init__(self) -> None:
        self.connected: bool = False
        self.account_id = "SIM_001"
        self.cash: float = 100_000.0
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.order_counter: int = 0

    def connect(self) -> bool:
        self.connected = True
        logger.info("Connected to simulated broker")
        return True

    def validate_connection(self) -> bool:
        if not self.connected:
            raise NetworkError("Not connected to simulated broker")
        return True

    def disconnect(self) -> None:
        self.connected = False

    def get_account_id(self) -> str:
        return self.account_id

    def get_account(self) -> AccountInfo:
        positions_value = sum(
            float(position.mark_price * position.quantity) for position in self.positions.values()
        )
        equity = self.cash + positions_value

        return AccountInfo(
            account_id=self.account_id,
            cash=self.cash,
            portfolio_value=equity,
            buying_power=self.cash * 2,
            positions_value=positions_value,
            margin_used=positions_value,
            pattern_day_trader=False,
            day_trades_remaining=3,
            equity=equity,
            last_equity=equity,
        )

    def get_positions(self) -> list[Position]:
        return list(self.positions.values())

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal | int,
        order_type: OrderType,
        *,
        limit_price: Decimal | float | None = None,
        stop_price: Decimal | float | None = None,
        time_in_force: str = "day",
        **_: Any,
    ) -> Order:
        self.validate_connection()
        SymbolValidator().validate(symbol, "symbol")
        quantity_decimal = quantity if isinstance(quantity, Decimal) else Decimal(str(quantity))
        PositiveNumberValidator(allow_zero=False).validate(float(quantity_decimal), "quantity")

        from .adapters import to_core_tif  # Local import avoids cycles

        self.order_counter += 1
        order_id = f"SIM_{self.order_counter:06d}"

        limit_decimal = Decimal(str(limit_price)) if limit_price is not None else None
        stop_decimal = Decimal(str(stop_price)) if stop_price is not None else None

        if order_type == OrderType.MARKET:
            fill_price = Decimal(str(100 + random.uniform(-5, 5)))
            order = Order(
                id=order_id,
                client_id=None,
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity_decimal,
                price=limit_decimal,
                stop_price=stop_decimal,
                tif=to_core_tif(time_in_force),
                status=CoreOrderStatus.FILLED,
                filled_quantity=quantity_decimal,
                avg_fill_price=fill_price,
                submitted_at=datetime.now(),
                updated_at=datetime.now(),
            )
            self._apply_fill(symbol, side, quantity_decimal, fill_price)
        else:
            order = Order(
                id=order_id,
                client_id=None,
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity_decimal,
                price=limit_decimal,
                stop_price=stop_decimal,
                tif=to_core_tif(time_in_force),
                status=CoreOrderStatus.SUBMITTED,
                filled_quantity=Decimal("0"),
                avg_fill_price=None,
                submitted_at=datetime.now(),
                updated_at=datetime.now(),
            )

        self.orders.append(order)
        return order

    def _apply_fill(self, symbol: str, side: OrderSide, quantity: Decimal, price: Decimal) -> None:
        if side == OrderSide.BUY:
            position = self.positions.get(symbol)
            if position:
                weighted_cost = position.entry_price * position.quantity + price * quantity
                new_quantity = position.quantity + quantity
                position.quantity = new_quantity
                position.entry_price = weighted_cost / new_quantity
                position.mark_price = price
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    mark_price=price,
                    unrealized_pnl=Decimal("0"),
                    realized_pnl=Decimal("0"),
                    leverage=None,
                    side="long",
                )
            self.cash -= float(price * quantity) + float(quantity) * 0.01
        else:
            position = self.positions.get(symbol)
            if position:
                if quantity >= position.quantity:
                    del self.positions[symbol]
                else:
                    position.quantity -= quantity
                    position.mark_price = price
            self.cash += float(price * quantity)

    def cancel_order(self, order_id: str) -> bool:
        cancellable = {
            CoreOrderStatus.PENDING,
            CoreOrderStatus.SUBMITTED,
            CoreOrderStatus.PARTIALLY_FILLED,
        }
        for order in self.orders:
            if order.id == order_id and order.status in cancellable:
                order.status = CoreOrderStatus.CANCELLED
                return True
        return False

    def get_orders(self, status: str = "open") -> list[Order]:
        if status == "open":
            return [
                order
                for order in self.orders
                if order.status
                in {
                    CoreOrderStatus.PENDING,
                    CoreOrderStatus.SUBMITTED,
                    CoreOrderStatus.PARTIALLY_FILLED,
                }
            ]
        if status == "closed":
            return [
                order
                for order in self.orders
                if order.status
                in {
                    CoreOrderStatus.FILLED,
                    CoreOrderStatus.CANCELLED,
                    CoreOrderStatus.REJECTED,
                }
            ]
        return list(self.orders)

    def get_quote(self, symbol: str) -> Quote:
        base_price = 100.0 + random.uniform(-10, 10)
        spread = 0.02
        return Quote(
            symbol=symbol,
            bid=Decimal(str(base_price - spread / 2)),
            ask=Decimal(str(base_price + spread / 2)),
            last=Decimal(str(base_price)),
            ts=datetime.now(),
        )

    def get_market_hours(self) -> MarketHours:
        now = datetime.now()
        is_weekday = now.weekday() < 5
        return MarketHours(
            is_open=is_weekday and 9 <= now.hour < 16,
            open_time=now.replace(hour=9, minute=30),
            close_time=now.replace(hour=16, minute=0),
            extended_hours_open=is_weekday and 4 <= now.hour < 20,
        )


__all__ = ["BrokerInterface", "SimulatedBroker"]
