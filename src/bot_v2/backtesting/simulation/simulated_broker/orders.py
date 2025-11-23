from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from bot_v2.features.brokerages.core.interfaces import (
    InsufficientFunds,
    InvalidRequestError,
    NotFoundError,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)

from .market import MarketState
from .portfolio import PortfolioManager


class OrderEngine:
    def __init__(
        self,
        portfolio: PortfolioManager,
        market_state: MarketState,
        fill_model: Any,
    ) -> None:
        self._portfolio = portfolio
        self._market_state = market_state
        self._fill_model = fill_model

        self._orders: Dict[str, Order] = {}
        self._open_orders: Dict[str, Order] = {}

    def place_order(
        self,
        *,
        current_time: datetime,
        symbol: str,
        side: str,
        order_type: str,
        quantity: str,
        limit_price: str | None,
        stop_price: str | None,
        time_in_force: str | None,
        post_only: bool,
        reduce_only: bool,
        client_order_id: str | None,
    ) -> Order:
        try:
            quantity_decimal = Decimal(quantity)
            side_enum = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            type_enum = OrderType[order_type.upper()]
            tif = TimeInForce[time_in_force.upper()] if time_in_force else TimeInForce.GTC
            limit_px = Decimal(limit_price) if limit_price else None
            stop_px = Decimal(stop_price) if stop_price else None
        except (ValueError, KeyError) as exc:
            raise InvalidRequestError(f"Invalid order parameters: {exc}") from exc

        try:
            product = self._portfolio.get_product(symbol)
        except KeyError as exc:
            raise InvalidRequestError(f"Unknown product: {symbol}") from exc

        if quantity_decimal < product.min_size:
            raise InvalidRequestError(
                f"Order size {quantity_decimal} below minimum {product.min_size}"
            )

        quote = self._market_state.get_quote(symbol)
        notional = quantity_decimal * (limit_px or quote.last)
        if side_enum == OrderSide.BUY and not self._portfolio.has_sufficient_margin(notional):
            raise InsufficientFunds(f"Insufficient funds for order (need {notional})")

        order_id = str(uuid.uuid4())
        order = Order(
            id=order_id,
            client_id=client_order_id,
            symbol=symbol,
            side=side_enum,
            type=type_enum,
            quantity=quantity_decimal,
            price=limit_px,
            stop_price=stop_px,
            tif=tif,
            status=OrderStatus.PENDING,
            submitted_at=current_time,
            updated_at=current_time,
            post_only=post_only,
            reduce_only=reduce_only,
        )

        self._orders[order_id] = order

        if type_enum == OrderType.MARKET:
            self._fill_market_order(order)
        else:
            self._open_orders[order_id] = order
            order.status = OrderStatus.SUBMITTED

        return order

    def cancel_order(self, order_id: str, current_time: datetime) -> bool:
        if order_id not in self._orders:
            raise NotFoundError(f"Order not found: {order_id}")

        order = self._orders[order_id]
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return False

        order.status = OrderStatus.CANCELLED
        order.updated_at = current_time
        self._open_orders.pop(order_id, None)
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def list_orders(self, status: str | None = None, symbol: str | None = None) -> List[Order]:
        orders = list(self._orders.values())
        if status:
            status_enum = OrderStatus[status.upper()]
            orders = [o for o in orders if o.status == status_enum]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def list_fills(self, symbol: str | None = None, limit: int = 100) -> List[dict]:
        return []

    def process_pending_orders(self) -> None:
        filled_ids: List[str] = []
        for order_id, order in list(self._open_orders.items()):
            symbol = order.symbol
            current_bar = self._market_state.get_bar(symbol)
            if current_bar is None:
                continue
            quote = self._market_state.get_quote(symbol)

            if order.type == OrderType.LIMIT:
                fill_result = self._fill_model.try_fill_limit_order(
                    order=order,
                    current_bar=current_bar,
                    best_bid=quote.bid,
                    best_ask=quote.ask,
                )
            elif order.type in (OrderType.STOP, OrderType.STOP_LIMIT):
                fill_result = self._fill_model.try_fill_stop_order(
                    order=order,
                    current_bar=current_bar,
                    best_bid=quote.bid,
                    best_ask=quote.ask,
                    next_bar=self._market_state.get_next_bar(symbol),
                )
            else:
                continue

            if fill_result.filled:
                self._portfolio.execute_fill(
                    order=order,
                    fill_price=fill_result.fill_price,
                    fill_quantity=fill_result.fill_quantity,
                    is_maker=fill_result.is_maker,
                    current_time=self._market_state.current_time,
                )
                filled_ids.append(order_id)

        for order_id in filled_ids:
            self._open_orders.pop(order_id, None)

    def _fill_market_order(self, order: Order) -> None:
        symbol = order.symbol
        quote = self._market_state.get_quote(symbol)
        current_bar = self._market_state.get_bar(symbol)

        fill_result = self._fill_model.fill_market_order(
            order=order,
            current_bar=current_bar,
            best_bid=quote.bid,
            best_ask=quote.ask,
            next_bar=self._market_state.get_next_bar(symbol),
        )
        if fill_result.filled:
            self._portfolio.execute_fill(
                order=order,
                fill_price=fill_result.fill_price,
                fill_quantity=fill_result.fill_quantity,
                is_maker=fill_result.is_maker,
                current_time=self._market_state.current_time,
            )


__all__ = ["OrderEngine"]
