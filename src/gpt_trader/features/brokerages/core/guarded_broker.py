"""Guarded broker wrapper to enforce the canonical order submission path."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from collections.abc import Iterator

from gpt_trader.features.brokerages.core.protocols import BrokerProtocol
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="broker_guard")


@dataclass(frozen=True)
class OrderGuardContext:
    mode: str
    reason: str | None = None


_order_guard_context: ContextVar[OrderGuardContext | None] = ContextVar(
    "order_guard_context",
    default=None,
)


@contextmanager
def guarded_order_context(reason: str | None = None) -> Iterator[None]:
    """Mark a broker call as flowing through the guard stack."""
    token = _order_guard_context.set(OrderGuardContext(mode="guarded", reason=reason))
    try:
        yield
    finally:
        _order_guard_context.reset(token)


@contextmanager
def bypass_order_guard(reason: str) -> Iterator[None]:
    """Allow a direct broker call with an explicit bypass reason."""
    token = _order_guard_context.set(OrderGuardContext(mode="bypass", reason=reason))
    try:
        yield
    finally:
        _order_guard_context.reset(token)


def get_order_guard_context() -> OrderGuardContext | None:
    return _order_guard_context.get()


class OrderGuardBypassError(RuntimeError):
    """Raised when a direct broker order is attempted outside the guard stack."""


class GuardedBroker(BrokerProtocol):
    """Broker wrapper that enforces guard-stack order submission."""

    def __init__(self, broker: BrokerProtocol, *, strict: bool = True) -> None:
        self._broker = broker
        self._strict = strict

    def place_order(
        self,
        symbol: str,
        side: Any = None,
        order_type: Any = None,
        quantity: Decimal | None = None,
        **kwargs: Any,
    ) -> Any:
        context = get_order_guard_context()
        if context is None:
            message = (
                "Direct broker.place_order call blocked. "
                "Use TradingEngine.submit_order() for live orders."
            )
            logger.error(
                message,
                symbol=symbol,
                side=str(side),
                order_type=str(order_type),
            )
            if self._strict:
                raise OrderGuardBypassError(message)
        elif context.mode == "bypass":
            logger.warning(
                "Order guard bypassed",
                symbol=symbol,
                side=str(side),
                order_type=str(order_type),
                reason=context.reason,
            )

        return self._broker.place_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            **kwargs,
        )

    def cancel_order(self, order_id: str) -> bool:
        return self._broker.cancel_order(order_id)

    def get_product(self, symbol: str) -> Any:
        return self._broker.get_product(symbol)

    def get_quote(self, symbol: str) -> Any:
        return self._broker.get_quote(symbol)

    def get_ticker(self, product_id: str) -> dict[str, Any]:
        return self._broker.get_ticker(product_id)

    def list_positions(self) -> list[Any]:
        return self._broker.list_positions()

    def list_balances(self) -> list[Any]:
        return self._broker.list_balances()

    def get_candles(self, symbol: str, **kwargs: Any) -> list[Any]:
        return self._broker.get_candles(symbol, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._broker, name)


__all__ = [
    "GuardedBroker",
    "OrderGuardBypassError",
    "bypass_order_guard",
    "guarded_order_context",
    "get_order_guard_context",
]
