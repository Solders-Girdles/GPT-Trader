"""
Simplified legacy live trading interface.

This module now defers to :mod:`bot_v2.features.live_trade.session` for the
core implementations so we maintain a single source of truth for legacy flows
while keeping the thin convenience wrapper that tutorials import.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, cast

from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderType,
    Position,
    Quote,
    TimeInForce,
)
from bot_v2.features.live_trade.session import (
    cancel_order as _cancel_order,
)
from bot_v2.features.live_trade.session import (
    close_all_positions as _close_all_positions,
)
from bot_v2.features.live_trade.session import (
    connect_broker as _connect_broker,
)
from bot_v2.features.live_trade.session import (
    disconnect as _disconnect,
)
from bot_v2.features.live_trade.session import (
    get_account as _get_account,
)
from bot_v2.features.live_trade.session import (
    get_account_snapshot as _get_account_snapshot,
)
from bot_v2.features.live_trade.session import (
    get_broker_client,
    get_connection,
    get_risk_manager,
)
from bot_v2.features.live_trade.session import (
    get_market_hours as _get_market_hours,
)
from bot_v2.features.live_trade.session import (
    get_orders as _get_orders,
)
from bot_v2.features.live_trade.session import (
    get_positions as _get_positions,
)
from bot_v2.features.live_trade.session import (
    get_positions_trading as _get_positions_trading,
)
from bot_v2.features.live_trade.session import (
    get_quote as _get_quote,
)
from bot_v2.features.live_trade.session import (
    place_order as _place_order,
)
from bot_v2.features.live_trade.session import (
    run_strategy as _run_strategy,
)
from bot_v2.features.live_trade.types import (
    AccountInfo,
    BrokerConnection,
    MarketHours,
)
from bot_v2.types.trading import AccountSnapshot, TradingPosition


def connect_broker(
    broker_name: str = "simulated",
    api_key: str = "",
    api_secret: str = "",
    is_paper: bool = True,
    base_url: str | None = None,
) -> BrokerConnection:
    """Connect to the simulated broker stub used by legacy demos."""
    return _connect_broker(
        broker_name=broker_name,
        api_key=api_key,
        api_secret=api_secret,
        is_paper=is_paper,
        base_url=base_url,
    )


def place_order(
    symbol: str,
    side: OrderSide,
    quantity: Decimal | int,
    order_type: OrderType = OrderType.MARKET,
    limit_price: Decimal | float | None = None,
    stop_price: Decimal | float | None = None,
    time_in_force: TimeInForce = TimeInForce.GTC,
) -> Order | None:
    """Place an order using the shared session implementation."""
    order = _place_order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        time_in_force=time_in_force,
    )
    return order


def cancel_order(order_id: str) -> bool:
    """Cancel an order with standard session handling."""
    return _cancel_order(order_id)


def get_positions() -> list[Position]:
    """Return current positions from the shared session helper."""
    positions = _get_positions()
    return cast(list[Position], positions)


def get_positions_trading() -> list[TradingPosition]:
    """Return current positions using the shared trading type schema."""
    return _get_positions_trading()


def get_account() -> AccountInfo | None:
    """Get account information using the shared session helper."""
    return _get_account()


def get_account_snapshot() -> AccountSnapshot | None:
    """Return the active account as a shared account snapshot."""
    return _get_account_snapshot()


def get_orders(status: str = "open") -> list[Order]:
    """Return orders for the active broker session."""
    return _get_orders(status)


def get_quote(symbol: str) -> Quote | None:
    """Return a real-time quote for the provided symbol."""
    return _get_quote(symbol)


def get_market_hours() -> MarketHours:
    """Return market hours information from the active broker."""
    return _get_market_hours()


def close_all_positions() -> bool:
    """Close all open positions using the shared session helper."""
    return _close_all_positions()


def disconnect() -> None:
    """Disconnect from the broker and clean up session state."""
    _disconnect()


def run_strategy(
    symbols: list[str],
    strategy_name: str = "baseline_perps",
    iterations: int = 3,
    mark_cache: dict[str, Decimal] | None = None,
    mark_windows: dict[str, list[Decimal]] | None = None,
    *,
    strategy_override: Any | None = None,
) -> dict[str, Any]:
    """Run the legacy demo strategy via the session helper."""
    return _run_strategy(
        symbols=symbols,
        strategy_name=strategy_name,
        iterations=iterations,
        mark_cache=mark_cache,
        mark_windows=mark_windows,
        strategy_override=strategy_override,
    )


__all__ = [
    "connect_broker",
    "disconnect",
    "place_order",
    "cancel_order",
    "get_orders",
    "close_all_positions",
    "get_positions",
    "get_positions_trading",
    "get_account",
    "get_account_snapshot",
    "get_quote",
    "get_market_hours",
    "run_strategy",
    "get_connection",
    "get_broker_client",
    "get_risk_manager",
]
