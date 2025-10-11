"""Session helpers powering the legacy live trade facade."""

from __future__ import annotations

from .account import (
    get_account,
    get_account_snapshot,
    get_market_hours,
    get_orders,
    get_positions,
    get_positions_trading,
    get_quote,
)
from .actions import cancel_order, close_all_positions, place_order
from .registry import (
    connect_broker,
    disconnect,
    get_broker_client,
    get_connection,
    get_execution_engine,
    get_risk_manager,
)
from .strategy_demo import run_strategy

__all__ = [
    # Registry
    "connect_broker",
    "disconnect",
    "get_connection",
    "get_broker_client",
    "get_risk_manager",
    "get_execution_engine",
    # Account access
    "get_positions",
    "get_positions_trading",
    "get_account",
    "get_account_snapshot",
    "get_orders",
    "get_quote",
    "get_market_hours",
    # Actions
    "place_order",
    "cancel_order",
    "close_all_positions",
    # Strategy demo
    "run_strategy",
]
