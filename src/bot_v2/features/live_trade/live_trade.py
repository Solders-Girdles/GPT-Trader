"""
Legacy live trading orchestration retained for compatibility demos.

Implementation now lives in ``bot_v2.features.live_trade.session`` so that
historical imports continue to function while the codebase is cleaned up.
"""

from __future__ import annotations

from bot_v2.features.brokerages.core.interfaces import Order, Position, Quote
from bot_v2.features.live_trade.session import (
    cancel_order,
    close_all_positions,
    connect_broker,
    disconnect,
    get_account,
    get_account_snapshot,
    get_broker_client,
    get_connection,
    get_execution_engine,
    get_market_hours,
    get_orders,
    get_positions,
    get_positions_trading,
    get_quote,
    get_risk_manager,
    place_order,
    run_strategy,
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
    "get_execution_engine",
    "Order",
    "Position",
    "Quote",
]
