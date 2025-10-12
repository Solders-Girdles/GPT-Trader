"""Smoke tests ensuring the live_trade shim re-exports the session API."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "attribute",
    [
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
    ],
)
def test_live_trade_module_reexports_session_api(attribute: str) -> None:
    shim = importlib.import_module("bot_v2.features.live_trade.live_trade")
    session = importlib.import_module("bot_v2.features.live_trade.session")

    assert hasattr(shim, attribute)
    assert getattr(shim, attribute) is getattr(session, attribute)
