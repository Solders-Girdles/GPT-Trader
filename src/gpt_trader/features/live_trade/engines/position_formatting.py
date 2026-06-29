"""Pure position-formatting helpers extracted from ``TradingEngine``.

Transform engine ``Position`` objects into the dict shapes consumed by the
strategy decision layer, the risk manager, and the status reporter. Every
function is pure — it depends only on its arguments, holds no engine state, and
performs no IO. The engine keeps thin ``_``-prefixed delegators.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from gpt_trader.core import OrderSide, Position


def build_position_state(symbol: str, positions: dict[str, Position]) -> dict[str, Any] | None:
    """Build a position-state dict for ``strategy.decide()``."""
    if symbol not in positions:
        return None
    pos = positions[symbol]
    return {
        "quantity": pos.quantity,
        "entry_price": pos.entry_price,
        "side": pos.side,
    }


def resolve_close_order(position_state: dict[str, Any]) -> tuple[OrderSide, Decimal] | None:
    """Derive close side/quantity from a strategy position-state dict."""
    raw_quantity = position_state.get("quantity", Decimal("0"))
    try:
        signed_quantity = Decimal(str(raw_quantity))
    except (InvalidOperation, TypeError, ValueError):
        return None

    close_quantity = abs(signed_quantity)
    if close_quantity <= 0:
        return None

    side_raw = str(position_state.get("side", "")).strip().lower()
    if side_raw == "long":
        return OrderSide.SELL, close_quantity
    if side_raw == "short":
        return OrderSide.BUY, close_quantity

    # Legacy fallback: infer direction from quantity sign when side is missing.
    inferred_side = OrderSide.BUY if signed_quantity < 0 else OrderSide.SELL
    return inferred_side, close_quantity


def positions_to_risk_format(positions: dict[str, Position]) -> dict[str, dict[str, Any]]:
    """Convert ``Position`` objects to the dict format expected by the risk manager."""
    return {
        symbol: {
            "quantity": pos.quantity,
            "mark": pos.mark_price,
        }
        for symbol, pos in positions.items()
    }


def positions_to_status_format(positions: dict[str, Position]) -> dict[str, dict[str, Any]]:
    """Convert ``Position`` objects to the dict format for ``StatusReporter`` (full TUI data)."""
    return {
        symbol: {
            "quantity": str(pos.quantity),
            "mark_price": str(pos.mark_price),
            "entry_price": str(pos.entry_price),
            "unrealized_pnl": str(pos.unrealized_pnl),
            "realized_pnl": str(pos.realized_pnl),
            "side": pos.side,
        }
        for symbol, pos in positions.items()
    }
