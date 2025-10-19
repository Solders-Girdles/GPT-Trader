"""Shared helpers for Coinbase PnL comparison tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pytest

from bot_v2.features.live_trade.pnl_tracker import (
    FundingCalculator,
    PnLTracker,
    PositionState,
)


def ensure_advanced_pnl_available() -> None:
    """Skip the calling test module if the advanced PnL API is unavailable."""

    position_methods = ["update_mark"]
    tracker_attrs = [
        "update_position",
        "update_marks",
        "accrue_funding",
        "get_total_pnl",
    ]
    funding_methods = [
        "calculate_funding",
        "is_funding_due",
        "accrue_if_due",
    ]

    def _has_all(obj, names) -> bool:
        return all(hasattr(obj, name) for name in names)

    if not (
        _has_all(PositionState, position_methods)
        and _has_all(PnLTracker, tracker_attrs)
        and _has_all(FundingCalculator, funding_methods)
    ):
        pytest.skip(
            "Skipping comprehensive PnL tests: Advanced PnL API not available in this build",
            allow_module_level=True,
        )


@dataclass(frozen=True)
class TradeOp:
    symbol: str
    side: str
    size: Decimal
    price: Decimal
    is_reduce: bool = False


def apply_trades(tracker: PnLTracker, trades: Iterable[TradeOp]) -> list[dict | None]:
    """Apply a sequence of trades to the tracker and collect results."""
    results: list[dict | None] = []
    for trade in trades:
        result = tracker.update_position(
            trade.symbol,
            trade.side,
            trade.size,
            trade.price,
            is_reduce=trade.is_reduce,
        )
        results.append(result)
    return results


def apply_marks(tracker: PnLTracker, marks: Mapping[str, Decimal]) -> None:
    """Update mark prices in bulk for the provided tracker."""
    tracker.update_marks(marks)


def make_position(
    *,
    product_id: str,
    side: str,
    entry_price: Decimal,
    size: Decimal,
    timestamp=None,
) -> PositionState:
    """Create a :class:`PositionState` with sensible defaults for tests."""

    position = PositionState(symbol=product_id)
    position.side = side
    position.quantity = size
    position.avg_entry_price = entry_price
    position.realized_pnl = Decimal("0")
    position.unrealized_pnl = Decimal("0")
    position.last_funding_time = timestamp or datetime.now()
    return position
