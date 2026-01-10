"""Structured trace for order decision paths."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass
class OrderDecisionTrace:
    """Captures guard and validation outcomes for an order decision."""

    symbol: str
    side: str
    price: Decimal
    equity: Decimal
    quantity: Decimal | None
    reduce_only: bool
    reason: str
    reduce_only_final: bool | None = None
    bot_id: str | None = None
    outcomes: dict[str, dict[str, Any]] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def record_outcome(
        self,
        stage: str,
        status: str,
        detail: str | None = None,
        **extra: Any,
    ) -> None:
        payload: dict[str, Any] = {"status": status}
        if detail is not None:
            payload["detail"] = detail
        payload.update(extra)
        self.outcomes[stage] = payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "price": str(self.price),
            "equity": str(self.equity),
            "quantity": str(self.quantity) if self.quantity is not None else None,
            "reduce_only": self.reduce_only,
            "reduce_only_final": self.reduce_only_final,
            "reason": self.reason,
            "bot_id": self.bot_id,
            "outcomes": self.outcomes,
            "timestamp": self.timestamp,
        }
