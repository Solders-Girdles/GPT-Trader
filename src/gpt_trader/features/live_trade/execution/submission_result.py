"""Typed results for order submission outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gpt_trader.features.live_trade.execution.decision_trace import OrderDecisionTrace


class OrderSubmissionStatus(str, Enum):
    """Submission outcome status."""

    SUCCESS = "success"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True)
class OrderSubmissionResult:
    """Typed result for guard + submission flow outcomes."""

    status: OrderSubmissionStatus
    order_id: str | None = None
    reason: str | None = None
    error: str | None = None
    decision_trace: OrderDecisionTrace | None = None

    @property
    def success(self) -> bool:
        return self.status is OrderSubmissionStatus.SUCCESS

    @property
    def blocked(self) -> bool:
        return self.status is OrderSubmissionStatus.BLOCKED

    @property
    def failed(self) -> bool:
        return self.status is OrderSubmissionStatus.FAILED
