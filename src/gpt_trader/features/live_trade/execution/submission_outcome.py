"""Order submission outcome contract.

Inert result types returned by OrderSubmitter for a broker submission attempt.
Kept separate so consumers can depend on the outcome shape without importing the
order-submission machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class OrderSubmissionOutcomeStatus(str, Enum):
    """Outcome status for broker submission attempts."""

    SUCCESS = "success"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True)
class OrderSubmissionOutcome:
    """Structured result for broker submission attempts."""

    status: OrderSubmissionOutcomeStatus
    order_id: str | None = None
    order: Any | None = None
    reason: str | None = None
    reason_detail: str | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.status is OrderSubmissionOutcomeStatus.SUCCESS

    @property
    def rejected(self) -> bool:
        return self.status is OrderSubmissionOutcomeStatus.REJECTED

    @property
    def failed(self) -> bool:
        return self.status is OrderSubmissionOutcomeStatus.FAILED

    @property
    def error_message(self) -> str | None:
        if self.reason and self.reason_detail:
            return f"{self.reason}:{self.reason_detail}"
        return self.reason or self.error
