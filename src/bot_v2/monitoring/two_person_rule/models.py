"""Data models for the two-person approval workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class ChangeType(Enum):
    """Types of configuration changes requiring approval."""

    RISK_LIMIT = "risk_limit"
    LEVERAGE = "leverage"
    POSITION_SIZE = "position_size"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    LIQUIDATION_BUFFER = "liquidation_buffer"
    CIRCUIT_BREAKER = "circuit_breaker"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    SYMBOL_UNIVERSE = "symbol_universe"
    PROFILE_CHANGE = "profile_change"


class ApprovalStatus(Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    APPLIED = "applied"


@dataclass
class ConfigChange:
    """Configuration change with before/after values."""

    change_type: ChangeType
    field_name: str
    old_value: Any
    new_value: Any
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "change_type": self.change_type.value,
            "field_name": self.field_name,
            "old_value": self._stringify(self.old_value),
            "new_value": self._stringify(self.new_value),
            "description": self.description,
        }

    @staticmethod
    def _stringify(value: Any) -> str:
        try:
            return json.dumps(value)
        except Exception:
            return str(value)


@dataclass
class ApprovalRequest:
    """Two-person approval request."""

    request_id: str
    requester_id: str
    requester_name: str
    changes: list[ConfigChange]
    status: ApprovalStatus
    created_at: datetime
    expires_at: datetime
    approver_id: str | None = None
    approver_name: str | None = None
    approved_at: datetime | None = None
    rejection_reason: str | None = None
    applied_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if the request has expired."""
        return datetime.now(UTC) >= self.expires_at and self.status == ApprovalStatus.PENDING

    @property
    def requires_approval(self) -> bool:
        """Check if the request still requires approval."""
        return self.status == ApprovalStatus.PENDING

    def mark_expired(self) -> None:
        """Update state to expired when appropriate."""
        if self.status == ApprovalStatus.PENDING and datetime.now(UTC) >= self.expires_at:
            self.status = ApprovalStatus.EXPIRED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "request_id": self.request_id,
            "requester_id": self.requester_id,
            "requester_name": self.requester_name,
            "changes": [c.to_dict() for c in self.changes],
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "approver_id": self.approver_id,
            "approver_name": self.approver_name,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "rejection_reason": self.rejection_reason,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "metadata": self.metadata,
        }


__all__ = [
    "ApprovalRequest",
    "ApprovalStatus",
    "ChangeType",
    "ConfigChange",
]
