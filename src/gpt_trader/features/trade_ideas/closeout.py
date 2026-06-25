"""Broker-neutral closeout attribution for terminal trade ideas.

Closeout attribution is intentionally separate from the lifecycle audit log:
terminal workflow states stay terminal, while this append-only log records why
the idea resolved and how realized profit/loss compares with the original
max-loss snapshot.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from gpt_trader.errors import ValidationError


class CloseoutResolution(str, Enum):
    THESIS_TARGET = "thesis_target"
    INVALIDATION = "invalidation"
    EXPIRY = "expiry"


class DuplicateCloseoutAttributionError(ValidationError):
    """Raised when a decision already has conflicting closeout attribution."""


class CloseoutAttributionIntegrityError(ValidationError):
    """Raised when a closeout attribution record is malformed."""


def _decimal_or_none(value: Any, field: str) -> Decimal | None:
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except Exception as error:
        raise ValueError(f"{field} must be a finite decimal") from error
    if not parsed.is_finite():
        raise ValueError(f"{field} must be finite")
    return parsed


def _decimal_to_str(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _string_sequence(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a JSON array of strings")
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{field}[{index}] must be a string")
    return tuple(value)


def _string_value(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _require_timezone_aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")


@dataclass(frozen=True, slots=True)
class MaxLossSnapshot:
    amount: Decimal | None = None
    percent_of_account: Decimal | None = None
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount", _decimal_or_none(self.amount, "max_loss.amount"))
        object.__setattr__(
            self,
            "percent_of_account",
            _decimal_or_none(self.percent_of_account, "max_loss.percent_of_account"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "amount": _decimal_to_str(self.amount),
            "percent_of_account": _decimal_to_str(self.percent_of_account),
            "assumptions": list(self.assumptions),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MaxLossSnapshot:
        return cls(
            amount=_decimal_or_none(payload.get("amount"), "max_loss.amount"),
            percent_of_account=_decimal_or_none(
                payload.get("percent_of_account"), "max_loss.percent_of_account"
            ),
            assumptions=_string_sequence(payload.get("assumptions", ()), "max_loss.assumptions"),
        )


@dataclass(frozen=True, slots=True)
class CloseoutAttribution:
    decision_id: str
    timestamp: datetime
    actor_type: str
    actor_id: str
    terminal_event_id: str
    record_hash: str
    resolution: CloseoutResolution
    max_loss: MaxLossSnapshot
    realized_profit_loss_amount: Decimal | None = None
    realized_profit_loss_percent: Decimal | None = None
    realized_profit_loss_unavailable_reason: str = ""
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_timezone_aware(self.timestamp, "timestamp")
        object.__setattr__(
            self,
            "realized_profit_loss_amount",
            _decimal_or_none(
                self.realized_profit_loss_amount,
                "realized_profit_loss_amount",
            ),
        )
        object.__setattr__(
            self,
            "realized_profit_loss_percent",
            _decimal_or_none(
                self.realized_profit_loss_percent,
                "realized_profit_loss_percent",
            ),
        )
        if (
            self.realized_profit_loss_amount is None
            and self.realized_profit_loss_percent is None
            and not self.realized_profit_loss_unavailable_reason.strip()
        ):
            raise ValueError(
                "realized profit/loss requires amount, percent, or an unavailable reason"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "terminal_event_id": self.terminal_event_id,
            "record_hash": self.record_hash,
            "resolution": self.resolution.value,
            "realized_profit_loss_amount": _decimal_to_str(self.realized_profit_loss_amount),
            "realized_profit_loss_percent": _decimal_to_str(self.realized_profit_loss_percent),
            "realized_profit_loss_unavailable_reason": (
                self.realized_profit_loss_unavailable_reason
            ),
            "max_loss": self.max_loss.to_dict(),
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CloseoutAttribution:
        return cls(
            decision_id=_string_value(payload["decision_id"], "decision_id"),
            timestamp=datetime.fromisoformat(_string_value(payload["timestamp"], "timestamp")),
            actor_type=_string_value(payload["actor_type"], "actor_type"),
            actor_id=_string_value(payload["actor_id"], "actor_id"),
            terminal_event_id=_string_value(
                payload["terminal_event_id"],
                "terminal_event_id",
            ),
            record_hash=_string_value(payload["record_hash"], "record_hash"),
            resolution=CloseoutResolution(payload["resolution"]),
            realized_profit_loss_amount=_decimal_or_none(
                payload.get("realized_profit_loss_amount"),
                "realized_profit_loss_amount",
            ),
            realized_profit_loss_percent=_decimal_or_none(
                payload.get("realized_profit_loss_percent"),
                "realized_profit_loss_percent",
            ),
            realized_profit_loss_unavailable_reason=_string_value(
                payload.get("realized_profit_loss_unavailable_reason", ""),
                "realized_profit_loss_unavailable_reason",
            ),
            max_loss=MaxLossSnapshot.from_dict(payload["max_loss"]),
            evidence=_string_sequence(payload.get("evidence", ()), "evidence"),
        )


class CloseoutAttributionLog:
    """Append-only JSONL log keyed by decision id."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: CloseoutAttribution) -> CloseoutAttribution:
        existing = self.get(record.decision_id)
        if existing is not None:
            if existing == record:
                return existing
            raise DuplicateCloseoutAttributionError(
                f"Closeout attribution already exists for decision_id '{record.decision_id}'",
                field="decision_id",
                value=record.decision_id,
            )

        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return record

    def get(self, decision_id: str) -> CloseoutAttribution | None:
        for record in self.read_records(decision_id):
            return record
        return None

    def read_records(self, decision_id: str | None = None) -> list[CloseoutAttribution]:
        if not self._path.exists():
            return []
        records: list[CloseoutAttribution] = []
        seen: dict[str, CloseoutAttribution] = {}
        with self._path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = CloseoutAttribution.from_dict(json.loads(line))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise CloseoutAttributionIntegrityError(
                        f"Closeout attribution log line {line_number} is malformed: {error}",
                        field="line",
                        value=line_number,
                    ) from error
                existing = seen.get(record.decision_id)
                if existing is not None and existing != record:
                    raise DuplicateCloseoutAttributionError(
                        f"Closeout attribution log line {line_number} conflicts for "
                        f"decision_id '{record.decision_id}'",
                        field="decision_id",
                        value=record.decision_id,
                    )
                seen[record.decision_id] = record
                if decision_id is None or record.decision_id == decision_id:
                    records.append(record)
        return records
