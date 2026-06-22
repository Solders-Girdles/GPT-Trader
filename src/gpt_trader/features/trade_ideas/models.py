"""Broker-neutral trade-idea records.

Implements the record contract from the accepted Pre-Migration Decision
Framework (docs/PRE_MIGRATION_DECISION_FRAMEWORK.md): every AI-generated idea
carries its thesis, entry zone, invalidation, max loss, sizing, horizon, and
confidence before any human review. Broker payloads are derived artifacts
created after approval; nothing in this module is venue-specific beyond the
ticket envelope.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any


class AutonomyMode(str, Enum):
    RESEARCH_ONLY = "research_only"
    HUMAN_APPROVED_EXECUTION = "human_approved_execution"
    BOUNDED_AUTONOMY = "bounded_autonomy"


class ProductType(str, Enum):
    SPOT = "spot"
    FUTURES = "futures"
    OPTIONS = "options"
    EVENT_CONTRACT = "event_contract"
    OTHER = "other"


class TradeDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    SPREAD = "spread"
    HEDGE = "hedge"
    NO_TRADE = "no_trade"


class ConfidenceLabel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TicketVenue(str, Enum):
    COINBASE = "coinbase"
    MANUAL = "manual"
    NONE = "none"


class TicketStatus(str, Enum):
    NOT_CREATED = "not_created"
    DRAFTED = "drafted"
    APPROVED = "approved"
    SUBMITTED = "submitted"
    CANCELLED = "cancelled"


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _decimal_to_str(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _require_timezone_aware(value: datetime | None, field: str) -> None:
    if value is None:
        return
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")


def _parse_expires_at(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("time_horizon.expires_at must be an ISO datetime string")
    expires_at = datetime.fromisoformat(value)
    _require_timezone_aware(expires_at, "time_horizon.expires_at")
    return expires_at


@dataclass(frozen=True, slots=True)
class EntryZone:
    """Price range or conditional trigger that defines a valid entry."""

    lower: Decimal | None = None
    upper: Decimal | None = None
    trigger: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": _decimal_to_str(self.lower),
            "upper": _decimal_to_str(self.upper),
            "trigger": self.trigger,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EntryZone:
        return cls(
            lower=_decimal_or_none(payload.get("lower")),
            upper=_decimal_or_none(payload.get("upper")),
            trigger=payload.get("trigger", ""),
        )


@dataclass(frozen=True, slots=True)
class MaxLoss:
    """Dollar and percent loss estimate, including assumptions."""

    amount: Decimal | None = None
    percent_of_account: Decimal | None = None
    assumptions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "amount": _decimal_to_str(self.amount),
            "percent_of_account": _decimal_to_str(self.percent_of_account),
            "assumptions": list(self.assumptions),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MaxLoss:
        return cls(
            amount=_decimal_or_none(payload.get("amount")),
            percent_of_account=_decimal_or_none(payload.get("percent_of_account")),
            assumptions=tuple(payload.get("assumptions", ())),
        )


@dataclass(frozen=True, slots=True)
class SizingRecommendation:
    """Proposed size and how it was derived."""

    quantity: Decimal | None = None
    notional: Decimal | None = None
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantity": _decimal_to_str(self.quantity),
            "notional": _decimal_to_str(self.notional),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SizingRecommendation:
        return cls(
            quantity=_decimal_or_none(payload.get("quantity")),
            notional=_decimal_or_none(payload.get("notional")),
            rationale=payload.get("rationale", ""),
        )


@dataclass(frozen=True, slots=True)
class TimeHorizon:
    """Expected holding period plus a hard review/expiry deadline."""

    expected_hold: str = ""
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_timezone_aware(self.expires_at, "time_horizon.expires_at")

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_hold": self.expected_hold,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TimeHorizon:
        return cls(
            expected_hold=payload.get("expected_hold", ""),
            expires_at=_parse_expires_at(payload.get("expires_at")),
        )


@dataclass(frozen=True, slots=True)
class Confidence:
    """Bounded confidence label plus why it may be wrong."""

    label: ConfidenceLabel
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label.value, "rationale": self.rationale}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Confidence:
        return cls(
            label=ConfidenceLabel(payload["label"]),
            rationale=payload.get("rationale", ""),
        )


@dataclass(frozen=True, slots=True)
class BrokerTicket:
    """Venue envelope for the derived broker payload; broker-neutral by default."""

    venue: TicketVenue = TicketVenue.NONE
    status: TicketStatus = TicketStatus.NOT_CREATED

    def to_dict(self) -> dict[str, Any]:
        return {"venue": self.venue.value, "status": self.status.value}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BrokerTicket:
        return cls(
            venue=TicketVenue(payload.get("venue", TicketVenue.NONE.value)),
            status=TicketStatus(payload.get("status", TicketStatus.NOT_CREATED.value)),
        )


@dataclass(frozen=True, slots=True)
class TradeIdea:
    """Immutable trade-idea record.

    Workflow state is intentionally not a field here: state lives in the
    append-only audit log (see audit.py) so the original thesis is never
    overwritten when reviewers change or approve a ticket.
    """

    decision_id: str
    autonomy_mode: AutonomyMode
    thesis: str
    instrument: str
    product_type: ProductType
    direction: TradeDirection
    entry_zone: EntryZone
    invalidation: str
    target_exit: str
    max_loss: MaxLoss
    sizing_recommendation: SizingRecommendation
    time_horizon: TimeHorizon
    data_used: tuple[str, ...]
    confidence: Confidence
    failure_mode: str
    do_not_trade_if: tuple[str, ...] = ()
    broker_ticket: BrokerTicket = field(default_factory=BrokerTicket)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "autonomy_mode": self.autonomy_mode.value,
            "thesis": self.thesis,
            "instrument": self.instrument,
            "product_type": self.product_type.value,
            "direction": self.direction.value,
            "entry_zone": self.entry_zone.to_dict(),
            "invalidation": self.invalidation,
            "target_exit": self.target_exit,
            "max_loss": self.max_loss.to_dict(),
            "sizing_recommendation": self.sizing_recommendation.to_dict(),
            "time_horizon": self.time_horizon.to_dict(),
            "data_used": list(self.data_used),
            "confidence": self.confidence.to_dict(),
            "failure_mode": self.failure_mode,
            "do_not_trade_if": list(self.do_not_trade_if),
            "broker_ticket": self.broker_ticket.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TradeIdea:
        return cls(
            decision_id=payload["decision_id"],
            autonomy_mode=AutonomyMode(payload["autonomy_mode"]),
            thesis=payload["thesis"],
            instrument=payload["instrument"],
            product_type=ProductType(payload["product_type"]),
            direction=TradeDirection(payload["direction"]),
            entry_zone=EntryZone.from_dict(payload.get("entry_zone", {})),
            invalidation=payload["invalidation"],
            target_exit=payload["target_exit"],
            max_loss=MaxLoss.from_dict(payload.get("max_loss", {})),
            sizing_recommendation=SizingRecommendation.from_dict(
                payload.get("sizing_recommendation", {})
            ),
            time_horizon=TimeHorizon.from_dict(payload.get("time_horizon", {})),
            data_used=tuple(payload.get("data_used", ())),
            confidence=Confidence.from_dict(payload["confidence"]),
            failure_mode=payload["failure_mode"],
            do_not_trade_if=tuple(payload.get("do_not_trade_if", ())),
            broker_ticket=BrokerTicket.from_dict(payload.get("broker_ticket", {})),
        )

    def record_hash(self) -> str:
        """Stable content hash used by audit events to pin the record version."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
