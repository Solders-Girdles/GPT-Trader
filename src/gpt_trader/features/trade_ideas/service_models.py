"""Trade-idea read-model contracts.

Inert data types shared by the trade-idea service and its review adapters: the
list/query/result/view value objects and the not-found / duplicate / pre-approval
errors. The audited service logic that produces them lives in service.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Generic, TypeVar

from gpt_trader.errors import ValidationError
from gpt_trader.features.trade_ideas.audit import AuditEvent
from gpt_trader.features.trade_ideas.closeout import CloseoutAttribution
from gpt_trader.features.trade_ideas.models import (
    ConfidenceLabel,
    TradeDirection,
    TradeIdea,
)
from gpt_trader.features.trade_ideas.workflow import TradeIdeaState

_QueryItem = TypeVar("_QueryItem")


class UnknownTradeIdeaError(ValidationError):
    """Raised when a decision_id has no stored record."""


class DuplicateTradeIdeaError(ValidationError):
    """Raised when a new proposal reuses an existing decision_id."""


class PreApprovalBrokerTicketError(ValidationError):
    """Raised when a proposed record carries broker-specific ticket state."""


@dataclass(frozen=True, slots=True)
class TradeIdeaView:
    """A record plus its derived workflow state and full history."""

    idea: TradeIdea
    state: TradeIdeaState
    events: tuple[AuditEvent, ...]
    closeout_attribution: CloseoutAttribution | None = None


@dataclass(frozen=True, slots=True)
class TradeIdeaQueryPage(Generic[_QueryItem]):
    """Stable read-only query page for audit and closeout reporting."""

    items: tuple[_QueryItem, ...]
    total_count: int
    limit: int | None
    offset: int


class TradeIdeaListSortKey(str, Enum):
    DECISION_ID = "decision_id"
    STATE = "state"
    INSTRUMENT = "instrument"
    DIRECTION = "direction"
    CONFIDENCE = "confidence"
    MAX_LOSS_PCT = "max_loss_pct"
    EXPIRES_AT = "expires_at"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"


@dataclass(frozen=True, slots=True)
class TradeIdeaListQuery:
    """Shared read query for trade-idea review adapters."""

    state: TradeIdeaState | None = None
    instrument: str | None = None
    decision_id: str | None = None
    direction: TradeDirection | None = None
    min_confidence: ConfidenceLabel | None = None
    max_confidence: ConfidenceLabel | None = None
    updated_since: datetime | None = None
    updated_until: datetime | None = None
    sort_by: TradeIdeaListSortKey | None = None
    descending: bool = False
    limit: int | None = None
    offset: int = 0


@dataclass(frozen=True, slots=True)
class TradeIdeaListResult:
    """Filtered trade-idea views plus pagination metadata."""

    views: tuple[TradeIdeaView, ...]
    total_count: int
    offset: int
    limit: int | None
    has_more: bool

    @property
    def returned_count(self) -> int:
        return len(self.views)
