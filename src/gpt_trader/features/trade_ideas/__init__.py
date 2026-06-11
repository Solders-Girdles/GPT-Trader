"""Broker-neutral trade-idea records, workflow, and audit trail.

First building block of the accepted staged-autonomy direction
(docs/PRE_MIGRATION_DECISION_FRAMEWORK.md): AI proposes complete trade-idea
records, humans approve them, and every state change lands in an append-only
audit log. No module in this slice may submit orders.
"""

from gpt_trader.features.trade_ideas.audit import (
    ActorType,
    AuditAction,
    AuditEvent,
    AuditIntegrityError,
    TradeIdeaAuditLog,
    new_event_id,
)
from gpt_trader.features.trade_ideas.eligibility import evaluate_eligibility, is_eligible
from gpt_trader.features.trade_ideas.models import (
    AutonomyMode,
    BrokerTicket,
    Confidence,
    ConfidenceLabel,
    EntryZone,
    MaxLoss,
    ProductType,
    SizingRecommendation,
    TicketStatus,
    TicketVenue,
    TimeHorizon,
    TradeDirection,
    TradeIdea,
)
from gpt_trader.features.trade_ideas.workflow import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    InvalidTransitionError,
    TradeIdeaState,
    validate_transition,
)

__all__ = [
    "ALLOWED_TRANSITIONS",
    "TERMINAL_STATES",
    "ActorType",
    "AuditAction",
    "AuditEvent",
    "AuditIntegrityError",
    "AutonomyMode",
    "BrokerTicket",
    "Confidence",
    "ConfidenceLabel",
    "EntryZone",
    "InvalidTransitionError",
    "MaxLoss",
    "ProductType",
    "SizingRecommendation",
    "TicketStatus",
    "TicketVenue",
    "TimeHorizon",
    "TradeDirection",
    "TradeIdea",
    "TradeIdeaAuditLog",
    "TradeIdeaState",
    "evaluate_eligibility",
    "is_eligible",
    "new_event_id",
    "validate_transition",
]
