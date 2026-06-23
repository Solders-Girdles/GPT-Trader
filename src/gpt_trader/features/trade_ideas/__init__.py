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
from gpt_trader.features.trade_ideas.baseline import (
    BaselineProposer,
    BaselineProposerConfig,
)
from gpt_trader.features.trade_ideas.budget import (
    DEFAULT_RISK_BUDGET,
    BudgetIntegrityError,
    BudgetLogEntry,
    RiskBudget,
    RiskBudgetLog,
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
from gpt_trader.features.trade_ideas.policy import ApprovalPolicy, PolicyViolationError
from gpt_trader.features.trade_ideas.proposer import Proposer
from gpt_trader.features.trade_ideas.replay import (
    ReplayOutcome,
    ReplayReport,
    ReplayResult,
    ReplayRunnerConfig,
    ReplayScoringError,
    ScoringLevels,
    TradeIdeaReplayRunner,
    extract_numeric_scoring_levels,
    score_trade_idea,
)
from gpt_trader.features.trade_ideas.service import (
    TradeIdeaService,
    TradeIdeaView,
    UnknownTradeIdeaError,
)
from gpt_trader.features.trade_ideas.snapshot import (
    MarketSnapshot,
    SnapshotIntegrityError,
    SymbolSeries,
)
from gpt_trader.features.trade_ideas.store import TradeIdeaStore
from gpt_trader.features.trade_ideas.workflow import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    InvalidTransitionError,
    TradeIdeaState,
    validate_transition,
)

__all__ = [
    "ALLOWED_TRANSITIONS",
    "DEFAULT_RISK_BUDGET",
    "TERMINAL_STATES",
    "ActorType",
    "ApprovalPolicy",
    "AuditAction",
    "AuditEvent",
    "AuditIntegrityError",
    "AutonomyMode",
    "BaselineProposer",
    "BaselineProposerConfig",
    "BrokerTicket",
    "BudgetIntegrityError",
    "BudgetLogEntry",
    "Confidence",
    "ConfidenceLabel",
    "EntryZone",
    "InvalidTransitionError",
    "MarketSnapshot",
    "MaxLoss",
    "PolicyViolationError",
    "ProductType",
    "Proposer",
    "ReplayOutcome",
    "ReplayReport",
    "ReplayResult",
    "ReplayRunnerConfig",
    "ReplayScoringError",
    "RiskBudget",
    "RiskBudgetLog",
    "ScoringLevels",
    "SizingRecommendation",
    "SnapshotIntegrityError",
    "SymbolSeries",
    "TicketStatus",
    "TicketVenue",
    "TimeHorizon",
    "TradeDirection",
    "TradeIdea",
    "TradeIdeaAuditLog",
    "TradeIdeaReplayRunner",
    "TradeIdeaService",
    "TradeIdeaState",
    "TradeIdeaStore",
    "TradeIdeaView",
    "UnknownTradeIdeaError",
    "evaluate_eligibility",
    "extract_numeric_scoring_levels",
    "is_eligible",
    "new_event_id",
    "score_trade_idea",
    "validate_transition",
]
