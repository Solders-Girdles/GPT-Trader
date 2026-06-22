"""Trade-idea lifecycle service: the one audited code path for every actor.

Humans, development agents, and (eventually) operating agents all act through
this service. Every action is identity-stamped, checked against the approval
policy, and appended to the audit log; interfaces such as CLI, TUI, or MCP
servers must stay thin adapters over these methods.
"""

from __future__ import annotations

import getpass
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import InvalidOperation
from pathlib import Path

from gpt_trader.errors import ValidationError
from gpt_trader.features.trade_ideas.audit import (
    ActorType,
    AuditAction,
    AuditEvent,
    TradeIdeaAuditLog,
    new_event_id,
)
from gpt_trader.features.trade_ideas.budget import (
    DEFAULT_RISK_BUDGET,
    BudgetLogEntry,
    RiskBudget,
    RiskBudgetLog,
)
from gpt_trader.features.trade_ideas.models import TradeIdea
from gpt_trader.features.trade_ideas.policy import ApprovalPolicy, PolicyViolationError
from gpt_trader.features.trade_ideas.store import TradeIdeaStore
from gpt_trader.features.trade_ideas.workflow import TradeIdeaState, validate_transition

DEFAULT_IDEAS_ROOT = Path("var/data/trade_ideas")
IDEAS_ROOT_ENV_VAR = "GPT_TRADER_IDEAS_ROOT"
ACTOR_ENV_VAR = "GPT_TRADER_ACTOR"
EXPIRABLE_STATES = frozenset(
    {
        TradeIdeaState.PROPOSED,
        TradeIdeaState.NEEDS_CHANGES,
        TradeIdeaState.APPROVED,
    }
)


class UnknownTradeIdeaError(ValidationError):
    """Raised when a decision_id has no stored record."""


class DuplicateTradeIdeaError(ValidationError):
    """Raised when a new proposal reuses an existing decision_id."""


@dataclass(frozen=True, slots=True)
class TradeIdeaView:
    """A record plus its derived workflow state and full history."""

    idea: TradeIdea
    state: TradeIdeaState
    events: tuple[AuditEvent, ...]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def resolve_ideas_root(root: Path | None = None) -> Path:
    """Resolve the trade-idea storage root from arg, environment, or default."""
    if root is not None:
        return root
    configured_root = os.environ.get(IDEAS_ROOT_ENV_VAR, "").strip()
    if configured_root:
        return Path(configured_root)
    return DEFAULT_IDEAS_ROOT


def resolve_trade_idea_actor_id(actor_id: str | None = None) -> str:
    """Resolve actor identity from arg, environment, or the current OS user."""
    if actor_id and actor_id.strip():
        return actor_id
    configured_actor = os.environ.get(ACTOR_ENV_VAR, "").strip()
    if configured_actor:
        return configured_actor
    return getpass.getuser()


class TradeIdeaService:
    def __init__(
        self,
        root: Path,
        *,
        policy: ApprovalPolicy | None = None,
        now_factory: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._store = TradeIdeaStore(root / "records")
        self._audit = TradeIdeaAuditLog(root / "audit.jsonl")
        self._budget_log = RiskBudgetLog(root / "risk_budget.jsonl")
        self._policy = policy or ApprovalPolicy()
        self._now = now_factory

    @property
    def audit_log(self) -> TradeIdeaAuditLog:
        return self._audit

    # -- budget ----------------------------------------------------------

    def current_budget(self) -> RiskBudget:
        """Return the active budget, seeding the accepted defaults on first use."""
        budget = self._budget_log.current()
        if budget is not None:
            return budget
        self._budget_log.append(
            BudgetLogEntry(
                timestamp=self._now(),
                actor_type=ActorType.SYSTEM,
                actor_id="seed-defaults",
                budget=DEFAULT_RISK_BUDGET,
            )
        )
        return DEFAULT_RISK_BUDGET

    def approval_violations(
        self,
        idea: TradeIdea,
        *,
        actor_type: ActorType = ActorType.HUMAN,
    ) -> list[str]:
        """Return every current-policy reason an idea could not be approved."""
        return self._policy.approval_violations(
            idea,
            actor_type=actor_type,
            budget=self.current_budget(),
            open_approved_count=self.open_approved_count(),
            now=self._now(),
        )

    def validate_new_proposal(self, idea: TradeIdea) -> None:
        """Validate proposal lifecycle preconditions without writing state."""
        self._require_new_decision_id(idea.decision_id)

    def validate_resubmission(self, idea: TradeIdea) -> None:
        """Validate resubmission lifecycle preconditions without writing state."""
        self._require_idea(idea.decision_id)
        validate_transition(self._audit.current_state(idea.decision_id), TradeIdeaState.PROPOSED)

    def update_budget(self, budget: RiskBudget, actor_type: ActorType, actor_id: str) -> None:
        """Enact a new budget version, subject to the autonomy-mode policy."""
        violations = self._policy.budget_change_violations(actor_type)
        if violations:
            raise PolicyViolationError(
                "Budget change refused: " + "; ".join(violations), violations
            )
        self.current_budget()  # ensure the seed entry exists so versions stay contiguous
        self._budget_log.append(
            BudgetLogEntry(
                timestamp=self._now(),
                actor_type=actor_type,
                actor_id=actor_id,
                budget=budget,
            )
        )

    # -- lifecycle actions -------------------------------------------------

    def propose(
        self,
        idea: TradeIdea,
        actor_id: str,
        actor_type: ActorType = ActorType.AI,
        reason: str = "New trade idea proposed",
        evidence: tuple[str, ...] = (),
    ) -> TradeIdeaView:
        self.validate_new_proposal(idea)
        self._store.save(idea)
        self._append(
            idea,
            action=AuditAction.PROPOSED,
            after_state=TradeIdeaState.PROPOSED,
            actor_type=actor_type,
            actor_id=actor_id,
            reason=reason,
            evidence=evidence,
        )
        return self.get(idea.decision_id)

    def request_changes(self, decision_id: str, actor_id: str, reason: str) -> TradeIdeaView:
        idea = self._require_idea(decision_id)
        self._append(
            idea,
            action=AuditAction.CHANGED,
            after_state=TradeIdeaState.NEEDS_CHANGES,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
            reason=reason,
        )
        return self.get(decision_id)

    def resubmit(
        self,
        idea: TradeIdea,
        actor_id: str,
        actor_type: ActorType = ActorType.AI,
        reason: str = "Revised after requested changes",
    ) -> TradeIdeaView:
        self.validate_resubmission(idea)
        self._store.save(idea)
        self._append(
            idea,
            action=AuditAction.PROPOSED,
            after_state=TradeIdeaState.PROPOSED,
            actor_type=actor_type,
            actor_id=actor_id,
            reason=reason,
        )
        return self.get(idea.decision_id)

    def approve(self, decision_id: str, actor_id: str, reason: str) -> TradeIdeaView:
        idea = self._require_idea(decision_id)
        violations = self._policy.approval_violations(
            idea,
            actor_type=ActorType.HUMAN,
            budget=self.current_budget(),
            open_approved_count=self.open_approved_count(),
            now=self._now(),
        )
        if violations:
            raise PolicyViolationError(
                f"Approval of '{decision_id}' refused: " + "; ".join(violations), violations
            )
        self._append(
            idea,
            action=AuditAction.APPROVED,
            after_state=TradeIdeaState.APPROVED,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
            reason=reason,
        )
        return self.get(decision_id)

    def reject(
        self,
        decision_id: str,
        actor_id: str,
        reason: str,
        actor_type: ActorType = ActorType.HUMAN,
    ) -> TradeIdeaView:
        idea = self._require_idea(decision_id)
        self._append(
            idea,
            action=AuditAction.REJECTED,
            after_state=TradeIdeaState.REJECTED,
            actor_type=actor_type,
            actor_id=actor_id,
            reason=reason,
        )
        return self.get(decision_id)

    def cancel(
        self,
        decision_id: str,
        actor_id: str,
        reason: str,
        actor_type: ActorType = ActorType.HUMAN,
    ) -> TradeIdeaView:
        idea = self._require_idea(decision_id)
        self._append(
            idea,
            action=AuditAction.CANCELLED,
            after_state=TradeIdeaState.CANCELLED,
            actor_type=actor_type,
            actor_id=actor_id,
            reason=reason,
        )
        return self.get(decision_id)

    def expire(
        self,
        decision_id: str,
        actor_id: str = "expiry-sweep",
        reason: str = "Idea passed its review or execution deadline",
        actor_type: ActorType = ActorType.SYSTEM,
    ) -> TradeIdeaView:
        idea = self._require_idea(decision_id)
        self._append(
            idea,
            action=AuditAction.EXPIRED,
            after_state=TradeIdeaState.EXPIRED,
            actor_type=actor_type,
            actor_id=actor_id,
            reason=reason,
        )
        return self.get(decision_id)

    def expire_due_ideas(
        self,
        *,
        actor_id: str = "expiry-sweep",
        reason: str = "Idea passed its review or execution deadline",
        actor_type: ActorType = ActorType.SYSTEM,
    ) -> list[TradeIdeaView]:
        """Expire all stale ideas that can legally transition to expired."""
        now = self._now()
        expired: list[TradeIdeaView] = []
        for view in self.list_views():
            if view.state not in EXPIRABLE_STATES:
                continue
            expires_at = view.idea.time_horizon.expires_at
            if expires_at is not None and expires_at <= now:
                expired.append(
                    self.expire(
                        view.idea.decision_id,
                        actor_id=actor_id,
                        reason=reason,
                        actor_type=actor_type,
                    )
                )
        return expired

    def record_submission(
        self,
        decision_id: str,
        actor_id: str,
        venue: str,
        external_order_id: str = "",
        reason: str = "Approved ticket submitted",
        actor_type: ActorType = ActorType.SYSTEM,
    ) -> TradeIdeaView:
        idea = self._require_idea(decision_id)
        self._append(
            idea,
            action=AuditAction.SUBMITTED,
            after_state=TradeIdeaState.SUBMITTED,
            actor_type=actor_type,
            actor_id=actor_id,
            reason=reason,
            venue=venue,
            external_order_id=external_order_id,
        )
        return self.get(decision_id)

    def record_fill(
        self,
        decision_id: str,
        actor_id: str,
        venue: str,
        external_order_id: str = "",
        reason: str = "Venue confirmed fill",
        actor_type: ActorType = ActorType.VENUE,
    ) -> TradeIdeaView:
        idea = self._require_idea(decision_id)
        self._append(
            idea,
            action=AuditAction.FILLED,
            after_state=TradeIdeaState.FILLED,
            actor_type=actor_type,
            actor_id=actor_id,
            reason=reason,
            venue=venue,
            external_order_id=external_order_id,
        )
        return self.get(decision_id)

    # -- queries -----------------------------------------------------------

    def get(self, decision_id: str) -> TradeIdeaView:
        idea = self._require_idea(decision_id)
        events = tuple(self._audit.read_events(decision_id))
        state = events[-1].after_state if events else TradeIdeaState.PROPOSED
        return TradeIdeaView(idea=idea, state=state, events=events)

    def list_views(self, state: TradeIdeaState | None = None) -> list[TradeIdeaView]:
        views = [self.get(decision_id) for decision_id in self._store.list_decision_ids()]
        if state is None:
            return views
        return [view for view in views if view.state is state]

    def open_approved_count(self) -> int:
        return len(self.list_views(TradeIdeaState.APPROVED))

    # -- internals -----------------------------------------------------------

    def _require_idea(self, decision_id: str) -> TradeIdea:
        try:
            idea = self._store.load_latest(decision_id)
        except (InvalidOperation, TypeError, ValueError) as error:
            raise ValidationError(
                f"Stored trade idea '{decision_id}' is invalid: {error}",
                field="decision_id",
                value=decision_id,
            ) from error
        if idea is None:
            raise UnknownTradeIdeaError(
                f"No trade idea stored for decision_id '{decision_id}'",
                field="decision_id",
                value=decision_id,
            )
        return idea

    def _require_new_decision_id(self, decision_id: str) -> None:
        if self._store.exists(decision_id) or self._audit.current_state(decision_id) is not None:
            raise DuplicateTradeIdeaError(
                f"Trade idea decision_id '{decision_id}' already exists; use resubmit "
                "after requested changes",
                field="decision_id",
                value=decision_id,
            )

    def _append(
        self,
        idea: TradeIdea,
        *,
        action: AuditAction,
        after_state: TradeIdeaState,
        actor_type: ActorType,
        actor_id: str,
        reason: str,
        evidence: tuple[str, ...] = (),
        venue: str = "",
        external_order_id: str = "",
    ) -> None:
        self._audit.append(
            AuditEvent(
                event_id=new_event_id(),
                timestamp=self._now(),
                decision_id=idea.decision_id,
                actor_type=actor_type,
                actor_id=actor_id,
                action=action,
                before_state=self._audit.current_state(idea.decision_id),
                after_state=after_state,
                reason=reason,
                record_hash=idea.record_hash(),
                evidence=evidence,
                venue=venue,
                external_order_id=external_order_id,
            )
        )


def create_trade_idea_service(
    root: Path | None = None,
    *,
    policy: ApprovalPolicy | None = None,
    now_factory: Callable[[], datetime] = _utc_now,
) -> TradeIdeaService:
    """Resolve root (arg > GPT_TRADER_IDEAS_ROOT > default) and build the service."""
    return TradeIdeaService(resolve_ideas_root(root), policy=policy, now_factory=now_factory)
