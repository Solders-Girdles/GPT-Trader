"""Trade-idea lifecycle service: the one audited code path for every actor.

Humans, development agents, and (eventually) operating agents all act through
this service. Every action is identity-stamped, checked against the approval
policy, and appended to the audit log; interfaces such as CLI, TUI, or MCP
servers must stay thin adapters over these methods.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
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
from gpt_trader.features.trade_ideas.workflow import TradeIdeaState


class UnknownTradeIdeaError(ValidationError):
    """Raised when a decision_id has no stored record."""


@dataclass(frozen=True, slots=True)
class TradeIdeaView:
    """A record plus its derived workflow state and full history."""

    idea: TradeIdea
    state: TradeIdeaState
    events: tuple[AuditEvent, ...]


def _utc_now() -> datetime:
    return datetime.now(UTC)


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
        self._require_idea(idea.decision_id)
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
        idea = self._store.load_latest(decision_id)
        if idea is None:
            raise UnknownTradeIdeaError(
                f"No trade idea stored for decision_id '{decision_id}'",
                field="decision_id",
                value=decision_id,
            )
        return idea

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
