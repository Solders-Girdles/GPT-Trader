"""Approval policy: encodes the autonomy mode as enforceable checks.

This module is the seam where autonomy is handed over. Moving up the ladder
(human approval -> bounded autonomy) means changing policy data and rules
here — never the service plumbing or the audit trail. In the current accepted
mode (``human_approved_execution``), only a human actor can move an idea to
``approved``, and approvals must clear the eligibility gate and the current
risk budget.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from gpt_trader.errors import ValidationError
from gpt_trader.features.trade_ideas.audit import ActorType
from gpt_trader.features.trade_ideas.budget import RiskBudget
from gpt_trader.features.trade_ideas.eligibility import evaluate_eligibility
from gpt_trader.features.trade_ideas.models import (
    AutonomyMode,
    ProductType,
    TradeDirection,
    TradeIdea,
)


class PolicyViolationError(ValidationError):
    """Raised when an action violates the active approval policy."""

    def __init__(self, message: str, violations: list[str] | None = None) -> None:
        super().__init__(message)
        self.violations = violations or []


BOUNDED_AUTONOMY_NON_HUMAN_APPROVAL_VIOLATION = (
    "Autonomy mode 'bounded_autonomy' does not permit non-human approvals until a "
    "strategy envelope, kill-switch evidence, and audit evidence are modeled "
    "or a later decision packet scopes a narrower exception"
)
BOUNDED_AUTONOMY_NON_HUMAN_BUDGET_CHANGE_VIOLATION = (
    "Autonomy mode 'bounded_autonomy' does not permit non-human budget changes "
    "until a budget meta-envelope is modeled or a later decision packet scopes "
    "a narrower exception"
)


class ApprovalPolicy:
    """Checks workflow actions against the active autonomy mode and budget."""

    def __init__(self, autonomy_mode: AutonomyMode = AutonomyMode.HUMAN_APPROVED_EXECUTION) -> None:
        self._autonomy_mode = autonomy_mode

    @property
    def autonomy_mode(self) -> AutonomyMode:
        return self._autonomy_mode

    def approval_violations(
        self,
        idea: TradeIdea,
        actor_type: ActorType,
        budget: RiskBudget,
        open_approved_count: int,
        now: datetime,
        review_started_at: datetime | None = None,
    ) -> list[str]:
        """Return every reason this approval must be refused; empty means allowed."""
        violations: list[str] = []

        if self._autonomy_mode is AutonomyMode.RESEARCH_ONLY:
            violations.append("Autonomy mode 'research_only' does not permit approvals")
        elif self._autonomy_mode is AutonomyMode.HUMAN_APPROVED_EXECUTION:
            if actor_type is not ActorType.HUMAN:
                violations.append(
                    "Autonomy mode 'human_approved_execution' requires a human approver; "
                    f"got actor_type '{actor_type.value}'"
                )
        elif (
            self._autonomy_mode is AutonomyMode.BOUNDED_AUTONOMY
            and actor_type is not ActorType.HUMAN
        ):
            violations.append(
                BOUNDED_AUTONOMY_NON_HUMAN_APPROVAL_VIOLATION
                + f"; got actor_type '{actor_type.value}'"
            )

        violations.extend(evaluate_eligibility(idea))

        percent = idea.max_loss.percent_of_account
        if percent is None:
            violations.append(
                "max_loss.percent_of_account is required to verify the idea against the budget"
            )
        elif percent > budget.max_loss_per_idea_pct:
            violations.append(
                f"max_loss {percent}% exceeds budget cap of "
                f"{budget.max_loss_per_idea_pct}% per idea"
            )

        if idea.product_type is ProductType.FUTURES and not budget.allow_futures_leverage:
            violations.append(
                "product_type futures requires risk budget allow_futures_leverage=true"
            )

        if idea.direction is TradeDirection.SHORT and not budget.allow_naked_shorts:
            violations.append("direction short requires risk budget allow_naked_shorts=true")

        if open_approved_count >= budget.max_concurrent_approved_tickets:
            violations.append(
                f"{open_approved_count} tickets already approved; budget allows "
                f"{budget.max_concurrent_approved_tickets} concurrent approved tickets"
            )

        expires_at = idea.time_horizon.expires_at
        if expires_at is not None and expires_at <= now:
            violations.append(f"Idea expired at {expires_at.isoformat()}; approve nothing stale")

        review_latency_violation = self.review_latency_violation(
            review_started_at=review_started_at,
            budget=budget,
            now=now,
        )
        if review_latency_violation is not None:
            violations.append(review_latency_violation)

        return violations

    def review_latency_violation(
        self,
        *,
        review_started_at: datetime | None,
        budget: RiskBudget,
        now: datetime,
    ) -> str | None:
        """Return a violation when the active review window has elapsed."""
        if review_started_at is None:
            return None
        review_deadline = review_started_at + timedelta(hours=budget.max_review_latency_hours)
        if review_deadline > now:
            return None
        return (
            f"Idea review deadline expired at {review_deadline.isoformat()} "
            f"after max_review_latency_hours={budget.max_review_latency_hours}"
        )

    def budget_change_violations(self, actor_type: ActorType) -> list[str]:
        """Budget renegotiation rules for the current autonomy mode.

        Agents may *propose* budget changes at any stage; until a budget
        meta-envelope is modeled, only a human can enact one.
        """
        if self._autonomy_mode is AutonomyMode.BOUNDED_AUTONOMY:
            if actor_type is ActorType.HUMAN:
                return []
            return [
                BOUNDED_AUTONOMY_NON_HUMAN_BUDGET_CHANGE_VIOLATION
                + f"; got actor_type '{actor_type.value}'"
            ]
        if actor_type is not ActorType.HUMAN:
            return [
                f"Autonomy mode '{self._autonomy_mode.value}' requires a human to enact "
                f"budget changes; got actor_type '{actor_type.value}'"
            ]
        return []
