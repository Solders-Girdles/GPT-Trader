from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea

from gpt_trader.features.trade_ideas import (
    DEFAULT_RISK_BUDGET,
    ActorType,
    ApprovalPolicy,
    AutonomyMode,
    MaxLoss,
    TradeIdea,
)

NOW = datetime(2026, 6, 12, 10, 0, tzinfo=UTC)


def violations(
    idea: TradeIdea,
    *,
    policy: ApprovalPolicy | None = None,
    actor_type: ActorType = ActorType.HUMAN,
    open_approved_count: int = 0,
) -> list[str]:
    active = policy or ApprovalPolicy()
    return active.approval_violations(
        idea,
        actor_type=actor_type,
        budget=DEFAULT_RISK_BUDGET,
        open_approved_count=open_approved_count,
        now=NOW,
    )


def test_eligible_idea_with_human_approver_passes(trade_idea: TradeIdea) -> None:
    assert violations(trade_idea) == []


def test_ai_actor_cannot_approve_in_human_approved_mode(trade_idea: TradeIdea) -> None:
    found = violations(trade_idea, actor_type=ActorType.AI)

    assert any("requires a human approver" in violation for violation in found)


def test_research_only_mode_blocks_all_approvals(trade_idea: TradeIdea) -> None:
    policy = ApprovalPolicy(AutonomyMode.RESEARCH_ONLY)

    found = violations(trade_idea, policy=policy)

    assert any("research_only" in violation for violation in found)


def test_ineligible_idea_cannot_be_approved() -> None:
    idea = build_trade_idea(invalidation="")

    assert any("invalidation" in violation for violation in violations(idea))


def test_max_loss_above_budget_cap_is_refused() -> None:
    idea = build_trade_idea(
        max_loss=MaxLoss(amount=Decimal("900"), percent_of_account=Decimal("9"))
    )

    found = violations(idea)

    assert any("exceeds budget cap" in violation for violation in found)


def test_missing_percent_cannot_be_verified() -> None:
    idea = build_trade_idea(max_loss=MaxLoss(amount=Decimal("250")))

    found = violations(idea)

    assert any("percent_of_account is required" in violation for violation in found)


def test_concurrent_approved_cap_is_enforced(trade_idea: TradeIdea) -> None:
    found = violations(
        trade_idea,
        open_approved_count=DEFAULT_RISK_BUDGET.max_concurrent_approved_tickets,
    )

    assert any("concurrent approved tickets" in violation for violation in found)


def test_stale_idea_cannot_be_approved(trade_idea: TradeIdea) -> None:
    policy = ApprovalPolicy()

    found = policy.approval_violations(
        trade_idea,
        actor_type=ActorType.HUMAN,
        budget=DEFAULT_RISK_BUDGET,
        open_approved_count=0,
        now=datetime(2026, 7, 1, tzinfo=UTC),
    )

    assert any("expired" in violation for violation in found)


def test_budget_changes_require_human_below_bounded_autonomy() -> None:
    policy = ApprovalPolicy()

    assert policy.budget_change_violations(ActorType.HUMAN) == []
    assert policy.budget_change_violations(ActorType.AI) != []


def test_bounded_autonomy_lets_agents_change_budgets() -> None:
    policy = ApprovalPolicy(AutonomyMode.BOUNDED_AUTONOMY)

    assert policy.budget_change_violations(ActorType.AI) == []
