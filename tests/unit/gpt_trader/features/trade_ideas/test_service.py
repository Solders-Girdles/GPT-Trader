from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea

from gpt_trader.features.trade_ideas import (
    DEFAULT_RISK_BUDGET,
    ActorType,
    AuditAction,
    MaxLoss,
    PolicyViolationError,
    RiskBudget,
    TimeHorizon,
    TradeIdeaService,
    TradeIdeaState,
    UnknownTradeIdeaError,
)


@pytest.fixture
def service(tmp_path: Path) -> TradeIdeaService:
    return TradeIdeaService(
        tmp_path / "trade_ideas",
        now_factory=lambda: datetime(2026, 6, 12, 10, 0, tzinfo=UTC),
    )


def test_propose_creates_proposed_view(service: TradeIdeaService) -> None:
    view = service.propose(build_trade_idea(), actor_id="idea-generator-v1")

    assert view.state is TradeIdeaState.PROPOSED
    assert view.events[0].actor_type is ActorType.AI
    assert view.events[0].record_hash == view.idea.record_hash()


def test_full_lifecycle_to_fill(service: TradeIdeaService) -> None:
    idea = build_trade_idea()
    service.propose(idea, actor_id="idea-generator-v1")
    service.approve(idea.decision_id, actor_id="rj", reason="Thesis and risk verified")
    service.record_submission(idea.decision_id, actor_id="executor", venue="coinbase")
    view = service.record_fill(
        idea.decision_id, actor_id="coinbase", venue="coinbase", external_order_id="abc-123"
    )

    assert view.state is TradeIdeaState.FILLED
    assert [event.action for event in view.events] == [
        AuditAction.PROPOSED,
        AuditAction.APPROVED,
        AuditAction.SUBMITTED,
        AuditAction.FILLED,
    ]


def test_changes_loop_keeps_every_record_version(service: TradeIdeaService) -> None:
    idea = build_trade_idea()
    service.propose(idea, actor_id="idea-generator-v1")
    service.request_changes(idea.decision_id, actor_id="rj", reason="Tighten the invalidation")
    revised = build_trade_idea(invalidation="Daily close below 59000")
    view = service.resubmit(revised, actor_id="idea-generator-v1")

    assert view.state is TradeIdeaState.PROPOSED
    assert view.idea.invalidation == "Daily close below 59000"
    hashes = {event.record_hash for event in view.events}
    assert idea.record_hash() in hashes
    assert revised.record_hash() in hashes


def test_approval_refused_for_budget_violation(service: TradeIdeaService) -> None:
    idea = build_trade_idea(
        max_loss=MaxLoss(amount=Decimal("900"), percent_of_account=Decimal("9"))
    )
    service.propose(idea, actor_id="idea-generator-v1")

    with pytest.raises(PolicyViolationError) as exc_info:
        service.approve(idea.decision_id, actor_id="rj", reason="Looks fine")

    assert any("exceeds budget cap" in violation for violation in exc_info.value.violations)
    assert service.get(idea.decision_id).state is TradeIdeaState.PROPOSED


def test_unknown_decision_id_is_an_error(service: TradeIdeaService) -> None:
    with pytest.raises(UnknownTradeIdeaError):
        service.approve("trade-missing", actor_id="rj", reason="?")


def test_list_views_filters_by_state(service: TradeIdeaService) -> None:
    first = build_trade_idea(decision_id="trade-20260612-001")
    second = build_trade_idea(decision_id="trade-20260612-002")
    service.propose(first, actor_id="idea-generator-v1")
    service.propose(second, actor_id="idea-generator-v1")
    service.approve(first.decision_id, actor_id="rj", reason="Verified")

    approved = service.list_views(TradeIdeaState.APPROVED)

    assert [view.idea.decision_id for view in approved] == ["trade-20260612-001"]
    assert service.open_approved_count() == 1


def test_budget_seeds_defaults_on_first_use(service: TradeIdeaService) -> None:
    assert service.current_budget() == DEFAULT_RISK_BUDGET


def test_human_can_renegotiate_budget(service: TradeIdeaService) -> None:
    widened = RiskBudget.from_dict(
        {**DEFAULT_RISK_BUDGET.to_dict(), "version": 2, "max_loss_per_idea_pct": "8"}
    )

    service.update_budget(widened, actor_type=ActorType.HUMAN, actor_id="rj")

    assert service.current_budget().max_loss_per_idea_pct == Decimal("8")


def test_agent_budget_change_refused_in_current_mode(service: TradeIdeaService) -> None:
    widened = RiskBudget.from_dict(
        {**DEFAULT_RISK_BUDGET.to_dict(), "version": 2, "max_loss_per_idea_pct": "8"}
    )

    with pytest.raises(PolicyViolationError):
        service.update_budget(widened, actor_type=ActorType.AI, actor_id="idea-generator-v1")

    assert service.current_budget() == DEFAULT_RISK_BUDGET


def test_expire_is_a_system_action(service: TradeIdeaService) -> None:
    idea = build_trade_idea()
    service.propose(idea, actor_id="idea-generator-v1")

    view = service.expire(idea.decision_id)

    assert view.state is TradeIdeaState.EXPIRED
    assert view.events[-1].actor_type is ActorType.SYSTEM


def test_expire_due_ideas_skips_submitted_and_continues(tmp_path: Path) -> None:
    current_time = datetime(2026, 6, 10, 10, 0, tzinfo=UTC)
    service = TradeIdeaService(
        tmp_path / "trade_ideas",
        now_factory=lambda: current_time,
    )
    expires_soon = TimeHorizon(
        expected_hold="1 day",
        expires_at=datetime(2026, 6, 11, 10, 0, tzinfo=UTC),
    )
    submitted = build_trade_idea(
        decision_id="trade-20260612-submitted",
        time_horizon=expires_soon,
    )
    proposed = build_trade_idea(
        decision_id="trade-20260612-proposed",
        time_horizon=expires_soon,
    )
    service.propose(submitted, actor_id="idea-generator-v1")
    service.approve(submitted.decision_id, actor_id="rj", reason="Fresh at approval time")
    service.record_submission(submitted.decision_id, actor_id="manual", venue="coinbase")
    service.propose(proposed, actor_id="idea-generator-v1")

    current_time = datetime(2026, 6, 12, 10, 0, tzinfo=UTC)
    expired = service.expire_due_ideas()

    assert [view.idea.decision_id for view in expired] == [proposed.decision_id]
    assert service.get(submitted.decision_id).state is TradeIdeaState.SUBMITTED
    assert service.get(proposed.decision_id).state is TradeIdeaState.EXPIRED
