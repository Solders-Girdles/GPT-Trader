from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea

from gpt_trader.features.trade_ideas import (
    ActorType,
    AuditAction,
    PaperFillEvent,
    PaperFillProfileError,
    PaperFillReconciler,
    TradeIdeaService,
    TradeIdeaState,
    validate_paper_reconciliation_profile,
)


def _service(root: Path) -> TradeIdeaService:
    return TradeIdeaService(
        root,
        now_factory=lambda: datetime(2026, 6, 12, 10, 0, tzinfo=UTC),
    )


def _approved_idea(service: TradeIdeaService) -> str:
    idea = build_trade_idea()
    service.propose(idea, actor_id="idea-generator-v1")
    service.approve(idea.decision_id, actor_id="rj", reason="Risk verified")
    return idea.decision_id


def _fill_event(
    *,
    decision_id: str | None = None,
    client_order_id: str = "",
    order_id: str = "MOCK_000001",
    symbol: str = "BTC-USD",
    side: str = "buy",
) -> PaperFillEvent:
    return PaperFillEvent(
        order_id=order_id,
        client_order_id=client_order_id,
        symbol=symbol,
        side=side,
        quantity=Decimal("0.1"),
        price=Decimal("60750"),
        status="filled",
        decision_id=decision_id,
    )


def test_reconciler_records_matched_approved_fill_through_service(tmp_path: Path) -> None:
    service = _service(tmp_path / "ideas")
    decision_id = _approved_idea(service)

    report = PaperFillReconciler(service, actor_id="paper-reconciler").reconcile_fills(
        [_fill_event(client_order_id=decision_id)],
        apply=True,
    )

    assert report.matched_count == 1
    assert report.recorded_count == 1
    view = service.get(decision_id)
    assert view.state is TradeIdeaState.FILLED
    assert [event.action for event in view.events[-2:]] == [
        AuditAction.SUBMITTED,
        AuditAction.FILLED,
    ]
    assert [event.external_order_id for event in view.events[-2:]] == [
        "MOCK_000001",
        "MOCK_000001",
    ]
    assert view.events[-2].actor_type is ActorType.SYSTEM
    assert view.events[-1].actor_type is ActorType.VENUE


def test_reconciler_reports_unmatched_fill_without_mutation(tmp_path: Path) -> None:
    service = _service(tmp_path / "ideas")
    decision_id = _approved_idea(service)
    original_events = tuple(service.get(decision_id).events)

    report = PaperFillReconciler(service).reconcile_fills(
        [_fill_event(order_id="MOCK_000002", client_order_id="unknown", symbol="ETH-USD")],
        apply=True,
    )

    assert report.matched_count == 0
    assert report.unmatched_count == 1
    assert report.unmatched[0].reason == "no approved idea matched fill"
    view = service.get(decision_id)
    assert view.state is TradeIdeaState.APPROVED
    assert tuple(view.events) == original_events


def test_reconciler_dry_run_reports_match_without_mutation(tmp_path: Path) -> None:
    service = _service(tmp_path / "ideas")
    decision_id = _approved_idea(service)

    report = PaperFillReconciler(service).reconcile_fills(
        [_fill_event(decision_id=decision_id, order_id="MOCK_000003")],
        apply=False,
    )

    assert report.mode == "dry_run"
    assert report.matched_count == 1
    assert report.recorded_count == 0
    assert service.get(decision_id).state is TradeIdeaState.APPROVED


def test_reconciler_skips_duplicate_fill_in_same_apply_pass(tmp_path: Path) -> None:
    service = _service(tmp_path / "ideas")
    decision_id = _approved_idea(service)
    event = _fill_event(client_order_id=decision_id, order_id="MOCK_000004")

    report = PaperFillReconciler(service).reconcile_fills([event, event], apply=True)

    assert report.matched_count == 1
    assert report.skipped_count == 1
    assert report.skipped[0].reason == "fill already recorded on trade-idea audit trail"
    view = service.get(decision_id)
    assert view.state is TradeIdeaState.FILLED
    assert [audit_event.action for audit_event in view.events].count(AuditAction.FILLED) == 1


def test_live_profiles_are_rejected_for_paper_reconciliation() -> None:
    assert validate_paper_reconciliation_profile("paper") == "paper"
    assert validate_paper_reconciliation_profile("dev") == "dev"
    assert validate_paper_reconciliation_profile("mock") == "mock"

    with pytest.raises(PaperFillProfileError):
        validate_paper_reconciliation_profile("prod")
