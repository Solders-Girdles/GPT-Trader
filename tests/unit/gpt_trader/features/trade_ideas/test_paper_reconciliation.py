from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.gpt_trader.features.trade_ideas.conftest import (
    approved_idea as _approved_idea,
)
from tests.unit.gpt_trader.features.trade_ideas.conftest import (
    paper_fill_event as _fill_event,
)
from tests.unit.gpt_trader.features.trade_ideas.conftest import (
    reconciliation_service as _service,
)
from tests.unit.gpt_trader.features.trade_ideas.conftest import (
    submitted_idea as _submitted_idea,
)

from gpt_trader.features.trade_ideas import (
    ActorType,
    AuditAction,
    PaperFillProfileError,
    PaperFillReconciler,
    TradeIdeaState,
    validate_paper_reconciliation_profile,
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


def test_reconciler_does_not_fall_back_from_unknown_decision_id_to_client_order_id(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "ideas")
    decision_id = _approved_idea(service)
    original_events = tuple(service.get(decision_id).events)

    report = PaperFillReconciler(service).reconcile_fills(
        [
            _fill_event(
                decision_id="trade-20260612-missing",
                client_order_id=decision_id,
                symbol="BTC-USD",
                side="buy",
            )
        ],
        apply=True,
    )

    assert report.matched_count == 0
    assert report.unmatched_count == 1
    assert report.unmatched[0].reason == "explicit decision_id not found: trade-20260612-missing"
    view = service.get(decision_id)
    assert view.state is TradeIdeaState.APPROVED
    assert tuple(view.events) == original_events


def test_reconciler_does_not_fall_back_from_unknown_decision_id_to_symbol_side(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "ideas")
    decision_id = _approved_idea(service)
    original_events = tuple(service.get(decision_id).events)

    report = PaperFillReconciler(service).reconcile_fills(
        [
            _fill_event(
                decision_id="trade-20260612-missing",
                client_order_id="",
                symbol="BTC-USD",
                side="buy",
            )
        ],
        apply=True,
    )

    assert report.matched_count == 0
    assert report.unmatched_count == 1
    assert report.unmatched[0].reason == "explicit decision_id not found: trade-20260612-missing"
    view = service.get(decision_id)
    assert view.state is TradeIdeaState.APPROVED
    assert tuple(view.events) == original_events


def test_reconciler_rejects_symbol_conflict_on_explicit_decision_id_without_mutation(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "ideas")
    decision_id = _approved_idea(service)
    original_events = tuple(service.get(decision_id).events)

    report = PaperFillReconciler(service).reconcile_fills(
        [_fill_event(decision_id=decision_id, symbol="ETH-USD", side="buy")],
        apply=True,
    )

    assert report.matched_count == 0
    assert report.unmatched_count == 1
    entry = report.unmatched[0]
    assert entry.reason == "fill symbol conflicts with matched trade idea"
    assert entry.decision_id == decision_id
    assert entry.match_method == "decision_id"
    assert entry.context == {
        "field": "symbol",
        "event_symbol": "ETH-USD",
        "idea_symbol": "BTC-USD",
    }
    view = service.get(decision_id)
    assert view.state is TradeIdeaState.APPROVED
    assert tuple(view.events) == original_events


def test_reconciler_rejects_side_conflict_on_explicit_decision_id_without_mutation(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "ideas")
    decision_id = _approved_idea(service)
    original_events = tuple(service.get(decision_id).events)

    report = PaperFillReconciler(service).reconcile_fills(
        [_fill_event(decision_id=decision_id, symbol="BTC-USD", side="sell")],
        apply=True,
    )

    assert report.matched_count == 0
    assert report.unmatched_count == 1
    entry = report.unmatched[0]
    assert entry.reason == "fill side conflicts with matched trade idea"
    assert entry.decision_id == decision_id
    assert entry.match_method == "decision_id"
    assert entry.context == {
        "field": "side",
        "event_side": "sell",
        "idea_direction": "long",
        "expected_side": "buy",
    }
    view = service.get(decision_id)
    assert view.state is TradeIdeaState.APPROVED
    assert tuple(view.events) == original_events


def test_reconciler_rejects_symbol_conflict_on_client_order_id_without_mutation(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "ideas")
    decision_id = _approved_idea(service)
    original_events = tuple(service.get(decision_id).events)

    report = PaperFillReconciler(service).reconcile_fills(
        [_fill_event(client_order_id=decision_id, symbol="ETH-USD", side="buy")],
        apply=True,
    )

    assert report.matched_count == 0
    assert report.unmatched_count == 1
    entry = report.unmatched[0]
    assert entry.reason == "fill symbol conflicts with matched trade idea"
    assert entry.decision_id == decision_id
    assert entry.match_method == "client_order_id"
    assert entry.context["field"] == "symbol"
    view = service.get(decision_id)
    assert view.state is TradeIdeaState.APPROVED
    assert tuple(view.events) == original_events


def test_reconciler_rejects_side_conflict_on_client_order_id_without_mutation(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "ideas")
    decision_id = _approved_idea(service)
    original_events = tuple(service.get(decision_id).events)

    report = PaperFillReconciler(service).reconcile_fills(
        [_fill_event(client_order_id=decision_id, symbol="BTC-USD", side="sell")],
        apply=True,
    )

    assert report.matched_count == 0
    assert report.unmatched_count == 1
    entry = report.unmatched[0]
    assert entry.reason == "fill side conflicts with matched trade idea"
    assert entry.decision_id == decision_id
    assert entry.match_method == "client_order_id"
    assert entry.context["field"] == "side"
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


def test_reconciler_refreshes_symbol_side_candidates_after_applied_fill(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "ideas")
    first_decision_id = _approved_idea(service, decision_id="trade-20260612-001")
    second_decision_id = _approved_idea(service, decision_id="trade-20260612-002")

    report = PaperFillReconciler(service).reconcile_fills(
        [
            _fill_event(
                client_order_id=first_decision_id,
                order_id="MOCK_000005",
            ),
            _fill_event(
                order_id="MOCK_000006",
                client_order_id="",
            ),
        ],
        apply=True,
    )

    assert report.matched_count == 2
    assert report.unmatched_count == 0
    assert [entry.decision_id for entry in report.matched] == [
        first_decision_id,
        second_decision_id,
    ]
    assert [entry.match_method for entry in report.matched] == [
        "client_order_id",
        "symbol_side",
    ]
    assert service.get(first_decision_id).state is TradeIdeaState.FILLED
    assert service.get(second_decision_id).state is TradeIdeaState.FILLED


def test_reconciler_rerun_skips_legacy_fill_already_audited_on_other_idea(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "ideas")
    first_decision_id = _approved_idea(service, decision_id="trade-20260612-001")
    legacy_fill = _fill_event(order_id="MOCK_LEGACY", client_order_id="")

    first_report = PaperFillReconciler(service).reconcile_fills([legacy_fill], apply=True)
    assert first_report.matched_count == 1
    assert first_report.matched[0].decision_id == first_decision_id

    # A newer same-symbol/side idea is approved, then reconciliation is rerun
    # over the full event store (which still contains the legacy fill).
    second_decision_id = _approved_idea(service, decision_id="trade-20260612-002")

    rerun_report = PaperFillReconciler(service).reconcile_fills([legacy_fill], apply=True)

    assert rerun_report.matched_count == 0
    assert rerun_report.skipped_count == 1
    assert rerun_report.skipped[0].reason == "fill already recorded on trade-idea audit trail"
    # The legacy fill must not be re-recorded against the newer idea.
    assert service.get(second_decision_id).state is TradeIdeaState.APPROVED
    assert service.get(first_decision_id).state is TradeIdeaState.FILLED


def test_submitted_idea_only_matches_legacy_fill_with_its_order_id(tmp_path: Path) -> None:
    service = _service(tmp_path / "ideas")
    decision_id = _submitted_idea(service, external_order_id="ORDER_A")
    reconciler = PaperFillReconciler(service)

    # Legacy fill (no ids) for a different order id must not attach to ORDER_A.
    mismatched = reconciler.reconcile_fills(
        [_fill_event(order_id="ORDER_B", client_order_id="")], apply=True
    )
    assert mismatched.matched_count == 0
    assert service.get(decision_id).state is TradeIdeaState.SUBMITTED

    # The fill carrying the submitted order id reconciles normally.
    matched = reconciler.reconcile_fills(
        [_fill_event(order_id="ORDER_A", client_order_id="")], apply=True
    )
    assert matched.matched_count == 1
    assert service.get(decision_id).state is TradeIdeaState.FILLED


def test_reconciler_dry_run_previews_refreshed_symbol_side_candidates(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "ideas")
    first_decision_id = _approved_idea(service, decision_id="trade-20260612-001")
    second_decision_id = _approved_idea(service, decision_id="trade-20260612-002")
    fills = [
        _fill_event(
            client_order_id=first_decision_id,
            order_id="MOCK_000007",
        ),
        _fill_event(
            order_id="MOCK_000008",
            client_order_id="",
        ),
    ]

    dry_run_report = PaperFillReconciler(service).reconcile_fills(fills, apply=False)

    assert dry_run_report.mode == "dry_run"
    assert dry_run_report.matched_count == 2
    assert dry_run_report.unmatched_count == 0
    assert [entry.decision_id for entry in dry_run_report.matched] == [
        first_decision_id,
        second_decision_id,
    ]
    assert [entry.match_method for entry in dry_run_report.matched] == [
        "client_order_id",
        "symbol_side",
    ]
    assert [entry.final_state for entry in dry_run_report.matched] == [
        TradeIdeaState.FILLED.value,
        TradeIdeaState.FILLED.value,
    ]
    assert service.get(first_decision_id).state is TradeIdeaState.APPROVED
    assert service.get(second_decision_id).state is TradeIdeaState.APPROVED

    apply_report = PaperFillReconciler(service).reconcile_fills(fills, apply=True)

    assert [entry.decision_id for entry in apply_report.matched] == [
        first_decision_id,
        second_decision_id,
    ]
    assert [entry.match_method for entry in apply_report.matched] == [
        "client_order_id",
        "symbol_side",
    ]
    assert apply_report.unmatched_count == 0


def test_live_profiles_are_rejected_for_paper_reconciliation() -> None:
    assert validate_paper_reconciliation_profile("paper") == "paper"
    assert validate_paper_reconciliation_profile("dev") == "dev"
    assert validate_paper_reconciliation_profile("mock") == "mock"

    with pytest.raises(PaperFillProfileError):
        validate_paper_reconciliation_profile("prod")
