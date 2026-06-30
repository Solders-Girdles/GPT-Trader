"""Conflict / no-fallback safety tests for paper-fill reconciliation.

Extracted from test_paper_reconciliation.py: these assert the reconciler never
mis-matches a fill to the wrong trade idea (unknown decision_id must not fall
back to client_order_id or symbol/side; symbol/side conflicts are rejected
without mutating the idea).
"""

from __future__ import annotations

from pathlib import Path

from tests.unit.gpt_trader.features.trade_ideas.conftest import (
    approved_idea as _approved_idea,
)
from tests.unit.gpt_trader.features.trade_ideas.conftest import (
    paper_fill_event as _fill_event,
)
from tests.unit.gpt_trader.features.trade_ideas.conftest import (
    reconciliation_service as _service,
)

from gpt_trader.features.trade_ideas import (
    PaperFillReconciler,
    TradeIdeaState,
)


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
