from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea
from textual.app import App

from gpt_trader.features.trade_ideas import (
    ActorType,
    AuditAction,
    MaxLoss,
    TradeIdeaService,
    TradeIdeaState,
)
from gpt_trader.tui.screens.ideas_review_screen import IdeasReviewScreen


class IdeasReviewTestApp(App):
    def __init__(self, screen: IdeasReviewScreen) -> None:
        super().__init__()
        self.ideas_screen = screen


@pytest.fixture
def service(tmp_path: Path) -> TradeIdeaService:
    return TradeIdeaService(
        tmp_path / "trade_ideas",
        now_factory=lambda: datetime(2026, 6, 12, 10, 0, tzinfo=UTC),
    )


def _app(service: TradeIdeaService) -> IdeasReviewTestApp:
    return IdeasReviewTestApp(IdeasReviewScreen(service=service, reviewer_id="rj"))


def _text(widget) -> str:
    return str(widget.render())


async def _push_ideas_screen(pilot) -> IdeasReviewScreen:
    await pilot.app.push_screen(pilot.app.ideas_screen)
    await pilot.pause()
    return pilot.app.ideas_screen


@pytest.mark.asyncio
async def test_empty_store_renders_empty_state(service: TradeIdeaService) -> None:
    async with _app(service).run_test(size=(100, 30)) as pilot:
        screen = await _push_ideas_screen(pilot)
        detail = _text(screen.query_one("#ideas-review-detail"))
        table = screen.query_one("#ideas-review-table")

    assert "No trade ideas match" in str(detail)
    assert table.row_count == 0


@pytest.mark.asyncio
async def test_populated_queue_renders_detail_and_policy(service: TradeIdeaService) -> None:
    idea = build_trade_idea()
    service.propose(idea, actor_id="idea-generator-v1")

    async with _app(service).run_test(size=(120, 36)) as pilot:
        screen = await _push_ideas_screen(pilot)
        detail = _text(screen.query_one("#ideas-review-detail"))
        table = screen.query_one("#ideas-review-table")

    assert table.row_count == 1
    assert idea.decision_id in detail
    assert "BTC reclaiming the 50-day average" in detail
    assert "PASS would pass approval policy" in detail
    assert "History:" in detail


@pytest.mark.asyncio
async def test_policy_violations_are_listed(service: TradeIdeaService) -> None:
    idea = build_trade_idea(
        max_loss=MaxLoss(amount=Decimal("1200"), percent_of_account=Decimal("12"))
    )
    service.propose(idea, actor_id="idea-generator-v1")

    async with _app(service).run_test(size=(120, 36)) as pilot:
        screen = await _push_ideas_screen(pilot)
        detail = _text(screen.query_one("#ideas-review-detail"))

    assert "FAIL max_loss 12% exceeds budget cap" in detail


@pytest.mark.asyncio
async def test_approve_flow_records_human_reason(service: TradeIdeaService) -> None:
    idea = build_trade_idea()
    service.propose(idea, actor_id="idea-generator-v1")

    async with _app(service).run_test(size=(120, 36)) as pilot:
        await _push_ideas_screen(pilot)
        await pilot.press("a")
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert service.get(idea.decision_id).state is TradeIdeaState.PROPOSED
        await pilot.click("#ideas-reason-input")
        await pilot.press(*"Risk verified")
        await pilot.press("ctrl+s")
        await pilot.pause()

    view = service.get(idea.decision_id)
    assert view.state is TradeIdeaState.APPROVED
    event = view.events[-1]
    assert event.action is AuditAction.APPROVED
    assert event.actor_type is ActorType.HUMAN
    assert event.actor_id == "rj"
    assert event.reason == "Risk verified"


@pytest.mark.asyncio
async def test_approve_refusal_keeps_state_unchanged(service: TradeIdeaService) -> None:
    idea = build_trade_idea(
        max_loss=MaxLoss(amount=Decimal("1200"), percent_of_account=Decimal("12"))
    )
    service.propose(idea, actor_id="idea-generator-v1")

    async with _app(service).run_test(size=(120, 36)) as pilot:
        await _push_ideas_screen(pilot)
        await pilot.press("a")
        await pilot.pause()
        await pilot.click("#ideas-reason-input")
        await pilot.press(*"Risk accepted")
        await pilot.press("ctrl+s")
        await pilot.pause()

    view = service.get(idea.decision_id)
    assert view.state is TradeIdeaState.PROPOSED
    assert [event.action for event in view.events] == [AuditAction.PROPOSED]


@pytest.mark.asyncio
async def test_reject_and_request_changes_record_events(service: TradeIdeaService) -> None:
    rejected = build_trade_idea(decision_id="trade-20260612-reject")
    needs_changes = build_trade_idea(decision_id="trade-20260612-changes")
    service.propose(rejected, actor_id="idea-generator-v1")
    service.propose(needs_changes, actor_id="idea-generator-v1")

    async with _app(service).run_test(size=(120, 36)) as pilot:
        screen = await _push_ideas_screen(pilot)
        screen._mutate("reject", rejected.decision_id, "No longer attractive")
        screen._mutate("request_changes", needs_changes.decision_id, "Add invalidation evidence")

    rejected_view = service.get(rejected.decision_id)
    changes_view = service.get(needs_changes.decision_id)
    assert rejected_view.state is TradeIdeaState.REJECTED
    assert rejected_view.events[-1].actor_type is ActorType.HUMAN
    assert rejected_view.events[-1].reason == "No longer attractive"
    assert changes_view.state is TradeIdeaState.NEEDS_CHANGES
    assert changes_view.events[-1].actor_type is ActorType.HUMAN
    assert changes_view.events[-1].reason == "Add invalidation evidence"


@pytest.mark.asyncio
async def test_terminal_state_action_is_noop(service: TradeIdeaService) -> None:
    idea = build_trade_idea()
    service.propose(idea, actor_id="idea-generator-v1")
    service.reject(idea.decision_id, actor_id="rj", reason="Rejected already")

    async with _app(service).run_test(size=(120, 36)) as pilot:
        await _push_ideas_screen(pilot)
        await pilot.press("a")
        await pilot.pause()

    view = service.get(idea.decision_id)
    assert view.state is TradeIdeaState.REJECTED
    assert [event.action for event in view.events] == [
        AuditAction.PROPOSED,
        AuditAction.REJECTED,
    ]


@pytest.mark.asyncio
async def test_filter_cycle_changes_visible_rows(service: TradeIdeaService) -> None:
    proposed = build_trade_idea(decision_id="trade-20260612-proposed")
    approved = build_trade_idea(decision_id="trade-20260612-approved")
    service.propose(proposed, actor_id="idea-generator-v1")
    service.propose(approved, actor_id="idea-generator-v1")
    service.approve(approved.decision_id, actor_id="rj", reason="Risk verified")

    async with _app(service).run_test(size=(120, 36)) as pilot:
        screen = await _push_ideas_screen(pilot)
        table = screen.query_one("#ideas-review-table")
        assert table.row_count == 2
        await pilot.press("f")
        await pilot.pause()
        assert table.row_count == 1
        assert "proposed" in _text(screen.query_one("#ideas-review-filter"))
        await pilot.press("f")
        await pilot.pause()
        assert table.row_count == 0
        assert "needs_changes" in _text(screen.query_one("#ideas-review-filter"))


@pytest.mark.asyncio
async def test_refresh_preserves_selected_visible_idea(service: TradeIdeaService) -> None:
    first = build_trade_idea(decision_id="trade-20260612-001")
    selected = build_trade_idea(decision_id="trade-20260612-002")
    third = build_trade_idea(decision_id="trade-20260612-003")
    service.propose(first, actor_id="idea-generator-v1")
    service.propose(selected, actor_id="idea-generator-v1")
    service.propose(third, actor_id="idea-generator-v1")

    async with _app(service).run_test(size=(120, 36)) as pilot:
        screen = await _push_ideas_screen(pilot)
        table = screen.query_one("#ideas-review-table")
        table.move_cursor(row=1)
        screen._select_row_key(selected.decision_id)
        await pilot.pause()

        screen.refresh_views(notify=False)
        await pilot.pause()

        selected_row = table.get_row_at(table.cursor_row)
        detail = _text(screen.query_one("#ideas-review-detail"))

    assert screen._selected_decision_id == selected.decision_id
    assert selected_row[0] == selected.decision_id
    assert table.cursor_row == 1
    assert selected.decision_id in detail
