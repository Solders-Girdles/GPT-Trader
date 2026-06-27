from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea

from gpt_trader.features.trade_ideas import AuditAction, TradeIdeaService


@pytest.fixture
def service(tmp_path: Path) -> TradeIdeaService:
    return TradeIdeaService(
        tmp_path / "trade_ideas",
        now_factory=lambda: datetime(2026, 6, 12, 10, 0, tzinfo=UTC),
    )


def test_audit_query_preserves_append_order_for_tied_timestamps(
    service: TradeIdeaService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_ids = iter(("evt-d", "evt-c", "evt-b", "evt-a"))
    monkeypatch.setattr(
        "gpt_trader.features.trade_ideas.service.new_event_id",
        lambda: next(event_ids),
    )
    idea = build_trade_idea()
    service.propose(idea, actor_id="idea-generator-v1")
    service.approve(idea.decision_id, actor_id="rj", reason="Thesis and risk verified")
    service.record_submission(idea.decision_id, actor_id="executor", venue="manual")
    service.record_fill(idea.decision_id, actor_id="executor", venue="manual")

    page = service.list_audit_events(decision_id=idea.decision_id)

    assert [event.action for event in page.items] == [
        AuditAction.PROPOSED,
        AuditAction.APPROVED,
        AuditAction.SUBMITTED,
        AuditAction.FILLED,
    ]
    assert [event.event_id for event in page.items] == ["evt-d", "evt-c", "evt-b", "evt-a"]
