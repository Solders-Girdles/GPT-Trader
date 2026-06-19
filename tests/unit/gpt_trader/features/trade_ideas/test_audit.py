from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gpt_trader.features.trade_ideas import (
    ActorType,
    AuditAction,
    AuditEvent,
    AuditIntegrityError,
    InvalidTransitionError,
    TradeIdea,
    TradeIdeaAuditLog,
    TradeIdeaState,
    new_event_id,
)


def build_event(
    idea: TradeIdea,
    *,
    action: AuditAction,
    before_state: TradeIdeaState | None,
    after_state: TradeIdeaState,
    actor_type: ActorType = ActorType.HUMAN,
    minute: int = 0,
) -> AuditEvent:
    return AuditEvent(
        event_id=new_event_id(),
        timestamp=datetime(2026, 6, 12, 10, minute, tzinfo=UTC),
        decision_id=idea.decision_id,
        actor_type=actor_type,
        actor_id="rj" if actor_type is ActorType.HUMAN else "idea-generator-v1",
        action=action,
        before_state=before_state,
        after_state=after_state,
        reason="test event",
        record_hash=idea.record_hash(),
    )


@pytest.fixture
def audit_log(tmp_path: Path) -> TradeIdeaAuditLog:
    return TradeIdeaAuditLog(tmp_path / "trade_ideas" / "audit.jsonl")


def propose(audit_log: TradeIdeaAuditLog, idea: TradeIdea) -> None:
    audit_log.append(
        build_event(
            idea,
            action=AuditAction.PROPOSED,
            before_state=None,
            after_state=TradeIdeaState.PROPOSED,
            actor_type=ActorType.AI,
        )
    )


def test_append_and_read_round_trip(audit_log: TradeIdeaAuditLog, trade_idea: TradeIdea) -> None:
    propose(audit_log, trade_idea)

    events = audit_log.read_events(trade_idea.decision_id)

    assert len(events) == 1
    assert events[0].action is AuditAction.PROPOSED
    assert events[0].record_hash == trade_idea.record_hash()


def test_log_is_one_json_object_per_line(
    audit_log: TradeIdeaAuditLog, trade_idea: TradeIdea
) -> None:
    propose(audit_log, trade_idea)
    audit_log.append(
        build_event(
            trade_idea,
            action=AuditAction.APPROVED,
            before_state=TradeIdeaState.PROPOSED,
            after_state=TradeIdeaState.APPROVED,
            minute=1,
        )
    )

    lines = audit_log.path.read_text().strip().splitlines()

    assert len(lines) == 2
    assert all(isinstance(json.loads(line), dict) for line in lines)


def test_current_state_tracks_latest_event(
    audit_log: TradeIdeaAuditLog, trade_idea: TradeIdea
) -> None:
    assert audit_log.current_state(trade_idea.decision_id) is None

    propose(audit_log, trade_idea)
    assert audit_log.current_state(trade_idea.decision_id) is TradeIdeaState.PROPOSED

    audit_log.append(
        build_event(
            trade_idea,
            action=AuditAction.APPROVED,
            before_state=TradeIdeaState.PROPOSED,
            after_state=TradeIdeaState.APPROVED,
            minute=1,
        )
    )
    assert audit_log.current_state(trade_idea.decision_id) is TradeIdeaState.APPROVED


def test_append_rejects_stale_before_state(
    audit_log: TradeIdeaAuditLog, trade_idea: TradeIdea
) -> None:
    propose(audit_log, trade_idea)

    with pytest.raises(AuditIntegrityError):
        propose(audit_log, trade_idea)


def test_append_rejects_illegal_transition(
    audit_log: TradeIdeaAuditLog, trade_idea: TradeIdea
) -> None:
    propose(audit_log, trade_idea)

    with pytest.raises(InvalidTransitionError):
        audit_log.append(
            build_event(
                trade_idea,
                action=AuditAction.SUBMITTED,
                before_state=TradeIdeaState.PROPOSED,
                after_state=TradeIdeaState.SUBMITTED,
                minute=1,
            )
        )

    assert audit_log.current_state(trade_idea.decision_id) is TradeIdeaState.PROPOSED


def test_rejected_append_leaves_log_untouched(
    audit_log: TradeIdeaAuditLog, trade_idea: TradeIdea
) -> None:
    propose(audit_log, trade_idea)
    before = audit_log.path.read_text()

    with pytest.raises(AuditIntegrityError):
        propose(audit_log, trade_idea)

    assert audit_log.path.read_text() == before


def test_read_events_filters_by_decision(audit_log: TradeIdeaAuditLog) -> None:
    from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea

    first = build_trade_idea(decision_id="trade-20260612-001")
    second = build_trade_idea(decision_id="trade-20260612-002")
    propose(audit_log, first)
    propose(audit_log, second)

    assert len(audit_log.read_events()) == 2
    assert [event.decision_id for event in audit_log.read_events("trade-20260612-002")] == [
        "trade-20260612-002"
    ]
