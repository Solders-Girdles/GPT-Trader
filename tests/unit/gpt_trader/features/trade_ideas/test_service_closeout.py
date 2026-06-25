from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea

from gpt_trader.features.trade_ideas import (
    ActorType,
    AuditAction,
    CloseoutAttributionIntegrityError,
    CloseoutResolution,
    DuplicateCloseoutAttributionError,
    InvalidTransitionError,
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


def test_record_closeout_attribution_for_filled_idea(service: TradeIdeaService) -> None:
    idea = build_trade_idea()
    service.propose(idea, actor_id="idea-generator-v1")
    service.approve(idea.decision_id, actor_id="rj", reason="Thesis and risk verified")
    service.record_submission(idea.decision_id, actor_id="executor", venue="coinbase")
    filled = service.record_fill(
        idea.decision_id,
        actor_id="coinbase",
        venue="coinbase",
        external_order_id="abc-123",
    )

    closeout = service.record_closeout_attribution(
        idea.decision_id,
        actor_id="rj",
        resolution=CloseoutResolution.THESIS_TARGET,
        realized_profit_loss_amount=Decimal("125.50"),
        realized_profit_loss_percent=Decimal("2.4"),
        evidence=("broker-statement:abc-123", "chart:target-hit"),
    )

    assert closeout.decision_id == idea.decision_id
    assert closeout.actor_type == ActorType.HUMAN.value
    assert closeout.terminal_event_id == filled.events[-1].event_id
    assert closeout.record_hash == filled.events[-1].record_hash
    assert closeout.max_loss.amount == idea.max_loss.amount
    assert closeout.max_loss.percent_of_account == idea.max_loss.percent_of_account
    assert closeout.evidence == ("broker-statement:abc-123", "chart:target-hit")
    view = service.get(idea.decision_id)
    assert view.state is TradeIdeaState.FILLED
    assert view.closeout_attribution == closeout
    assert [event.action for event in view.events] == [
        AuditAction.PROPOSED,
        AuditAction.APPROVED,
        AuditAction.SUBMITTED,
        AuditAction.FILLED,
    ]


def test_record_closeout_attribution_for_expired_idea_without_profit_loss(
    service: TradeIdeaService,
) -> None:
    idea = build_trade_idea()
    service.propose(idea, actor_id="idea-generator-v1")
    expired = service.expire(idea.decision_id)

    closeout = service.record_closeout_attribution(
        idea.decision_id,
        actor_id="expiry-sweep",
        actor_type=ActorType.SYSTEM,
        resolution="expiry",
        realized_profit_loss_unavailable_reason="Idea expired before entry fill",
        evidence=("expiry-sweep:2026-06-12",),
    )

    assert closeout.actor_type == ActorType.SYSTEM.value
    assert closeout.resolution is CloseoutResolution.EXPIRY
    assert closeout.realized_profit_loss_amount is None
    assert closeout.realized_profit_loss_percent is None
    assert closeout.realized_profit_loss_unavailable_reason == "Idea expired before entry fill"
    assert closeout.terminal_event_id == expired.events[-1].event_id


def test_record_closeout_attribution_rejects_non_terminal_idea(
    service: TradeIdeaService,
) -> None:
    idea = build_trade_idea()
    service.propose(idea, actor_id="idea-generator-v1")

    with pytest.raises(InvalidTransitionError, match="must be terminal"):
        service.record_closeout_attribution(
            idea.decision_id,
            actor_id="rj",
            resolution=CloseoutResolution.INVALIDATION,
            realized_profit_loss_amount=Decimal("-250"),
        )

    assert service.get_closeout_attribution(idea.decision_id) is None


def test_record_closeout_attribution_rejects_unknown_idea(
    service: TradeIdeaService,
) -> None:
    with pytest.raises(UnknownTradeIdeaError):
        service.record_closeout_attribution(
            "trade-missing",
            actor_id="rj",
            resolution=CloseoutResolution.INVALIDATION,
            realized_profit_loss_amount=Decimal("-250"),
        )


def test_record_closeout_attribution_rejects_duplicate_conflict(
    service: TradeIdeaService,
) -> None:
    idea = build_trade_idea()
    service.propose(idea, actor_id="idea-generator-v1")
    service.expire(idea.decision_id)
    original = service.record_closeout_attribution(
        idea.decision_id,
        actor_id="expiry-sweep",
        actor_type=ActorType.SYSTEM,
        resolution=CloseoutResolution.EXPIRY,
        realized_profit_loss_unavailable_reason="Idea expired before entry fill",
    )

    with pytest.raises(DuplicateCloseoutAttributionError):
        service.record_closeout_attribution(
            idea.decision_id,
            actor_id="rj",
            resolution=CloseoutResolution.INVALIDATION,
            realized_profit_loss_amount=Decimal("-250"),
            evidence=("manual-review",),
        )

    assert service.get_closeout_attribution(idea.decision_id) == original


def test_record_closeout_attribution_rejects_malformed_numeric_payload(
    service: TradeIdeaService,
) -> None:
    idea = build_trade_idea()
    service.propose(idea, actor_id="idea-generator-v1")
    service.expire(idea.decision_id)

    with pytest.raises(ValueError, match="realized_profit_loss_amount must be a finite decimal"):
        service.record_closeout_attribution(
            idea.decision_id,
            actor_id="rj",
            resolution=CloseoutResolution.INVALIDATION,
            realized_profit_loss_amount="not-a-decimal",
        )

    assert service.get_closeout_attribution(idea.decision_id) is None


def test_persisted_closeout_log_rejects_non_object_max_loss_with_line_context(
    service: TradeIdeaService,
) -> None:
    idea = build_trade_idea()
    service.propose(idea, actor_id="idea-generator-v1")
    expired = service.expire(idea.decision_id)
    payload = {
        "decision_id": idea.decision_id,
        "timestamp": "2026-06-12T10:05:00+00:00",
        "actor_type": "human",
        "actor_id": "rj",
        "terminal_event_id": expired.events[-1].event_id,
        "record_hash": expired.events[-1].record_hash,
        "resolution": CloseoutResolution.EXPIRY.value,
        "realized_profit_loss_amount": None,
        "realized_profit_loss_percent": None,
        "realized_profit_loss_unavailable_reason": "Idea expired before entry fill",
        "max_loss": None,
        "evidence": [],
    }
    service.closeout_log.path.parent.mkdir(parents=True, exist_ok=True)
    service.closeout_log.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(
        CloseoutAttributionIntegrityError,
        match="Closeout attribution log line 1 is malformed: max_loss must be a JSON object",
    ) as log_exc_info:
        service.closeout_log.read_records()
    assert log_exc_info.value.context["field"] == "line"
    assert log_exc_info.value.context["value"] == 1

    with pytest.raises(
        CloseoutAttributionIntegrityError,
        match="Closeout attribution log line 1 is malformed: max_loss must be a JSON object",
    ):
        service.get(idea.decision_id)

    with pytest.raises(
        CloseoutAttributionIntegrityError,
        match="Closeout attribution log line 1 is malformed: max_loss must be a JSON object",
    ):
        service.list_views()
