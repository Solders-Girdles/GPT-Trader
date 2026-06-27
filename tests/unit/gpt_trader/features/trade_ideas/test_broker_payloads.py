from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea

from gpt_trader.features.trade_ideas import (
    DEFAULT_RISK_BUDGET,
    InvalidTransitionError,
    TradeIdeaService,
    TradeIdeaState,
    canonical_ticket_json,
)


@pytest.fixture
def service(tmp_path: Path) -> TradeIdeaService:
    return TradeIdeaService(
        tmp_path / "trade_ideas",
        now_factory=lambda: datetime(2026, 6, 12, 10, 0, tzinfo=UTC),
    )


def test_export_broker_ticket_payload_for_approved_idea(service: TradeIdeaService) -> None:
    idea = build_trade_idea()
    service.propose(idea, actor_id="idea-generator-v1")
    service.approve(idea.decision_id, actor_id="rj", reason="Risk verified")

    payload = service.export_broker_ticket_payload(
        idea.decision_id,
        venue="coinbase",
        venue_order_type="limit",
        time_in_force="GTC",
    )

    assert payload["schema_version"] == "gpt-trader.trade_idea_ticket.v1"
    assert payload["decision_id"] == idea.decision_id
    assert payload["record_hash"] == idea.record_hash()
    assert payload["broker_ticket"]["source_record"] == idea.broker_ticket.to_dict()
    assert payload["broker_ticket"]["exported"] == {
        "venue": "coinbase",
        "status": "approved",
    }
    assert payload["venue_request"]["client_order_id"].startswith(
        f"gpt-trader-coinbase-{idea.decision_id}-"
    )
    assert payload["venue_payload"]["order_side"] == "buy"
    assert payload["policy_budget_snapshot"]["risk_budget"] == DEFAULT_RISK_BUDGET.to_dict()
    assert payload["provenance"]["approval_event"]["actor_id"] == "rj"
    assert payload["provenance"]["terminal_event"] is None


def test_export_broker_ticket_payload_is_canonical_and_deterministic(
    service: TradeIdeaService,
) -> None:
    idea = build_trade_idea()
    service.propose(idea, actor_id="idea-generator-v1")
    service.approve(idea.decision_id, actor_id="rj", reason="Risk verified")

    first = service.export_broker_ticket_payload(
        idea.decision_id,
        venue="manual",
        venue_order_type="operator_selected",
        time_in_force="operator_selected",
    )
    second = service.export_broker_ticket_payload(
        idea.decision_id,
        venue="manual",
        venue_order_type="operator_selected",
        time_in_force="operator_selected",
    )

    assert first == second
    assert canonical_ticket_json(first) == canonical_ticket_json(second)


def test_export_broker_ticket_rejects_unapproved_state(service: TradeIdeaService) -> None:
    idea = build_trade_idea()
    service.propose(idea, actor_id="idea-generator-v1")

    with pytest.raises(InvalidTransitionError) as exc_info:
        service.export_broker_ticket_payload(
            idea.decision_id,
            venue="manual",
            venue_order_type="operator_selected",
            time_in_force="operator_selected",
        )

    assert exc_info.value.context["field"] == "after_state"
    assert exc_info.value.context["value"] == TradeIdeaState.PROPOSED.value


def test_export_broker_ticket_allows_terminal_state_after_approval(
    service: TradeIdeaService,
) -> None:
    idea = build_trade_idea()
    service.propose(idea, actor_id="idea-generator-v1")
    service.approve(idea.decision_id, actor_id="rj", reason="Risk verified")
    service.record_submission(idea.decision_id, actor_id="operator", venue="manual")
    service.record_fill(
        idea.decision_id,
        actor_id="manual",
        venue="manual",
        external_order_id="external-123",
    )

    payload = service.export_broker_ticket_payload(
        idea.decision_id,
        venue="manual",
        venue_order_type="operator_selected",
        time_in_force="operator_selected",
        client_order_id="manual-ticket-1",
    )

    assert payload["broker_ticket"]["exported"] == {"venue": "manual", "status": "submitted"}
    assert payload["provenance"]["terminal_event"]["action"] == "filled"
    assert payload["provenance"]["terminal_event"]["external_order_id_present"] is True
    assert "external-123" not in canonical_ticket_json(payload)
