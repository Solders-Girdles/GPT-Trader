from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea

from gpt_trader.features.trade_ideas import (
    CloseoutAttributionIntegrityError,
    CloseoutResolution,
    TradeIdeaService,
)
from gpt_trader.features.trade_ideas.closeout import CloseoutAttribution


@pytest.fixture
def service(tmp_path: Path) -> TradeIdeaService:
    return TradeIdeaService(
        tmp_path / "trade_ideas",
        now_factory=lambda: datetime(2026, 6, 12, 10, 0, tzinfo=UTC),
    )


def _record_expired_closeout(service: TradeIdeaService) -> CloseoutAttribution:
    idea = build_trade_idea()
    service.propose(idea, actor_id="idea-generator-v1")
    service.expire(idea.decision_id)
    return service.record_closeout_attribution(
        idea.decision_id,
        actor_id="expiry-sweep",
        resolution=CloseoutResolution.EXPIRY,
        realized_profit_loss_unavailable_reason="Idea expired before entry fill",
    )


def _write_closeout_payload(
    service: TradeIdeaService,
    payload: dict[str, Any],
    *,
    append: bool = False,
) -> None:
    service.closeout_log.path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with service.closeout_log.path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def test_query_closeout_records_excludes_orphaned_log_records(
    service: TradeIdeaService,
) -> None:
    valid_closeout = _record_expired_closeout(service)
    orphan_payload = valid_closeout.to_dict()
    orphan_payload.update(
        {
            "decision_id": "trade-orphan-closeout",
            "terminal_event_id": "evt-orphan",
            "record_hash": "hash-orphan",
        }
    )
    _write_closeout_payload(service, orphan_payload, append=True)

    page = service.query_closeout_records()
    orphan_page = service.query_closeout_records(decision_id="trade-orphan-closeout")

    assert page.items == (valid_closeout,)
    assert page.total_count == 1
    assert orphan_page.items == ()
    assert orphan_page.total_count == 0


@pytest.mark.parametrize(
    ("tampered_field", "match"),
    [
        ("terminal_event_id", "terminal_event_id expected"),
        ("record_hash", "record_hash expected"),
        ("max_loss", "max_loss.amount expected"),
    ],
)
def test_query_closeout_records_rejects_tampered_current_attribution(
    service: TradeIdeaService,
    tampered_field: str,
    match: str,
) -> None:
    closeout = _record_expired_closeout(service)
    payload = closeout.to_dict()
    if tampered_field == "terminal_event_id":
        payload["terminal_event_id"] = "evt-stale"
    elif tampered_field == "record_hash":
        payload["record_hash"] = "stale-record-hash"
    else:
        max_loss = dict(payload["max_loss"])
        max_loss["amount"] = "251"
        payload["max_loss"] = max_loss
    _write_closeout_payload(service, payload)

    with pytest.raises(CloseoutAttributionIntegrityError, match=match):
        service.query_closeout_records()
