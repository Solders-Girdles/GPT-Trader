from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea

from gpt_trader.features.trade_ideas import (
    AuditIntegrityError,
    TradeIdeaService,
    TradeIdeaStore,
)


def test_interrupted_resubmit_latest_hash_mismatch_is_integrity_error(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trade_ideas"
    service = TradeIdeaService(
        root,
        now_factory=lambda: datetime(2026, 6, 12, 10, 0, tzinfo=UTC),
    )
    idea = build_trade_idea(decision_id="trade-20260612-interrupted-resubmit")
    service.propose(idea, actor_id="idea-generator-v1")
    service.request_changes(idea.decision_id, actor_id="rj", reason="Tighten invalidation")
    audit_path = root / "audit.jsonl"
    original_audit = audit_path.read_text(encoding="utf-8")
    unaudited_revision = build_trade_idea(
        decision_id=idea.decision_id,
        invalidation="Daily close below 59000",
    )
    TradeIdeaStore(root / "records").save(unaudited_revision)

    with pytest.raises(AuditIntegrityError, match="does not match latest audit record_hash"):
        service.get(idea.decision_id)
    with pytest.raises(AuditIntegrityError, match="does not match latest audit record_hash"):
        service.list_views()
    with pytest.raises(AuditIntegrityError, match="does not match latest audit record_hash"):
        service.reject(idea.decision_id, actor_id="rj", reason="Reject unaudited revision")

    assert audit_path.read_text(encoding="utf-8") == original_audit
    assert service.audit_log.read_events(idea.decision_id)[-1].record_hash == idea.record_hash()
    stored = TradeIdeaStore(root / "records").load_latest(idea.decision_id)
    assert stored is not None
    assert stored.record_hash() == unaudited_revision.record_hash()
