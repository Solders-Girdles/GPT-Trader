from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea

import gpt_trader.features.trade_ideas.service as trade_idea_service_module
from gpt_trader.features.trade_ideas import TradeIdea, TradeIdeaService


def test_propose_batch_rolls_back_partial_record_when_save_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "trade_ideas"
    service = TradeIdeaService(
        root,
        now_factory=lambda: datetime(2026, 6, 12, 10, 0, tzinfo=UTC),
    )
    first = build_trade_idea(decision_id="trade-20260612-batch-001")
    second = build_trade_idea(decision_id="trade-20260612-batch-002")
    original_save = service._store.save

    def save_with_partial_failure(idea: TradeIdea) -> str:
        if idea.decision_id == second.decision_id:
            decision_dir = root / "records" / idea.decision_id
            decision_dir.mkdir(parents=True)
            (decision_dir / "latest.json").write_text(
                json.dumps({"decision_id": idea.decision_id}),
                encoding="utf-8",
            )
            raise RuntimeError("forced partial save failure")
        return original_save(idea)

    monkeypatch.setattr(service._store, "save", save_with_partial_failure)

    with pytest.raises(RuntimeError, match="forced partial save failure"):
        service.propose_batch(
            (first, second),
            actor_id="idea-generator-v1",
        )

    assert not (root / "records" / first.decision_id).exists()
    assert not (root / "records" / second.decision_id).exists()
    assert not (root / "audit.jsonl").exists()


def test_propose_batch_surfaces_rollback_delete_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "trade_ideas"
    service = TradeIdeaService(
        root,
        now_factory=lambda: datetime(2026, 6, 12, 10, 0, tzinfo=UTC),
    )
    first = build_trade_idea(decision_id="trade-20260612-batch-001")
    second = build_trade_idea(decision_id="trade-20260612-batch-002")
    original_append = trade_idea_service_module.TradeIdeaAuditLog.append
    original_rmtree = trade_idea_service_module.shutil.rmtree
    append_calls = 0

    def fail_second_append(*args: object, **kwargs: object) -> None:
        nonlocal append_calls
        append_calls += 1
        if append_calls == 2:
            raise RuntimeError("forced second audit failure")
        original_append(*args, **kwargs)

    def fail_first_delete(path: Path) -> None:
        if path.name == first.decision_id:
            raise PermissionError("forced delete failure")
        original_rmtree(path)

    monkeypatch.setattr(trade_idea_service_module.TradeIdeaAuditLog, "append", fail_second_append)
    monkeypatch.setattr(trade_idea_service_module.shutil, "rmtree", fail_first_delete)

    with pytest.raises(PermissionError, match="forced delete failure"):
        service.propose_batch(
            (first, second),
            actor_id="idea-generator-v1",
        )

    assert append_calls == 2
    assert (root / "records" / first.decision_id).exists()
