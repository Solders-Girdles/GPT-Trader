from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from gpt_trader import cli
from gpt_trader.features.trade_ideas import TimeHorizon, TradeIdeaService, TradeIdeaState
from gpt_trader.persistence.event_store import EventStore
from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea


def _future_horizon() -> TimeHorizon:
    return TimeHorizon(
        expected_hold="3-10 days",
        expires_at=datetime(2035, 6, 19, 16, 0, tzinfo=UTC),
    )


def _run_json(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, dict[str, Any]]:
    exit_code = cli.main(argv)
    output = capsys.readouterr().out
    assert output
    return exit_code, json.loads(output)


def _seed_approved_idea(root: Path) -> str:
    service = TradeIdeaService(
        root,
        now_factory=lambda: datetime(2026, 6, 12, 10, 0, tzinfo=UTC),
    )
    idea = build_trade_idea(time_horizon=_future_horizon())
    service.propose(idea, actor_id="idea-generator-v1")
    service.approve(idea.decision_id, actor_id="rj", reason="Risk verified")
    return idea.decision_id


def _append_fill_event(event_store_root: Path, decision_id: str) -> None:
    event_store = EventStore(root=event_store_root)
    event_store.append_trade(
        "paper-bot",
        {
            "order_id": "MOCK_000001",
            "client_order_id": decision_id,
            "symbol": "BTC-USD",
            "side": "BUY",
            "quantity": "0.1",
            "price": "60750",
            "status": "filled",
        },
    )
    event_store.close()


def _root_args(root: Path) -> list[str]:
    return ["--ideas-root", str(root), "--format", "json"]


def test_reconcile_paper_fills_dry_run_does_not_mutate_audit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ideas_root = tmp_path / "ideas"
    event_store_root = tmp_path / "events"
    decision_id = _seed_approved_idea(ideas_root)
    _append_fill_event(event_store_root, decision_id)

    exit_code, response = _run_json(
        capsys,
        [
            "ideas",
            "reconcile-paper-fills",
            *_root_args(ideas_root),
            "--profile",
            "paper",
            "--event-store-root",
            str(event_store_root),
        ],
    )

    assert exit_code == 0
    assert response["data"]["mode"] == "dry_run"
    assert response["data"]["matched_count"] == 1
    assert response["data"]["recorded_count"] == 0
    assert response["metadata"]["was_noop"] is True
    assert TradeIdeaService(ideas_root).get(decision_id).state is TradeIdeaState.APPROVED


def test_reconcile_paper_fills_apply_records_submission_and_fill(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ideas_root = tmp_path / "ideas"
    event_store_root = tmp_path / "events"
    decision_id = _seed_approved_idea(ideas_root)
    _append_fill_event(event_store_root, decision_id)

    exit_code, response = _run_json(
        capsys,
        [
            "ideas",
            "reconcile-paper-fills",
            *_root_args(ideas_root),
            "--profile",
            "paper",
            "--event-store-root",
            str(event_store_root),
            "--apply",
        ],
    )

    assert exit_code == 0
    assert response["data"]["mode"] == "apply"
    assert response["data"]["recorded_count"] == 1
    assert response["metadata"]["was_noop"] is False
    assert TradeIdeaService(ideas_root).get(decision_id).state is TradeIdeaState.FILLED


def test_reconcile_paper_fills_rejects_live_profile_without_mutation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ideas_root = tmp_path / "ideas"
    event_store_root = tmp_path / "events"
    decision_id = _seed_approved_idea(ideas_root)
    _append_fill_event(event_store_root, decision_id)

    exit_code, response = _run_json(
        capsys,
        [
            "ideas",
            "reconcile-paper-fills",
            *_root_args(ideas_root),
            "--profile",
            "prod",
            "--event-store-root",
            str(event_store_root),
            "--apply",
        ],
    )

    assert exit_code == 1
    assert response["errors"][0]["code"] == "VALIDATION_ERROR"
    assert response["errors"][0]["details"]["field"] == "profile"
    assert TradeIdeaService(ideas_root).get(decision_id).state is TradeIdeaState.APPROVED
