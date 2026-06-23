from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from gpt_trader import cli
from gpt_trader.features.trade_ideas import TimeHorizon, TradeIdeaService, TradeIdeaStore
from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea


def _future_horizon() -> TimeHorizon:
    return TimeHorizon(
        expected_hold="3-10 days",
        expires_at=datetime(2035, 6, 19, 16, 0, tzinfo=UTC),
    )


def _idea_payload(**overrides: Any) -> dict[str, Any]:
    fields = {"time_horizon": _future_horizon(), **overrides}
    return build_trade_idea(**fields).to_dict()


def _write_idea(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run_json(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, dict[str, Any]]:
    exit_code = cli.main(argv)
    output = capsys.readouterr().out
    assert output
    return exit_code, json.loads(output)


def _root_args(root: Path) -> list[str]:
    return ["--ideas-root", str(root), "--format", "json"]


def test_propose_preview_ignores_unrelated_unaudited_latest_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ideas"
    service = TradeIdeaService(
        root,
        now_factory=lambda: datetime(2026, 6, 12, 10, 0, tzinfo=UTC),
    )
    bad = build_trade_idea(decision_id="trade-20350612-bad-needs-changes")
    service.propose(bad, actor_id="idea-generator-v1")
    service.request_changes(bad.decision_id, actor_id="rj", reason="Tighten invalidation")
    unaudited_revision = build_trade_idea(
        decision_id=bad.decision_id,
        invalidation="Daily close below 59000",
    )
    TradeIdeaStore(root / "records").save(unaudited_revision)
    payload = _idea_payload(decision_id="trade-20350612-clean-preview")
    path = _write_idea(tmp_path / "clean-preview.json", payload)

    exit_code, response = _run_json(
        capsys,
        [
            "ideas",
            "propose",
            *_root_args(root),
            "--actor",
            "idea-generator-v1",
            "--file",
            str(path),
        ],
    )

    assert exit_code == 0
    assert response["success"] is True
    assert response["data"]["decision_id"] == payload["decision_id"]
