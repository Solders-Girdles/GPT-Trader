from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from gpt_trader import cli
from gpt_trader.features.trade_ideas import CloseoutResolution, TimeHorizon
from gpt_trader.features.trade_ideas.service import TradeIdeaService
from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea


def _future_horizon() -> TimeHorizon:
    return TimeHorizon(
        expected_hold="3-10 days",
        expires_at=datetime(2035, 6, 19, 16, 0, tzinfo=UTC),
    )


def _idea(decision_id: str, **overrides: Any) -> Any:
    return build_trade_idea(
        decision_id=decision_id,
        time_horizon=_future_horizon(),
        **overrides,
    )


def _root_args(root: Path, *, output_format: str = "json") -> list[str]:
    return ["--ideas-root", str(root), "--format", output_format]


def _run_json(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, dict[str, Any]]:
    exit_code = cli.main(argv)
    output = capsys.readouterr().out
    assert output
    return exit_code, json.loads(output)


def _run_text(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, str]:
    exit_code = cli.main(argv)
    output = capsys.readouterr().out
    assert output
    return exit_code, output


def _seed_closeout_store(root: Path) -> TradeIdeaService:
    current_time = [datetime(2026, 5, 7, 12, 0, tzinfo=UTC)]
    service = TradeIdeaService(root, now_factory=lambda: current_time[0])

    first = _idea("trade-closeout-valid-a")
    service.propose(first, actor_id="generator-a")
    current_time[0] = datetime(2026, 5, 8, 12, 0, tzinfo=UTC)
    service.expire(first.decision_id, actor_id="expiry-sweep")
    service.record_closeout_attribution(
        first.decision_id,
        actor_id="expiry-sweep",
        resolution=CloseoutResolution.EXPIRY,
        realized_profit_loss_unavailable_reason="Expired before entry",
    )

    current_time[0] = datetime(2026, 5, 9, 12, 0, tzinfo=UTC)
    second = _idea("trade-closeout-valid-b")
    service.propose(second, actor_id="generator-b")
    current_time[0] = datetime(2026, 5, 10, 12, 0, tzinfo=UTC)
    service.expire(second.decision_id, actor_id="expiry-sweep")
    service.record_closeout_attribution(
        second.decision_id,
        actor_id="expiry-sweep",
        resolution=CloseoutResolution.EXPIRY,
        realized_profit_loss_unavailable_reason="Expired before entry",
    )
    return service


def _append_orphan_closeout(service: TradeIdeaService) -> None:
    payload = {
        "decision_id": "trade-closeout-orphan",
        "timestamp": "2026-06-06T12:00:00+00:00",
        "actor_type": "human",
        "actor_id": "rj",
        "terminal_event_id": "evt-orphan",
        "record_hash": "hash-orphan",
        "resolution": CloseoutResolution.EXPIRY.value,
        "realized_profit_loss_amount": None,
        "realized_profit_loss_percent": None,
        "realized_profit_loss_unavailable_reason": "Orphaned closeout fixture",
        "max_loss": {
            "amount": "250",
            "percent_of_account": "1.5",
            "assumptions": ["Fixture only"],
        },
        "evidence": ["orphan-evidence"],
    }
    with service.closeout_log.path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def test_closeout_list_and_export_omit_orphaned_closeout_records(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "ideas"
    service = _seed_closeout_store(root)
    _append_orphan_closeout(service)

    exit_code, response = _run_json(
        capsys,
        ["ideas", "closeout", "list", *_root_args(root)],
    )

    assert exit_code == 0
    assert response["data"]["pagination"]["total_count"] == 2
    assert {row["decision_id"] for row in response["data"]["closeouts"]} == {
        "trade-closeout-valid-a",
        "trade-closeout-valid-b",
    }

    exit_code, csv_output = _run_text(
        capsys,
        ["ideas", "closeout", "export", *_root_args(root, output_format="csv")],
    )

    assert exit_code == 0
    rows = list(csv.DictReader(io.StringIO(csv_output)))
    assert {row["decision_id"] for row in rows} == {
        "trade-closeout-valid-a",
        "trade-closeout-valid-b",
    }
