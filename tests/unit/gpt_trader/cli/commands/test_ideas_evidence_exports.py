from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from gpt_trader import cli
from gpt_trader.features.trade_ideas import (
    ActorType,
    CloseoutResolution,
    TimeHorizon,
    TradeIdeaService,
)
from gpt_trader.features.trade_ideas.report import build_trade_idea_track_record_report
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


def _root_args(root: Path, *, output_format: str = "json") -> list[str]:
    return ["--ideas-root", str(root), "--format", output_format]


def _seed_evidence_store(root: Path) -> TradeIdeaService:
    current_time = [datetime(2026, 5, 7, 12, 0, tzinfo=UTC)]
    service = TradeIdeaService(root, now_factory=lambda: current_time[0])

    filled = _idea("trade-evidence-filled")
    service.propose(filled, actor_id="generator-a")
    current_time[0] = datetime(2026, 5, 8, 12, 0, tzinfo=UTC)
    service.approve(filled.decision_id, actor_id="rj", reason="Risk verified")
    service.record_submission(filled.decision_id, actor_id="operator", venue="manual")
    service.record_fill(filled.decision_id, actor_id="operator", venue="manual")
    service.record_closeout_attribution(
        filled.decision_id,
        actor_id="rj",
        resolution=CloseoutResolution.THESIS_TARGET,
        realized_profit_loss_amount=Decimal("125.50"),
        realized_profit_loss_percent=Decimal("2.4"),
        evidence=("statement:manual-123",),
    )

    current_time[0] = datetime(2026, 6, 4, 12, 0, tzinfo=UTC)
    expired = _idea("trade-evidence-expired")
    service.propose(expired, actor_id="generator-b")
    current_time[0] = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)
    service.expire(expired.decision_id, actor_id="expiry-sweep")
    service.record_closeout_attribution(
        expired.decision_id,
        actor_id="expiry-sweep",
        actor_type=ActorType.SYSTEM,
        resolution=CloseoutResolution.EXPIRY,
        realized_profit_loss_unavailable_reason="Expired before entry",
    )
    return service


def test_report_json_artifact_schema_date_window_and_output_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "ideas"
    _seed_evidence_store(root)
    output_dir = tmp_path / "artifacts"

    exit_code, response = _run_json(
        capsys,
        [
            "ideas",
            "report",
            *_root_args(root),
            "--from",
            "2026-05-01",
            "--to",
            "2026-05-31T23:59:59Z",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert exit_code == 0
    data = response["data"]
    assert data["schema_version"] == "gpt-trader.trade_ideas.report.v1"
    assert data["quality_report_id"].startswith("tir-")
    assert data["generated_at"]
    assert data["filters"] == {
        "since": "2026-05-01T00:00:00+00:00",
        "until": "2026-05-31T23:59:59+00:00",
    }
    assert data["row_count"] == 1
    assert data["proposal_volume"]["idea_count"] == 1
    assert data["source"] == {
        "audit_event_count": 4,
        "closeout_count": 1,
        "idea_count": 1,
    }
    artifact_path = Path(data["artifact_path"])
    assert artifact_path.parent == output_dir
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact_payload["quality_report_id"] == data["quality_report_id"]


def test_report_id_is_deterministic_for_seeded_fixture(tmp_path: Path) -> None:
    root = tmp_path / "ideas"
    service = _seed_evidence_store(root)
    fixed_now = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)

    first = build_trade_idea_track_record_report(service, now=fixed_now)
    second = build_trade_idea_track_record_report(service, now=fixed_now)

    assert first == second
    assert first["quality_report_id"].startswith("tir-")


def test_report_csv_output_contains_flat_metric_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "ideas"
    _seed_evidence_store(root)

    exit_code, output = _run_text(
        capsys,
        ["ideas", "report", *_root_args(root, output_format="csv")],
    )

    assert exit_code == 0
    rows = list(csv.DictReader(io.StringIO(output)))
    assert rows[0].keys() == {
        "quality_report_id",
        "schema_version",
        "metric_path",
        "value",
    }
    assert any(
        row["metric_path"] == "proposal_volume.idea_count" and row["value"] == "2" for row in rows
    )


def test_audit_list_filters_paginates_and_exports_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "ideas"
    _seed_evidence_store(root)

    exit_code, response = _run_json(
        capsys,
        [
            "ideas",
            "audit",
            "list",
            *_root_args(root),
            "--actor",
            "rj",
            "--action",
            "approved",
            "--state",
            "approved",
            "--limit",
            "1",
            "--offset",
            "0",
        ],
    )

    assert exit_code == 0
    data = response["data"]
    assert data["schema_version"] == "gpt-trader.trade_ideas.audit_export.v1"
    assert data["pagination"] == {
        "total_count": 1,
        "returned_count": 1,
        "limit": 1,
        "offset": 0,
        "next_offset": None,
    }
    assert data["events"][0]["decision_id"] == "trade-evidence-filled"
    assert data["events"][0]["actor_id"] == "rj"

    exit_code, csv_output = _run_text(
        capsys,
        [
            "ideas",
            "audit",
            "export",
            *_root_args(root, output_format="csv"),
            "--decision-id",
            "trade-evidence-filled",
            "--action",
            "filled",
        ],
    )

    assert exit_code == 0
    rows = list(csv.DictReader(io.StringIO(csv_output)))
    assert [row["action"] for row in rows] == ["filled"]
    assert rows[0]["decision_id"] == "trade-evidence-filled"


def test_closeout_list_filters_joins_terminal_event_and_exports_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "ideas"
    _seed_evidence_store(root)

    exit_code, response = _run_json(
        capsys,
        [
            "ideas",
            "closeout",
            "list",
            *_root_args(root),
            "--actor",
            "rj",
            "--resolution",
            "thesis_target",
            "--has-evidence",
            "true",
        ],
    )

    assert exit_code == 0
    data = response["data"]
    assert data["schema_version"] == "gpt-trader.trade_ideas.closeout_export.v1"
    assert data["pagination"]["total_count"] == 1
    closeout = data["closeouts"][0]
    assert closeout["decision_id"] == "trade-evidence-filled"
    assert closeout["terminal_action"] == "filled"
    assert closeout["terminal_state"] == "filled"

    exit_code, csv_output = _run_text(
        capsys,
        [
            "ideas",
            "closeout",
            "export",
            *_root_args(root, output_format="csv"),
            "--resolution",
            "expiry",
        ],
    )

    assert exit_code == 0
    rows = list(csv.DictReader(io.StringIO(csv_output)))
    assert [row["decision_id"] for row in rows] == ["trade-evidence-expired"]
    assert rows[0]["terminal_state"] == "expired"
    assert rows[0]["realized_profit_loss_unavailable_reason"] == "Expired before entry"


def test_audit_and_closeout_no_match_results_are_successful_noops(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "ideas"
    _seed_evidence_store(root)

    exit_code, audit_response = _run_json(
        capsys,
        ["ideas", "audit", "list", *_root_args(root), "--actor", "nobody"],
    )
    assert exit_code == 0
    assert audit_response["metadata"]["was_noop"] is True
    assert audit_response["data"]["events"] == []
    assert audit_response["data"]["pagination"]["total_count"] == 0

    exit_code, closeout_response = _run_json(
        capsys,
        [
            "ideas",
            "closeout",
            "export",
            *_root_args(root),
            "--decision-id",
            "trade-no-match",
        ],
    )
    assert exit_code == 0
    assert closeout_response["metadata"]["was_noop"] is True
    assert closeout_response["data"]["row_count"] == 0
    assert closeout_response["data"]["rows"] == []
