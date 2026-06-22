from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from gpt_trader import cli
from gpt_trader.cli.response import CliErrorCode
from gpt_trader.features.trade_ideas import TimeHorizon
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


def _propose(
    capsys: pytest.CaptureFixture[str],
    root: Path,
    payload: dict[str, Any],
    *,
    filename: str = "idea.json",
) -> dict[str, Any]:
    path = _write_idea(root.parent / filename, payload)
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
    return response


def test_propose_duplicate_rejects_without_overwriting_record_or_audit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ideas"
    payload = _idea_payload(decision_id="trade-20350612-duplicate")
    first_response = _propose(capsys, root, payload)
    decision_dir = root / "records" / payload["decision_id"]
    latest_path = decision_dir / "latest.json"
    audit_path = root / "audit.jsonl"
    original_latest = latest_path.read_text(encoding="utf-8")
    original_audit = audit_path.read_text(encoding="utf-8")
    original_hash = first_response["data"]["record_hash"]
    revised_payload = {**payload, "thesis": "Edited thesis that must not persist"}
    revised_path = _write_idea(tmp_path / "duplicate-edited.json", revised_payload)

    exit_code, response = _run_json(
        capsys,
        [
            "ideas",
            "propose",
            *_root_args(root),
            "--actor",
            "idea-generator-v1",
            "--file",
            str(revised_path),
        ],
    )

    assert exit_code == 1
    assert response["errors"][0]["code"] == CliErrorCode.VALIDATION_ERROR.value
    assert response["errors"][0]["details"]["field"] == "decision_id"
    assert latest_path.read_text(encoding="utf-8") == original_latest
    assert audit_path.read_text(encoding="utf-8") == original_audit
    assert json.loads(latest_path.read_text(encoding="utf-8"))["thesis"] == payload["thesis"]
    audit_lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(audit_lines) == 1
    assert json.loads(audit_lines[0])["record_hash"] == original_hash


def test_propose_timezone_naive_expiry_returns_invalid_argument_without_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ideas"
    payload = _idea_payload(decision_id="trade-20350612-naive-expiry")
    payload["time_horizon"]["expires_at"] = "2035-06-19T16:00:00"
    path = _write_idea(tmp_path / "naive-expiry.json", payload)

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

    assert exit_code == 1
    assert response["errors"][0]["code"] == CliErrorCode.INVALID_ARGUMENT.value
    assert "time_horizon.expires_at must include a timezone" in response["errors"][0]["message"]
    assert not (root / "records" / payload["decision_id"]).exists()
    assert not (root / "audit.jsonl").exists()


def test_propose_rejects_path_traversal_decision_id_without_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ideas"
    payload = _idea_payload(decision_id="trade-20350612-path")
    payload["decision_id"] = "../outside-record"
    path = _write_idea(tmp_path / "path-traversal.json", payload)

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

    assert exit_code == 1
    assert response["errors"][0]["code"] == CliErrorCode.INVALID_ARGUMENT.value
    assert "decision_id must be a safe path segment" in response["errors"][0]["message"]
    assert not (root / "records").exists()
    assert not (tmp_path / "outside-record").exists()
    assert not (root / "audit.jsonl").exists()


def test_propose_rejects_non_finite_preview_numbers_without_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ideas"
    payload = _idea_payload(decision_id="trade-20350612-nan-loss")
    payload["max_loss"]["percent_of_account"] = "NaN"
    path = _write_idea(tmp_path / "nan-loss.json", payload)

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

    assert exit_code == 1
    assert response["errors"][0]["code"] == CliErrorCode.INVALID_ARGUMENT.value
    assert "max_loss.percent_of_account must be finite" in response["errors"][0]["message"]
    assert not (root / "records" / payload["decision_id"]).exists()
    assert not (root / "audit.jsonl").exists()


def test_propose_malformed_nested_section_returns_invalid_argument_without_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ideas"
    payload = _idea_payload(decision_id="trade-20350612-bad-entry-zone")
    payload["entry_zone"] = []
    path = _write_idea(tmp_path / "bad-entry-zone.json", payload)

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

    assert exit_code == 1
    assert response["errors"][0]["code"] == CliErrorCode.INVALID_ARGUMENT.value
    assert "entry_zone must be a JSON object" in response["errors"][0]["message"]
    assert not (root / "records" / payload["decision_id"]).exists()
    assert not (root / "audit.jsonl").exists()


def test_propose_preview_budget_failure_happens_before_record_or_audit_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ideas"
    root.mkdir()
    (root / "risk_budget.jsonl").write_text("{malformed budget json}\n", encoding="utf-8")
    payload = _idea_payload(decision_id="trade-20350612-bad-budget-preview")
    path = _write_idea(tmp_path / "bad-budget-preview.json", payload)

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

    assert exit_code == 1
    assert response["errors"][0]["code"] == CliErrorCode.OPERATION_FAILED.value
    assert not (root / "records" / payload["decision_id"]).exists()
    assert not (root / "audit.jsonl").exists()
