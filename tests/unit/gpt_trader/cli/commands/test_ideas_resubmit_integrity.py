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
) -> None:
    path = _write_idea(root.parent / "idea.json", payload)
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


def _request_changes(capsys: pytest.CaptureFixture[str], root: Path, decision_id: str) -> None:
    exit_code, response = _run_json(
        capsys,
        [
            "ideas",
            "request-changes",
            *_root_args(root),
            "--actor",
            "rj",
            decision_id,
            "--reason",
            "Need tighter risk",
        ],
    )
    assert exit_code == 0
    assert response["success"] is True


def test_resubmit_malformed_nested_section_returns_invalid_argument_without_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ideas"
    payload = _idea_payload(decision_id="trade-20350612-bad-resubmit-section")
    _propose(capsys, root, payload)
    _request_changes(capsys, root, payload["decision_id"])
    latest_path = root / "records" / payload["decision_id"] / "latest.json"
    audit_path = root / "audit.jsonl"
    original_latest = latest_path.read_text(encoding="utf-8")
    original_audit = audit_path.read_text(encoding="utf-8")
    revised = {**payload, "invalidation": "Daily close below 58000"}
    revised["max_loss"] = []
    revised_path = _write_idea(tmp_path / "bad-resubmit-section.json", revised)

    exit_code, response = _run_json(
        capsys,
        [
            "ideas",
            "resubmit",
            *_root_args(root),
            "--actor",
            "idea-generator-v1",
            "--file",
            str(revised_path),
        ],
    )

    assert exit_code == 1
    assert response["errors"][0]["code"] == CliErrorCode.INVALID_ARGUMENT.value
    assert "max_loss must be a JSON object" in response["errors"][0]["message"]
    assert latest_path.read_text(encoding="utf-8") == original_latest
    assert audit_path.read_text(encoding="utf-8") == original_audit


@pytest.mark.parametrize(
    ("field_path", "malformed_value", "message"),
    [
        ("data_used", "coinbase:candles:BTC-USD", "data_used must be a JSON array of strings"),
        ("data_used", 42, "data_used must be a JSON array of strings"),
        ("data_used", ["coinbase:candles:BTC-USD", 42], "data_used[1] must be a string"),
        (
            "do_not_trade_if",
            "FOMC announcement within 24 hours",
            "do_not_trade_if must be a JSON array of strings",
        ),
        (
            "do_not_trade_if",
            42,
            "do_not_trade_if must be a JSON array of strings",
        ),
        (
            "do_not_trade_if",
            ["FOMC announcement within 24 hours", 42],
            "do_not_trade_if[1] must be a string",
        ),
        (
            "max_loss.assumptions",
            "No slippage beyond 10 bps",
            "max_loss.assumptions must be a JSON array of strings",
        ),
        (
            "max_loss.assumptions",
            42,
            "max_loss.assumptions must be a JSON array of strings",
        ),
        (
            "max_loss.assumptions",
            ["No slippage beyond 10 bps", 42],
            "max_loss.assumptions[1] must be a string",
        ),
    ],
)
def test_resubmit_rejects_malformed_string_sequences_without_writes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field_path: str,
    malformed_value: object,
    message: str,
) -> None:
    root = tmp_path / "ideas"
    payload = _idea_payload(
        decision_id=f"trade-20350612-bad-resubmit-sequence-{field_path.replace('.', '-')}"
    )
    _propose(capsys, root, payload)
    _request_changes(capsys, root, payload["decision_id"])
    latest_path = root / "records" / payload["decision_id"] / "latest.json"
    audit_path = root / "audit.jsonl"
    original_latest = latest_path.read_text(encoding="utf-8")
    original_audit = audit_path.read_text(encoding="utf-8")
    revised = {**payload, "invalidation": "Daily close below 58000"}
    if field_path == "max_loss.assumptions":
        revised["max_loss"]["assumptions"] = malformed_value
    else:
        revised[field_path] = malformed_value
    revised_path = _write_idea(
        tmp_path / f"bad-resubmit-sequence-{field_path.replace('.', '-')}.json", revised
    )

    exit_code, response = _run_json(
        capsys,
        [
            "ideas",
            "resubmit",
            *_root_args(root),
            "--actor",
            "idea-generator-v1",
            "--file",
            str(revised_path),
        ],
    )

    assert exit_code == 1
    assert response["errors"][0]["code"] == CliErrorCode.INVALID_ARGUMENT.value
    assert message in response["errors"][0]["message"]
    assert latest_path.read_text(encoding="utf-8") == original_latest
    assert audit_path.read_text(encoding="utf-8") == original_audit


def test_resubmit_preview_budget_failure_happens_before_record_or_audit_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ideas"
    payload = _idea_payload(decision_id="trade-20350612-bad-budget-resubmit")
    _propose(capsys, root, payload)
    _request_changes(capsys, root, payload["decision_id"])
    latest_path = root / "records" / payload["decision_id"] / "latest.json"
    audit_path = root / "audit.jsonl"
    original_latest = latest_path.read_text(encoding="utf-8")
    original_audit = audit_path.read_text(encoding="utf-8")
    (root / "risk_budget.jsonl").write_text("{malformed budget json}\n", encoding="utf-8")
    revised = {**payload, "invalidation": "Daily close below 58000"}
    revised_path = _write_idea(tmp_path / "bad-budget-resubmit.json", revised)

    exit_code, response = _run_json(
        capsys,
        [
            "ideas",
            "resubmit",
            *_root_args(root),
            "--actor",
            "idea-generator-v1",
            "--file",
            str(revised_path),
        ],
    )

    assert exit_code == 1
    assert response["errors"][0]["code"] == CliErrorCode.OPERATION_FAILED.value
    assert latest_path.read_text(encoding="utf-8") == original_latest
    assert audit_path.read_text(encoding="utf-8") == original_audit
