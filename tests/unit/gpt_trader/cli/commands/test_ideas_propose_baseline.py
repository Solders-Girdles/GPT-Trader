from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from gpt_trader import cli
from gpt_trader.cli.response import CliErrorCode

AS_OF = datetime(2035, 6, 12, 0, 0, tzinfo=UTC)
GOLDEN_CROSS = ["100"] * 50 + ["102", "104", "106"]
FLAT = ["100"] * 53


def _run_json(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, dict[str, Any]]:
    exit_code = cli.main(argv)
    output = capsys.readouterr().out
    assert output
    return exit_code, json.loads(output)


def _root_args(root: Path) -> list[str]:
    return ["--ideas-root", str(root), "--format", "json"]


def _series_payload(
    closes: list[str],
    *,
    symbol: str = "BTC-USD",
    as_of: datetime = AS_OF,
    last_volume: str = "5000",
) -> dict[str, Any]:
    candles: list[dict[str, str]] = []
    for index, close in enumerate(closes):
        timestamp = as_of - timedelta(days=len(closes) - index)
        candles.append(
            {
                "ts": timestamp.isoformat(),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": last_volume if index == len(closes) - 1 else "1000",
            }
        )
    return {"symbol": symbol, "granularity": "1d", "candles": candles}


def _snapshot_payload(
    closes: list[str] = GOLDEN_CROSS,
    *,
    as_of: datetime = AS_OF,
    source: str = "local-fixture:coinbase-candles",
) -> dict[str, Any]:
    return {
        "as_of": as_of.isoformat(),
        "source": source,
        "series": [_series_payload(closes, as_of=as_of)],
    }


def _write_snapshot(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _propose_baseline(
    capsys: pytest.CaptureFixture[str],
    root: Path,
    snapshot_path: Path,
) -> tuple[int, dict[str, Any]]:
    return _run_json(
        capsys,
        [
            "ideas",
            "propose-baseline",
            *_root_args(root),
            "--snapshot",
            str(snapshot_path),
        ],
    )


def test_propose_baseline_persists_generated_proposal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ideas"
    snapshot_path = _write_snapshot(tmp_path / "snapshot.json", _snapshot_payload())

    exit_code, response = _propose_baseline(capsys, root, snapshot_path)

    assert exit_code == 0
    assert response["success"] is True
    assert response["command"] == "ideas propose-baseline"
    assert response["data"]["proposer_id"] == "baseline-ma-10-50"
    assert response["data"]["proposal_count"] == 1
    proposal = response["data"]["proposed"][0]
    assert proposal["decision_id"].startswith("trade-20350612-btcusd-")
    assert proposal["state"] == "proposed"
    assert proposal["record_hash"]
    assert proposal["approval_preview"] == {"violations": [], "warnings": []}
    assert (root / "records" / proposal["decision_id"] / "latest.json").exists()
    event = json.loads((root / "audit.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert event["actor_type"] == "ai"
    assert event["actor_id"] == "baseline-ma-10-50"
    assert "proposer_id=baseline-ma-10-50" in event["evidence"]


def test_propose_baseline_no_signal_is_noop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ideas"
    snapshot_path = _write_snapshot(tmp_path / "flat.json", _snapshot_payload(FLAT))

    exit_code, response = _propose_baseline(capsys, root, snapshot_path)

    assert exit_code == 0
    assert response["data"]["proposal_count"] == 0
    assert response["data"]["proposed"] == []
    assert response["metadata"]["was_noop"] is True
    assert not (root / "records").exists()
    assert not (root / "audit.jsonl").exists()


def test_propose_baseline_duplicate_decision_fails_without_extra_audit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ideas"
    snapshot_path = _write_snapshot(tmp_path / "snapshot.json", _snapshot_payload())
    first_exit_code, first_response = _propose_baseline(capsys, root, snapshot_path)
    assert first_exit_code == 0
    original_audit = (root / "audit.jsonl").read_text(encoding="utf-8")

    exit_code, response = _propose_baseline(capsys, root, snapshot_path)

    assert exit_code == 1
    assert response["errors"][0]["code"] == CliErrorCode.VALIDATION_ERROR.value
    assert response["errors"][0]["details"]["field"] == "decision_id"
    assert (root / "audit.jsonl").read_text(encoding="utf-8") == original_audit
    decision_id = first_response["data"]["proposed"][0]["decision_id"]
    latest = json.loads((root / "records" / decision_id / "latest.json").read_text())
    assert latest["decision_id"] == decision_id


def test_propose_baseline_malformed_snapshot_returns_invalid_argument(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ideas"
    snapshot_path = _write_snapshot(
        tmp_path / "malformed.json",
        {"as_of": AS_OF.isoformat(), "source": "local-fixture:coinbase-candles"},
    )

    exit_code, response = _propose_baseline(capsys, root, snapshot_path)

    assert exit_code == 1
    assert response["errors"][0]["code"] == CliErrorCode.INVALID_ARGUMENT.value
    assert response["errors"][0]["details"]["field"] == "series"
    assert not (root / "records").exists()
    assert not (root / "audit.jsonl").exists()


def test_propose_baseline_json_output_includes_approval_preview_warnings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ideas"
    historical_as_of = datetime(2020, 6, 12, 0, 0, tzinfo=UTC)
    snapshot_path = _write_snapshot(
        tmp_path / "historical.json",
        _snapshot_payload(as_of=historical_as_of),
    )

    exit_code, response = _propose_baseline(capsys, root, snapshot_path)

    assert exit_code == 0
    proposal = response["data"]["proposed"][0]
    assert proposal["decision_id"].startswith("trade-20200612-btcusd-")
    assert proposal["record_hash"]
    assert proposal["state"] == "proposed"
    assert any("expired" in item for item in proposal["approval_preview"]["violations"])
    assert any("would fail approval" in item for item in proposal["approval_preview"]["warnings"])
    assert response["warnings"] == proposal["approval_preview"]["warnings"]
