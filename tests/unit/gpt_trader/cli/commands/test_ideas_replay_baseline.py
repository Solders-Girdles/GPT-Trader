from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from gpt_trader import cli
from gpt_trader.cli.response import CliErrorCode

AS_OF = datetime(2026, 6, 12, 12, 0, tzinfo=UTC)


def _candle(
    offset_hours: int,
    *,
    open_: str = "101",
    high: str = "102",
    low: str = "100",
    close: str = "101",
    volume: str = "1000",
) -> dict[str, str]:
    return {
        "ts": (AS_OF + timedelta(hours=offset_hours)).isoformat(),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def _baseline_hit_fixture() -> dict[str, list[dict[str, str]]]:
    return {
        "candles": [
            _candle(-5, open_="100", high="100", low="100", close="100"),
            _candle(-4, open_="100", high="100", low="100", close="100"),
            _candle(-3, open_="100", high="100", low="100", close="100"),
            _candle(-2, open_="100", high="100", low="100", close="100"),
            _candle(-1, open_="110", high="110", low="110", close="110"),
            _candle(0, open_="110", high="112", low="109", close="111"),
            _candle(1, open_="111", high="126", low="111", close="126"),
        ]
    }


def _flat_fixture() -> dict[str, list[dict[str, str]]]:
    return {
        "candles": [
            _candle(offset, open_="100", high="100", low="100", close="100")
            for offset in range(-5, 2)
        ]
    }


def _write_fixture(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _baseline_args(path: Path, *, output_format: str = "json") -> list[str]:
    return [
        "ideas",
        "replay",
        "baseline",
        "--file",
        str(path),
        "--symbol",
        "BTC-USD",
        "--granularity",
        "ONE_HOUR",
        "--short-window",
        "2",
        "--long-window",
        "4",
        "--crossover-lookback",
        "1",
        "--expiry-hours",
        "3",
        "--format",
        output_format,
    ]


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


def test_replay_baseline_text_success_reports_replay_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _write_fixture(tmp_path / "candles.json", _baseline_hit_fixture())

    exit_code, output = _run_text(capsys, _baseline_args(fixture, output_format="text"))

    assert exit_code == 0
    assert "ideas replay baseline OK (BTC-USD ONE_HOUR, snapshots=2, ideas=1)" in output
    assert "proposer_id: baseline-ma-2-4" in output
    assert "target_hits=1" in output
    assert "average_return_r: 2" in output


def test_replay_baseline_malformed_input_returns_invalid_argument(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _baseline_hit_fixture()
    payload["candles"][0]["open"] = "not-a-decimal"
    fixture = _write_fixture(tmp_path / "bad-candles.json", payload)

    exit_code, response = _run_json(capsys, _baseline_args(fixture))

    assert exit_code == 1
    assert response["errors"][0]["code"] == CliErrorCode.INVALID_ARGUMENT.value
    assert response["errors"][0]["details"]["field"] == "candles[0].open"


def test_replay_baseline_no_idea_replay_is_successful_noop(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _write_fixture(tmp_path / "flat-candles.json", _flat_fixture())

    exit_code, response = _run_json(capsys, _baseline_args(fixture))

    assert exit_code == 0
    assert response["success"] is True
    assert response["metadata"]["was_noop"] is True
    assert response["data"]["snapshots_evaluated"] == 2
    assert response["data"]["ideas_proposed"] == 0
    assert response["data"]["target_hits"] == 0
    assert response["data"]["average_return_r"] is None


def test_replay_baseline_json_output_returns_replay_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _write_fixture(tmp_path / "candles.json", _baseline_hit_fixture())

    exit_code, response = _run_json(capsys, _baseline_args(fixture))

    assert exit_code == 0
    assert response["command"] == "ideas replay baseline"
    data = response["data"]
    assert data["proposer_id"] == "baseline-ma-2-4"
    assert data["snapshots_evaluated"] == 2
    assert data["ideas_proposed"] == 1
    assert data["target_hits"] == 1
    assert data["stop_hits"] == 0
    assert data["timed_out"] == 0
    assert data["not_filled"] == 0
    assert data["no_future_data"] == 0
    assert data["target_hit_rate"] == "1"
    assert data["average_return_r"] == "2"
    assert data["ideas"][0]["outcome"] == "target_hit"
