from __future__ import annotations

import argparse
import json

import pytest

from scripts.ops import decision_trace_probe


def test_main_outputs_json_payload_with_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    async def fake_run_probe(*_: object, **__: object) -> tuple[str, str | None, str | None]:
        return "blocked", "decision-1", "Risk limit\nexceeded"

    monkeypatch.setattr(decision_trace_probe, "_run_probe", fake_run_probe)
    monkeypatch.setattr(
        decision_trace_probe,
        "_read_latest_trace",
        lambda _: (42, "2026-01-01 00:00:00"),
    )

    args = argparse.Namespace(
        profile="canary",
        symbol="BTC-USD",
        side="buy",
        runtime_root=tmp_path,
        json=True,
    )
    monkeypatch.setattr(decision_trace_probe, "_parse_args", lambda: args)

    assert decision_trace_probe.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "blocked"
    assert payload["decision_id"] == "decision-1"
    assert payload["blocked_reason"] == "Risk limit exceeded"
    assert payload["latest_trace"]["found"] is True
    assert payload["latest_trace"]["id"] == 42
    assert payload["latest_trace"]["timestamp"] == "2026-01-01T00:00:00+00:00"


def test_main_outputs_text_by_default(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    async def fake_run_probe(*_: object, **__: object) -> tuple[str, str | None, str | None]:
        return "blocked", "decision-1", "risk limit exceeded"

    monkeypatch.setattr(decision_trace_probe, "_run_probe", fake_run_probe)
    monkeypatch.setattr(
        decision_trace_probe,
        "_read_latest_trace",
        lambda _: (7, "2026-02-01 12:00:00"),
    )

    args = argparse.Namespace(
        profile="canary",
        symbol="BTC-USD",
        side="buy",
        runtime_root=tmp_path,
        json=False,
    )
    monkeypatch.setattr(decision_trace_probe, "_parse_args", lambda: args)

    assert decision_trace_probe.main() == 0
    output = capsys.readouterr().out.strip().splitlines()

    assert output == [
        "status=blocked",
        "decision_id=decision-1",
        "blocked_reason=risk limit exceeded",
        "latest_trace_id=7",
        "latest_trace_ts=2026-02-01T12:00:00+00:00",
    ]


def test_invalid_profile_returns_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(
        profile="nope",
        symbol="BTC-USD",
        side="buy",
        runtime_root=".",
        json=True,
    )
    monkeypatch.setattr(decision_trace_probe, "_parse_args", lambda: args)

    assert decision_trace_probe.main() == decision_trace_probe.EXIT_INVALID_INPUT
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["error"].startswith("invalid profile: nope")
