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


def test_main_redacts_sensitive_text_output_reason(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    async def fake_run_probe(*_: object, **__: object) -> tuple[str, str | None, str | None]:
        return "blocked", "decision-1", "token=secret-token"

    monkeypatch.setattr(decision_trace_probe, "_run_probe", fake_run_probe)
    monkeypatch.setattr(decision_trace_probe, "_read_latest_trace", lambda _: (None, None))

    args = argparse.Namespace(
        profile="canary",
        symbol="BTC-USD",
        side="buy",
        runtime_root=tmp_path,
        json=False,
    )
    monkeypatch.setattr(decision_trace_probe, "_parse_args", lambda: args)

    assert decision_trace_probe.main() == 0
    output = capsys.readouterr().out

    assert "blocked_reason=token=[REDACTED]" in output
    assert "secret-token" not in output


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


def test_json_payload_has_deterministic_allowlisted_shape() -> None:
    payload = decision_trace_probe._build_json_payload(
        status="blocked",
        decision_id="decision-1",
        reason="risk limit exceeded",
        latest_id=7,
        latest_ts="not-a-timestamp",
    )

    assert list(payload) == [
        "status",
        "decision_id",
        "blocked_reason",
        "latest_trace",
    ]
    latest_trace = payload["latest_trace"]
    assert isinstance(latest_trace, dict)
    assert list(latest_trace) == ["found", "id", "timestamp"]
    assert payload == {
        "status": "blocked",
        "decision_id": "decision-1",
        "blocked_reason": "risk limit exceeded",
        "latest_trace": {
            "found": True,
            "id": 7,
            "timestamp": "not-a-timestamp",
        },
    }


def test_json_payload_handles_missing_trace_metadata() -> None:
    payload = decision_trace_probe._build_json_payload(
        status="blocked",
        decision_id=None,
        reason=None,
        latest_id=None,
        latest_ts=None,
    )

    assert payload == {
        "status": "blocked",
        "decision_id": None,
        "blocked_reason": None,
        "latest_trace": {
            "found": False,
            "id": None,
            "timestamp": None,
        },
    }


def test_json_payload_redacts_sensitive_reason_fields() -> None:
    payload = decision_trace_probe._build_json_payload(
        status="blocked",
        decision_id="decision-1",
        reason="api_key=sk-live secret='abc123' token: bearer-value password hunter2",
        latest_id=None,
        latest_ts=None,
    )

    assert payload["blocked_reason"] == (
        "api_key=[REDACTED] secret=[REDACTED] token=[REDACTED] password [REDACTED]"
    )
    assert "sk-live" not in json.dumps(payload)
    assert "abc123" not in json.dumps(payload)
    assert "bearer-value" not in json.dumps(payload)
    assert "hunter2" not in json.dumps(payload)


def test_json_payload_redacts_quoted_sensitive_reason_fields() -> None:
    payload = decision_trace_probe._build_json_payload(
        status="blocked",
        decision_id="decision-1",
        reason='{ "api_key": "sk-live", "nested": {"password": "hunter2"} }',
        latest_id=None,
        latest_ts=None,
    )

    rendered = json.dumps(payload)
    blocked_reason = payload["blocked_reason"]
    assert isinstance(blocked_reason, str)
    assert "sk-live" not in rendered
    assert "hunter2" not in rendered
    assert "api_key=[REDACTED]" in blocked_reason
    assert "password=[REDACTED]" in blocked_reason


def test_json_error_output_redacts_sensitive_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = decision_trace_probe._emit_error(
        "probe failed: password=secret-token",
        json_output=True,
        exit_code=decision_trace_probe.EXIT_PROBE_ERROR,
    )

    assert exit_code == decision_trace_probe.EXIT_PROBE_ERROR
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "error",
        "error": "probe failed: password=[REDACTED]",
    }
