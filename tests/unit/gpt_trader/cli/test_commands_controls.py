"""Unit tests for the controls summary CLI command."""

from __future__ import annotations

from types import SimpleNamespace

from scripts.ops import controls_smoke

from gpt_trader.cli.commands import controls as controls_cmd


def _make_smoke_result(
    name: str, status: str, detail: dict[str, str]
) -> controls_smoke.SmokeResult:
    """Helper to build deterministic smoke results."""

    return controls_smoke.SmokeResult(name=name, status=status, detail=detail)


def test_handle_summary_success(monkeypatch) -> None:
    results = [
        _make_smoke_result("guard_one", "passed", {"status": "blocked"}),
        _make_smoke_result("guard_two", "passed", {"status": "success"}),
    ]

    monkeypatch.setattr(controls_smoke, "run_smoke_checks", lambda: results)

    args = SimpleNamespace(max_top_failures=3)
    response = controls_cmd._handle_summary(args)

    assert response.exit_code == controls_smoke.EXIT_SUCCESS
    assert response.warnings == []
    summary = response.data["summary"]["severity_counts"]
    assert summary == {"pass": 2, "warn": 0, "fail": 0}
    assert response.data["summary"]["total_failing_checks"] == 0


def test_handle_summary_warns_on_guard_blocks(monkeypatch) -> None:
    results = [
        _make_smoke_result("guard_block", "failed", {"status": "blocked"}),
    ]

    monkeypatch.setattr(controls_smoke, "run_smoke_checks", lambda: results)

    args = SimpleNamespace(max_top_failures=3)
    response = controls_cmd._handle_summary(args)

    assert response.exit_code == controls_smoke.EXIT_GUARD_BLOCKED
    assert response.warnings == ["1 controls were guard-blocked (guard checks still functional)"]
    assert response.data["summary"]["severity_counts"]["warn"] == 1


def test_handle_summary_respects_max_failures(monkeypatch) -> None:
    results = [
        _make_smoke_result("a_fail", "failed", {"status": "success"}),
        _make_smoke_result("b_warn", "failed", {"status": "blocked"}),
    ]

    monkeypatch.setattr(controls_smoke, "run_smoke_checks", lambda: results)

    args = SimpleNamespace(max_top_failures=1)
    response = controls_cmd._handle_summary(args)

    summary = response.data["summary"]
    assert summary["max_displayed_failures"] == 1
    assert summary["total_failing_checks"] == 2
    assert summary["truncated"] is True
    assert summary["top_failing_checks"] == [
        {"name": "a_fail", "severity": "fail", "detail": {"status": "success"}}
    ]
