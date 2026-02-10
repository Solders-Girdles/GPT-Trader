from __future__ import annotations

from scripts.ops import controls_smoke


def test_determine_outcome_success() -> None:
    results = [
        controls_smoke.SmokeResult(
            name="kill_switch",
            status="passed",
            detail={"status": "blocked"},
        )
    ]

    outcome = controls_smoke.determine_outcome(results)

    assert outcome.label == "success"
    assert outcome.exit_code == controls_smoke.EXIT_SUCCESS


def test_determine_outcome_guard_blocked() -> None:
    results = [
        controls_smoke.SmokeResult(
            name="kill_switch",
            status="failed",
            detail={"status": "blocked"},
        ),
        controls_smoke.SmokeResult(
            name="reduce_only_exit",
            status="passed",
            detail={"status": "success"},
        ),
    ]

    outcome = controls_smoke.determine_outcome(results)

    assert outcome.label == "guard_blocked"
    assert outcome.exit_code == controls_smoke.EXIT_GUARD_BLOCKED


def test_determine_outcome_runtime_failure_for_unexpected_success() -> None:
    results = [
        controls_smoke.SmokeResult(
            name="reduce_only_exit",
            status="failed",
            detail={"status": "success"},
        )
    ]

    outcome = controls_smoke.determine_outcome(results)

    assert outcome.label == "runtime_failure"
    assert outcome.exit_code == controls_smoke.EXIT_RUNTIME_FAILURE


def test_summarize_smoke_results_captures_severity_counts() -> None:
    results = [
        controls_smoke.SmokeResult(
            name="fail_check",
            status="failed",
            detail={"status": "success"},
        ),
        controls_smoke.SmokeResult(
            name="warn_check",
            status="failed",
            detail={"status": "blocked"},
        ),
        controls_smoke.SmokeResult(
            name="pass_check",
            status="passed",
            detail={"status": "blocked"},
        ),
    ]

    summary = controls_smoke.summarize_smoke_results(results, max_top_failures=5)

    assert summary.severity_counts == {"pass": 1, "warn": 1, "fail": 1}
    assert summary.top_failing_checks[0]["name"] == "fail_check"
    assert summary.top_failing_checks[1]["name"] == "warn_check"
    assert summary.truncated is False


def test_summarize_smoke_results_truncates_when_limit_exceeded() -> None:
    results = [
        controls_smoke.SmokeResult(
            name="fail_check",
            status="failed",
            detail={"status": "success"},
        ),
        controls_smoke.SmokeResult(
            name="warn_check",
            status="failed",
            detail={"status": "blocked"},
        ),
    ]

    summary = controls_smoke.summarize_smoke_results(results, max_top_failures=1)

    assert summary.total_failing_checks == 2
    assert summary.max_displayed_failures == 1
    assert summary.truncated is True
    assert summary.top_failing_checks == [
        {"name": "fail_check", "severity": "fail", "detail": {"status": "success"}}
    ]
