from __future__ import annotations

import pytest

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


@pytest.mark.parametrize(
    "results,expected_counts,expected_first_failure",
    [
        (
            [
                controls_smoke.SmokeResult(
                    name="guard_success",
                    status="passed",
                    detail={"status": "success"},
                )
            ],
            {"pass": 1, "warn": 0, "fail": 0},
            None,
        ),
        (
            [
                controls_smoke.SmokeResult(
                    name="guard_warn",
                    status="failed",
                    detail={"status": "blocked"},
                )
            ],
            {"pass": 0, "warn": 1, "fail": 0},
            "warn",
        ),
        (
            [
                controls_smoke.SmokeResult(
                    name="guard_fail",
                    status="failed",
                    detail={"status": "error"},
                )
            ],
            {"pass": 0, "warn": 0, "fail": 1},
            "fail",
        ),
    ],
)
def test_summarize_smoke_results_is_deterministic_across_severity_levels(
    results,
    expected_counts,
    expected_first_failure,
) -> None:
    summary = controls_smoke.summarize_smoke_results(results, max_top_failures=3)

    assert summary.severity_counts == expected_counts
    if expected_first_failure is None:
        assert summary.top_failing_checks == []
    else:
        assert summary.top_failing_checks[0]["severity"] == expected_first_failure


def test_summarize_smoke_results_orders_failures_by_severity_then_name() -> None:
    results = [
        controls_smoke.SmokeResult(
            name="warn_beta",
            status="failed",
            detail={"status": "blocked"},
        ),
        controls_smoke.SmokeResult(
            name="fail_zeta",
            status="failed",
            detail={"status": "error"},
        ),
        controls_smoke.SmokeResult(
            name="warn_alpha",
            status="failed",
            detail={"status": "blocked"},
        ),
        controls_smoke.SmokeResult(
            name="fail_alpha",
            status="failed",
            detail={"status": "error"},
        ),
    ]

    summary = controls_smoke.summarize_smoke_results(results, max_top_failures=10)

    assert [entry["name"] for entry in summary.top_failing_checks] == [
        "fail_alpha",
        "fail_zeta",
        "warn_alpha",
        "warn_beta",
    ]


def test_build_summary_payload_validates_schema_and_includes_required_fields() -> None:
    results = [
        controls_smoke.SmokeResult(
            name="guard_pass",
            status="passed",
            detail={"status": "success"},
        ),
        controls_smoke.SmokeResult(
            name="guard_fail",
            status="failed",
            detail={"status": "error"},
        ),
    ]
    outcome = controls_smoke.ExitOutcome("runtime_failure", controls_smoke.EXIT_RUNTIME_FAILURE)

    payload = controls_smoke.build_summary_payload(results, outcome)

    controls_smoke.validate_summary_payload(payload)
    assert payload["summary"]["severity_counts"]["fail"] == 1
    assert (
        payload["summary"]["max_displayed_failures"] == controls_smoke.DEFAULT_SUMMARY_MAX_FAILURES
    )


def test_validate_summary_payload_rejects_missing_fields() -> None:
    results = [
        controls_smoke.SmokeResult(
            name="guard_fail",
            status="failed",
            detail={"status": "error"},
        )
    ]
    outcome = controls_smoke.ExitOutcome("runtime_failure", controls_smoke.EXIT_RUNTIME_FAILURE)
    payload = controls_smoke.build_summary_payload(results, outcome)

    missing_payload = dict(payload)
    missing_payload.pop("summary_version")

    with pytest.raises(ValueError, match="missing summary keys"):
        controls_smoke.validate_summary_payload(missing_payload)
