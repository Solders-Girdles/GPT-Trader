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

    outcome = controls_smoke._determine_outcome(results)

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

    outcome = controls_smoke._determine_outcome(results)

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

    outcome = controls_smoke._determine_outcome(results)

    assert outcome.label == "runtime_failure"
    assert outcome.exit_code == controls_smoke.EXIT_RUNTIME_FAILURE
