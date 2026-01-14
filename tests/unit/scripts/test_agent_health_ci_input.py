from __future__ import annotations

from pathlib import Path

from scripts.agents import health_report


def _write_file(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


def test_ci_input_parsing_and_merge(tmp_path: Path) -> None:
    pytest_payload = {
        "summary": {"passed": 3, "failed": 1, "errors": 0, "skipped": 2, "total": 6},
        "duration": 1.23,
    }
    pytest_path = _write_file(tmp_path / "pytest_report.json", json_dumps(pytest_payload))

    junit_xml = '<testsuite tests="4" failures="1" errors="0" skipped="1" time="2.5"></testsuite>'
    junit_path = _write_file(tmp_path / "junit.xml", junit_xml)

    pytest_entry = health_report._load_ci_input(pytest_path)
    junit_entry = health_report._load_ci_input(junit_path)

    assert pytest_entry["format"] == "pytest-json"
    assert pytest_entry["summary"]["total"] == 6
    assert pytest_entry["summary"]["passed"] == 3
    assert junit_entry["format"] == "junit-xml"
    assert junit_entry["summary"]["total"] == 4
    assert junit_entry["summary"]["failed"] == 1

    local_result = health_report.HealthCheckResult(
        name="tests",
        status="passed",
        summary="4 passed in 0.5s",
        duration_seconds=0.5,
        details={
            "summary_counts": {
                "passed": 4,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
                "total": 4,
                "duration_seconds": 0.5,
            }
        },
    )

    merged = health_report._build_test_summary(
        results=[local_result],
        ci_inputs=[pytest_entry, junit_entry],
    )

    assert merged is not None
    assert merged["total"] == 14
    assert merged["passed"] == 9
    assert merged["failed"] == 2
    assert merged["skipped"] == 3
    assert merged["errors"] == 0
    assert merged["duration_seconds"] == 4.23


def json_dumps(payload: object) -> str:
    import json

    return json.dumps(payload)
