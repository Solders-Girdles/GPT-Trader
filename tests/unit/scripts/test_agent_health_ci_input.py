from __future__ import annotations

import os
from pathlib import Path

from scripts.agents import health_report

from gpt_trader.config.types import Profile


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


def test_run_preflight_dev_profile_skips_remote_checks_by_default(monkeypatch) -> None:
    monkeypatch.delenv("COINBASE_PREFLIGHT_SKIP_REMOTE", raising=False)
    monkeypatch.delenv("COINBASE_PREFLIGHT_FORCE_REMOTE", raising=False)

    class FakePreflightCheck:
        def __init__(self, *, verbose: bool, profile: str) -> None:
            self.verbose = verbose
            self.profile = profile
            self.errors: list[str] = []
            self.warnings: list[str] = []
            self.successes: list[str] = []

        def __getattr__(self, name: str):
            if name == "check_api_connectivity":

                def check_api_connectivity() -> bool:
                    assert os.environ.get("COINBASE_PREFLIGHT_SKIP_REMOTE") == "1"
                    self.successes.append(name)
                    return True

                return check_api_connectivity
            if name.startswith("check_") or name == "simulate_dry_run":

                def check() -> bool:
                    self.successes.append(name)
                    return True

                return check
            raise AttributeError(name)

        def log_error(self, message: str) -> None:
            self.errors.append(message)

    monkeypatch.setattr(health_report, "PreflightCheck", FakePreflightCheck)

    result = health_report._run_preflight(Profile.DEV, verbose=False, warn_only=False)

    assert result.status == "passed"
    assert os.environ.get("COINBASE_PREFLIGHT_SKIP_REMOTE") is None


def test_run_preflight_prod_profile_keeps_remote_checks_strict(monkeypatch) -> None:
    monkeypatch.delenv("COINBASE_PREFLIGHT_SKIP_REMOTE", raising=False)
    monkeypatch.delenv("COINBASE_PREFLIGHT_FORCE_REMOTE", raising=False)

    class FakePreflightCheck:
        def __init__(self, *, verbose: bool, profile: str) -> None:
            self.verbose = verbose
            self.profile = profile
            self.errors: list[str] = []
            self.warnings: list[str] = []
            self.successes: list[str] = []

        def __getattr__(self, name: str):
            if name == "check_api_connectivity":

                def check_api_connectivity() -> bool:
                    assert os.environ.get("COINBASE_PREFLIGHT_SKIP_REMOTE") is None
                    self.successes.append(name)
                    return True

                return check_api_connectivity
            if name.startswith("check_") or name == "simulate_dry_run":

                def check() -> bool:
                    self.successes.append(name)
                    return True

                return check
            raise AttributeError(name)

        def log_error(self, message: str) -> None:
            self.errors.append(message)

    monkeypatch.setattr(health_report, "PreflightCheck", FakePreflightCheck)

    result = health_report._run_preflight(Profile.PROD, verbose=False, warn_only=False)

    assert result.status == "passed"
    assert os.environ.get("COINBASE_PREFLIGHT_SKIP_REMOTE") is None
