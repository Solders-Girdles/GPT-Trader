"""Tests for preflight report generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gpt_trader.preflight.core import PreflightCheck
from gpt_trader.preflight.report import format_preflight_report, generate_report


@pytest.fixture
def report_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _read_report(tmp_path: Path) -> dict[str, Any]:
    paths = sorted(tmp_path.glob("preflight_report_*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


class TestGenerateReport:
    """Test report generation."""

    def test_returns_ready_status_with_no_errors_few_warnings(
        self, report_cwd: Path, capsys: pytest.CaptureFixture
    ) -> None:
        checker = PreflightCheck()
        checker.context.successes.extend(["Success 1", "Success 2", "Success 3"])
        checker.context.warnings.extend(["Warning 1", "Warning 2"])  # <= 3 warnings

        success, status = generate_report(checker)

        assert success is True
        assert status == "READY"

        captured = capsys.readouterr()
        assert "READY" in captured.out
        assert "System is READY for production trading" in captured.out

    def test_returns_review_status_with_many_warnings(
        self, report_cwd: Path, capsys: pytest.CaptureFixture
    ) -> None:
        checker = PreflightCheck()
        checker.context.successes.append("Success 1")
        checker.context.warnings.extend(["Warning 1", "Warning 2", "Warning 3", "Warning 4"])

        success, status = generate_report(checker)

        assert success is True  # Still "success" (no errors)
        assert status == "REVIEW"

        captured = capsys.readouterr()
        assert "REVIEW" in captured.out
        assert "review before proceeding" in captured.out

    def test_returns_not_ready_status_with_errors(
        self, report_cwd: Path, capsys: pytest.CaptureFixture
    ) -> None:
        checker = PreflightCheck()
        checker.context.successes.append("Success 1")
        checker.context.errors.append("Critical error")

        success, status = generate_report(checker)

        assert success is False
        assert status == "NOT READY"

        captured = capsys.readouterr()
        assert "NOT READY" in captured.out
        assert "critical issues must be resolved" in captured.out

    def test_prints_summary_counts(self, report_cwd: Path, capsys: pytest.CaptureFixture) -> None:
        checker = PreflightCheck()
        checker.context.successes.extend(["S1", "S2", "S3"])
        checker.context.warnings.extend(["W1", "W2"])
        checker.context.errors.append("E1")

        generate_report(checker)

        captured = capsys.readouterr()
        assert "Passed: 3" in captured.out
        assert "Warnings: 2" in captured.out
        assert "Failed: 1" in captured.out

    def test_prints_recommendations_for_ready(
        self, report_cwd: Path, capsys: pytest.CaptureFixture
    ) -> None:
        checker = PreflightCheck(profile="canary")
        checker.context.successes.append("Success")

        generate_report(checker)

        captured = capsys.readouterr()
        assert "--dry-run" in captured.out
        assert "tiny positions" in captured.out

    def test_prints_recommendations_for_not_ready(
        self, report_cwd: Path, capsys: pytest.CaptureFixture
    ) -> None:
        checker = PreflightCheck()
        checker.context.errors.append("Error")

        generate_report(checker)

        captured = capsys.readouterr()
        assert "Fix all critical errors" in captured.out
        assert "Run tests:" in captured.out

    def test_saves_json_report(self, report_cwd: Path) -> None:
        checker = PreflightCheck(profile="prod")
        checker.context.successes.extend(["S1", "S2"])
        checker.context.warnings.append("W1")

        generate_report(checker)

        report_data = _read_report(report_cwd)
        assert report_data["profile"] == "prod"
        assert report_data["status"] == "READY"
        assert report_data["successes"] == 2
        assert report_data["warnings"] == 1
        assert report_data["errors"] == 0
        assert report_data["details"]["successes"] == ["S1", "S2"]
        assert report_data["details"]["warnings"] == ["W1"]
        assert "timestamp" in report_data

    def test_handles_file_save_error(
        self, report_cwd: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        checker = PreflightCheck()
        checker.context.successes.append("Success")

        def deny_write(*_args: object, **_kwargs: object) -> Any:
            raise PermissionError("Cannot write")

        monkeypatch.setattr("gpt_trader.preflight.report.write_preflight_report", deny_write)

        success, status = generate_report(checker)

        assert success is True
        assert status == "READY"

        captured = capsys.readouterr()
        assert "Could not save report" in captured.out
        assert not list(report_cwd.glob("preflight_report_*.json"))

    def test_prints_section_header(self, report_cwd: Path, capsys: pytest.CaptureFixture) -> None:
        checker = PreflightCheck()
        checker.context.successes.append("Success")

        generate_report(checker)

        captured = capsys.readouterr()
        assert "PREFLIGHT REPORT" in captured.out


class TestReportCalculations:
    """Test report calculations."""

    def test_total_checks_calculated_correctly(self, report_cwd: Path) -> None:
        checker = PreflightCheck()
        checker.context.successes.extend(["S1", "S2", "S3"])
        checker.context.warnings.extend(["W1", "W2"])
        checker.context.errors.append("E1")

        generate_report(checker)

        report_data = _read_report(report_cwd)
        assert report_data["total_checks"] == 6  # 3 + 2 + 1


class TestReportFormatting:
    """Test pure report formatting utilities."""

    def test_format_preflight_report_is_pure(self, report_cwd: Path) -> None:
        checker = PreflightCheck(profile="prod")
        checker.context.successes.extend(["S1", "S2"])
        checker.context.warnings.append("W1")
        timestamp = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        report_data = format_preflight_report(checker, timestamp=timestamp)

        assert report_data["timestamp"] == timestamp.isoformat()
        assert report_data["profile"] == "prod"
        assert report_data["status"] == "READY"
        assert report_data["details"]["successes"] == ["S1", "S2"]
        assert report_data["details"]["warnings"] == ["W1"]
        assert not list(report_cwd.glob("preflight_report_*.json"))
