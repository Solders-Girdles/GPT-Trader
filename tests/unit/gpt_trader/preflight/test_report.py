"""Tests for preflight report generation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gpt_trader.preflight.core import PreflightCheck
from gpt_trader.preflight.report import generate_report


class TestGenerateReport:
    """Test report generation."""

    def test_returns_ready_status_with_no_errors_few_warnings(
        self, capsys: pytest.CaptureFixture, tmp_path: Path
    ) -> None:
        """Should return READY status with no errors and few warnings."""
        checker = PreflightCheck()
        checker.context.successes.extend(["Success 1", "Success 2", "Success 3"])
        checker.context.warnings.extend(["Warning 1", "Warning 2"])  # <= 3 warnings

        with patch("gpt_trader.preflight.report.Path") as mock_path:
            mock_path.return_value = tmp_path / "report.json"
            success, status = generate_report(checker)

        assert success is True
        assert status == "READY"

        captured = capsys.readouterr()
        assert "READY" in captured.out
        assert "System is READY for production trading" in captured.out

    def test_returns_review_status_with_many_warnings(
        self, capsys: pytest.CaptureFixture, tmp_path: Path
    ) -> None:
        """Should return REVIEW status with >3 warnings but no errors."""
        checker = PreflightCheck()
        checker.context.successes.extend(["Success 1"])
        checker.context.warnings.extend(["Warning 1", "Warning 2", "Warning 3", "Warning 4"])

        with patch("gpt_trader.preflight.report.Path") as mock_path:
            mock_path.return_value = tmp_path / "report.json"
            success, status = generate_report(checker)

        assert success is True  # Still "success" (no errors)
        assert status == "REVIEW"

        captured = capsys.readouterr()
        assert "REVIEW" in captured.out
        assert "review before proceeding" in captured.out

    def test_returns_not_ready_status_with_errors(
        self, capsys: pytest.CaptureFixture, tmp_path: Path
    ) -> None:
        """Should return NOT READY status with any errors."""
        checker = PreflightCheck()
        checker.context.successes.extend(["Success 1"])
        checker.context.errors.append("Critical error")

        with patch("gpt_trader.preflight.report.Path") as mock_path:
            mock_path.return_value = tmp_path / "report.json"
            success, status = generate_report(checker)

        assert success is False
        assert status == "NOT READY"

        captured = capsys.readouterr()
        assert "NOT READY" in captured.out
        assert "critical issues must be resolved" in captured.out

    def test_prints_summary_counts(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        """Should print summary of passed/warnings/failed counts."""
        checker = PreflightCheck()
        checker.context.successes.extend(["S1", "S2", "S3"])
        checker.context.warnings.extend(["W1", "W2"])
        checker.context.errors.append("E1")

        with patch("gpt_trader.preflight.report.Path") as mock_path:
            mock_path.return_value = tmp_path / "report.json"
            generate_report(checker)

        captured = capsys.readouterr()
        assert "Passed: 3" in captured.out
        assert "Warnings: 2" in captured.out
        assert "Failed: 1" in captured.out

    def test_prints_recommendations_for_ready(
        self, capsys: pytest.CaptureFixture, tmp_path: Path
    ) -> None:
        """Should print appropriate recommendations for READY status."""
        checker = PreflightCheck(profile="canary")
        checker.context.successes.append("Success")

        with patch("gpt_trader.preflight.report.Path") as mock_path:
            mock_path.return_value = tmp_path / "report.json"
            generate_report(checker)

        captured = capsys.readouterr()
        assert "--dry-run" in captured.out
        assert "tiny positions" in captured.out

    def test_prints_recommendations_for_not_ready(
        self, capsys: pytest.CaptureFixture, tmp_path: Path
    ) -> None:
        """Should print appropriate recommendations for NOT READY status."""
        checker = PreflightCheck()
        checker.context.errors.append("Error")

        with patch("gpt_trader.preflight.report.Path") as mock_path:
            mock_path.return_value = tmp_path / "report.json"
            generate_report(checker)

        captured = capsys.readouterr()
        assert "Fix all critical errors" in captured.out
        assert "Run tests:" in captured.out

    def test_saves_json_report(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        """Should save JSON report to file."""
        checker = PreflightCheck(profile="prod")
        checker.context.successes.extend(["S1", "S2"])
        checker.context.warnings.append("W1")

        # Use actual path for this test
        report_path = tmp_path / "test_report.json"

        with patch("gpt_trader.preflight.report.Path") as mock_path_class:
            mock_path_class.return_value = report_path
            generate_report(checker)

        # Verify file was created with correct content
        assert report_path.exists()

        with open(report_path) as f:
            report_data = json.load(f)

        assert report_data["profile"] == "prod"
        assert report_data["status"] == "READY"
        assert report_data["successes"] == 2
        assert report_data["warnings"] == 1
        assert report_data["errors"] == 0
        assert report_data["details"]["successes"] == ["S1", "S2"]
        assert report_data["details"]["warnings"] == ["W1"]
        assert "timestamp" in report_data

    def test_handles_file_save_error(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        """Should handle file save errors gracefully."""
        checker = PreflightCheck()
        checker.context.successes.append("Success")

        with patch("gpt_trader.preflight.report.Path"):
            # Simulate open failing
            with patch("builtins.open", side_effect=PermissionError("Cannot write")):
                success, status = generate_report(checker)

        # Should still return correct status even if save fails
        assert success is True
        assert status == "READY"

        captured = capsys.readouterr()
        assert "Could not save report" in captured.out

    def test_prints_section_header(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        """Should print section header."""
        checker = PreflightCheck()
        checker.context.successes.append("Success")

        with patch("gpt_trader.preflight.report.Path") as mock_path:
            mock_path.return_value = tmp_path / "report.json"
            generate_report(checker)

        captured = capsys.readouterr()
        assert "PREFLIGHT REPORT" in captured.out


class TestReportCalculations:
    """Test report calculations."""

    def test_total_checks_calculated_correctly(
        self, capsys: pytest.CaptureFixture, tmp_path: Path
    ) -> None:
        """Total checks should be sum of successes + warnings + errors."""
        checker = PreflightCheck()
        checker.context.successes.extend(["S1", "S2", "S3"])
        checker.context.warnings.extend(["W1", "W2"])
        checker.context.errors.append("E1")

        with patch("gpt_trader.preflight.report.Path") as mock_path_class:
            report_path = tmp_path / "report.json"
            mock_path_class.return_value = report_path
            generate_report(checker)

        with open(report_path) as f:
            report_data = json.load(f)

        assert report_data["total_checks"] == 6  # 3 + 2 + 1
