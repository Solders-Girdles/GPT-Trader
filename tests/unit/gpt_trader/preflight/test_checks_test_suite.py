"""Tests for test suite preflight checks."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

from gpt_trader.preflight.checks.test_suite import check_test_suite
from gpt_trader.preflight.core import PreflightCheck


def _make_result(stdout: str, stderr: str = "") -> MagicMock:
    result = MagicMock()
    result.stdout = stdout
    result.stderr = stderr
    return result


@pytest.fixture
def run_stub(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    stub = MagicMock()
    monkeypatch.setattr(subprocess, "run", stub)
    return stub


class TestCheckTestSuite:
    """Test test suite validation."""

    def test_passes_when_all_tests_pass(self, run_stub: MagicMock) -> None:
        """Should pass when pytest reports all tests passing."""
        checker = PreflightCheck(profile="dev")

        run_stub.return_value = _make_result("15 passed in 2.5s")
        result = check_test_suite(checker)

        assert result is True
        assert any("15 core tests passed" in s for s in checker.successes)

    def test_fails_when_tests_fail(self, run_stub: MagicMock) -> None:
        """Should fail when pytest reports failing tests."""
        checker = PreflightCheck(profile="dev")

        run_stub.return_value = _make_result("12 passed, 3 failed in 2.5s")
        result = check_test_suite(checker)

        assert result is False
        assert any("3 tests failed" in w for w in checker.warnings)

    def test_fails_when_no_passed_in_output(self, run_stub: MagicMock) -> None:
        """Should fail when output doesn't contain 'passed'."""
        checker = PreflightCheck(profile="dev")

        run_stub.return_value = _make_result("ERROR: No tests collected")
        result = check_test_suite(checker)

        assert result is False
        assert any("failed" in e for e in checker.errors)

    def test_fails_on_timeout(self, run_stub: MagicMock) -> None:
        """Should fail when tests timeout."""
        checker = PreflightCheck(profile="dev")

        run_stub.side_effect = subprocess.TimeoutExpired("pytest", 30)
        result = check_test_suite(checker)

        assert result is False
        assert any("timed out" in e for e in checker.errors)

    def test_fails_on_exception(self, run_stub: MagicMock) -> None:
        """Should fail when subprocess raises an exception."""
        checker = PreflightCheck(profile="dev")

        run_stub.side_effect = FileNotFoundError("pytest not found")
        result = check_test_suite(checker)

        assert result is False
        assert any("Failed to run tests" in e for e in checker.errors)

    def test_shows_output_when_verbose(
        self, capsys: pytest.CaptureFixture, run_stub: MagicMock
    ) -> None:
        """Should show output when verbose and tests fail."""
        checker = PreflightCheck(profile="dev", verbose=True)

        run_stub.return_value = _make_result("Error details here", "Some warnings")
        check_test_suite(checker)

        captured = capsys.readouterr()
        assert "Error details here" in captured.out

    def test_prints_section_header(
        self, capsys: pytest.CaptureFixture, run_stub: MagicMock
    ) -> None:
        """Should print section header."""
        checker = PreflightCheck(profile="dev")

        run_stub.return_value = _make_result("10 passed")
        check_test_suite(checker)

        captured = capsys.readouterr()
        assert "TEST SUITE" in captured.out

    def test_parses_passed_count_correctly(self, run_stub: MagicMock) -> None:
        """Should correctly parse various passed count formats."""
        checker = PreflightCheck(profile="dev")

        run_stub.return_value = _make_result("===== 157 passed, 2 skipped in 5.23s =====")
        result = check_test_suite(checker)

        assert result is True
        assert any("157 core tests passed" in s for s in checker.successes)

    def test_handles_passed_without_count(self, run_stub: MagicMock) -> None:
        """Should handle 'passed' keyword without extractable count."""
        checker = PreflightCheck(profile="dev")

        # "passed" is present but no number before it
        run_stub.return_value = _make_result("All tests passed successfully")
        result = check_test_suite(checker)

        # Should still pass since "passed" is in output
        assert result is True
