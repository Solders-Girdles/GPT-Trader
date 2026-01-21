"""Tests for validation preflight checks: event store redaction and test suite."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gpt_trader.preflight.checks.event_store import check_event_store_redaction
from gpt_trader.preflight.checks.test_suite import check_test_suite
from gpt_trader.preflight.core import PreflightCheck


def _write_events_db(events_db_path: Path, *, events: list[tuple[str, str]] | None = None) -> None:
    events_db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(events_db_path) as connection:
        connection.execute(
            "CREATE TABLE events (id INTEGER PRIMARY KEY, event_type TEXT, payload TEXT)"
        )
        if not events:
            return
        connection.executemany(
            "INSERT INTO events (event_type, payload) VALUES (?, ?)",
            events,
        )


class TestCheckEventStoreRedaction:
    """Tests for event store redaction check."""

    def test_passes_when_no_sensitive_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime_root = tmp_path / "runtime_data" / "dev"
        events_path = runtime_root / "events.db"
        payload = json.dumps({"status": "ok", "nested": {"value": 1}})
        _write_events_db(events_path, events=[("heartbeat", payload)])

        monkeypatch.setenv("GPT_TRADER_RUNTIME_ROOT", str(tmp_path))
        checker = PreflightCheck(profile="dev")

        assert check_event_store_redaction(checker) is True
        assert any("Event store redaction check passed" in s for s in checker.successes)

    def test_passes_when_redacted_values_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime_root = tmp_path / "runtime_data" / "dev"
        events_path = runtime_root / "events.db"
        payload = json.dumps({"private_key": "[REDACTED]", "apiKey": "[REDACTED]"})
        _write_events_db(events_path, events=[("order", payload)])

        monkeypatch.setenv("GPT_TRADER_RUNTIME_ROOT", str(tmp_path))
        checker = PreflightCheck(profile="dev")

        assert check_event_store_redaction(checker) is True
        assert any("Event store redaction check passed" in s for s in checker.successes)

    def test_fails_on_pem_private_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime_root = tmp_path / "runtime_data" / "prod"
        events_path = runtime_root / "events.db"
        payload = json.dumps({"payload": "-----BEGIN EC PRIVATE KEY-----\\nABC"})
        _write_events_db(events_path, events=[("order", payload)])

        monkeypatch.setenv("GPT_TRADER_RUNTIME_ROOT", str(tmp_path))
        checker = PreflightCheck(profile="prod")

        assert check_event_store_redaction(checker) is False
        assert any("Event store redaction check found" in e for e in checker.errors)
        joined = " ".join(checker.errors)
        assert "BEGIN" not in joined
        assert "ABC" not in joined

    def test_fails_on_unredacted_authorization(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime_root = tmp_path / "runtime_data" / "prod"
        events_path = runtime_root / "events.db"
        payload = json.dumps({"authorization": "Bearer abc.def.ghi"})
        _write_events_db(events_path, events=[("auth", payload)])

        monkeypatch.setenv("GPT_TRADER_RUNTIME_ROOT", str(tmp_path))
        checker = PreflightCheck(profile="prod")

        assert check_event_store_redaction(checker) is False
        assert any("Unredacted secret in event" in e for e in checker.errors)
        assert any("authorization" in e for e in checker.errors)
        assert not any("Bearer" in e for e in checker.errors)

    def test_missing_events_db_warns_in_dev(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GPT_TRADER_RUNTIME_ROOT", str(tmp_path))
        checker = PreflightCheck(profile="dev")

        assert check_event_store_redaction(checker) is True
        assert any("Events DB not found" in w for w in checker.warnings)

    def test_missing_events_db_errors_in_prod(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GPT_TRADER_RUNTIME_ROOT", str(tmp_path))
        checker = PreflightCheck(profile="prod")

        assert check_event_store_redaction(checker) is False
        assert any("Events DB not found" in e for e in checker.errors)

    def test_missing_events_db_warn_only_in_prod(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GPT_TRADER_RUNTIME_ROOT", str(tmp_path))
        monkeypatch.setenv("GPT_TRADER_PREFLIGHT_WARN_ONLY", "1")
        checker = PreflightCheck(profile="prod")

        assert check_event_store_redaction(checker) is True
        assert any("Events DB not found" in w for w in checker.warnings)


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
