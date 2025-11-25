"""Tests for system preflight checks (Python version, disk space, time)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.preflight.checks.system import (
    check_disk_space,
    check_python_version,
    check_system_time,
)
from gpt_trader.preflight.core import PreflightCheck


class TestCheckPythonVersion:
    """Test Python version check."""

    def test_passes_on_python_312_or_higher(self, capsys: pytest.CaptureFixture) -> None:
        """Should pass on Python 3.12+."""
        checker = PreflightCheck()

        # sys.version_info is actually running, and we're on 3.12+
        # Just verify the check passes with the real Python version
        result = check_python_version(checker)

        # We're running on Python 3.12+, so this should pass
        assert result is True
        assert any("meets requirements" in s for s in checker.successes)

    def test_passes_on_current_python(self, capsys: pytest.CaptureFixture) -> None:
        """Should pass on current Python version (3.12+)."""
        checker = PreflightCheck()
        result = check_python_version(checker)

        # Tests only run on Python 3.12+, so this should always pass
        assert result is True

    def test_logs_python_version(self, capsys: pytest.CaptureFixture) -> None:
        """Should log the Python version."""
        checker = PreflightCheck()
        check_python_version(checker)

        captured = capsys.readouterr()
        assert f"{sys.version_info.major}.{sys.version_info.minor}" in captured.out

    def test_version_check_logic(self) -> None:
        """Test the version check logic directly."""
        # The check passes if major == 3 and minor >= 12
        assert sys.version_info.major == 3
        assert sys.version_info.minor >= 12  # Project requires 3.12+

    def test_prints_section_header(self, capsys: pytest.CaptureFixture) -> None:
        """Should print section header."""
        checker = PreflightCheck()
        check_python_version(checker)

        captured = capsys.readouterr()
        assert "PYTHON VERSION CHECK" in captured.out


class TestCheckDiskSpace:
    """Test disk space check."""

    def test_passes_with_plenty_of_space(self, capsys: pytest.CaptureFixture) -> None:
        """Should pass with >1GB free."""
        checker = PreflightCheck()

        # Mock 100GB total, 50GB free
        mock_usage = MagicMock()
        mock_usage.total = 100 * 1024**3
        mock_usage.free = 50 * 1024**3
        mock_usage.used = 50 * 1024**3

        with patch("shutil.disk_usage", return_value=mock_usage):
            result = check_disk_space(checker)

        assert result is True
        assert any("Disk space" in s for s in checker.successes)

    def test_warns_with_low_space(self, capsys: pytest.CaptureFixture) -> None:
        """Should warn with 0.5-1GB free."""
        checker = PreflightCheck()

        # Mock 100GB total, 0.7GB free
        mock_usage = MagicMock()
        mock_usage.total = 100 * 1024**3
        mock_usage.free = int(0.7 * 1024**3)
        mock_usage.used = int(99.3 * 1024**3)

        with patch("shutil.disk_usage", return_value=mock_usage):
            result = check_disk_space(checker)

        assert result is True  # Still passes but with warning
        assert any("Low disk space" in w for w in checker.warnings)

    def test_fails_with_critical_space(self, capsys: pytest.CaptureFixture) -> None:
        """Should fail with <0.5GB free."""
        checker = PreflightCheck()

        # Mock 100GB total, 0.3GB free
        mock_usage = MagicMock()
        mock_usage.total = 100 * 1024**3
        mock_usage.free = int(0.3 * 1024**3)
        mock_usage.used = int(99.7 * 1024**3)

        with patch("shutil.disk_usage", return_value=mock_usage):
            result = check_disk_space(checker)

        assert result is False
        assert any("Critical" in e for e in checker.errors)

    def test_handles_exception(self, capsys: pytest.CaptureFixture) -> None:
        """Should handle exceptions gracefully."""
        checker = PreflightCheck()

        with patch("shutil.disk_usage", side_effect=OSError("Permission denied")):
            result = check_disk_space(checker)

        assert result is False
        assert any("Failed to check disk space" in e for e in checker.errors)


class TestCheckSystemTime:
    """Test system time synchronization check."""

    def test_passes_with_reasonable_time_no_credentials(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """Should pass with reasonable time when no credentials available."""
        checker = PreflightCheck(profile="dev")

        with patch.dict("os.environ", {}, clear=True):
            # Mock datetime to return a reasonable time
            mock_now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
            with patch("gpt_trader.preflight.checks.system.datetime") as mock_datetime:
                mock_datetime.now.return_value = mock_now
                mock_datetime.fromisoformat = datetime.fromisoformat
                result = check_system_time(checker)

        assert result is True
        assert any("seems reasonable" in w for w in checker.warnings)

    def test_fails_with_unreasonable_time(self, capsys: pytest.CaptureFixture) -> None:
        """Should fail with clearly wrong system time."""
        checker = PreflightCheck(profile="dev")

        with patch.dict("os.environ", {}, clear=True):
            # Mock datetime to return an unreasonable time (year 2015)
            mock_now = datetime(2015, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
            with patch("gpt_trader.preflight.checks.system.datetime") as mock_datetime:
                mock_datetime.now.return_value = mock_now
                result = check_system_time(checker)

        assert result is False
        assert any("seems wrong" in e for e in checker.errors)

    def test_handles_exception(self, capsys: pytest.CaptureFixture) -> None:
        """Should handle exceptions gracefully."""
        checker = PreflightCheck()

        with patch("gpt_trader.preflight.checks.system.datetime") as mock_datetime:
            mock_datetime.now.side_effect = Exception("Time error")
            result = check_system_time(checker)

        assert result is False
        assert any("Failed to check system time" in e for e in checker.errors)

    def test_prints_section_header(self, capsys: pytest.CaptureFixture) -> None:
        """Should print section header."""
        checker = PreflightCheck(profile="dev")

        with patch.dict("os.environ", {}, clear=True):
            mock_now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
            with patch("gpt_trader.preflight.checks.system.datetime") as mock_datetime:
                mock_datetime.now.return_value = mock_now
                check_system_time(checker)

        captured = capsys.readouterr()
        assert "SYSTEM TIME SYNC" in captured.out
