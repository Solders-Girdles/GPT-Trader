"""Tests for PreflightCheck helper delegations."""

from __future__ import annotations

import pytest

from gpt_trader.preflight.core import PreflightCheck


class TestPreflightCheckLoggingHelpers:
    """Test logging helper delegations."""

    def test_log_success_delegates_to_context(self, capsys: pytest.CaptureFixture) -> None:
        """log_success should delegate to context."""
        check = PreflightCheck()
        check.log_success("Success message")

        assert "Success message" in check.successes
        captured = capsys.readouterr()
        assert "Success message" in captured.out

    def test_log_warning_delegates_to_context(self, capsys: pytest.CaptureFixture) -> None:
        """log_warning should delegate to context."""
        check = PreflightCheck()
        check.log_warning("Warning message")

        assert "Warning message" in check.warnings

    def test_log_error_delegates_to_context(self, capsys: pytest.CaptureFixture) -> None:
        """log_error should delegate to context."""
        check = PreflightCheck()
        check.log_error("Error message")

        assert "Error message" in check.errors

    def test_log_info_delegates_to_context(self, capsys: pytest.CaptureFixture) -> None:
        """log_info should delegate to context."""
        check = PreflightCheck(verbose=True)
        check.log_info("Info message")

        captured = capsys.readouterr()
        assert "Info message" in captured.out

    def test_section_header_delegates_to_context(self, capsys: pytest.CaptureFixture) -> None:
        """section_header should delegate to context."""
        check = PreflightCheck()
        check.section_header("Test Header")

        captured = capsys.readouterr()
        assert "Test Header" in captured.out


class TestPreflightCheckEnvironmentHelpers:
    """Test environment helper delegations.

    Note: PreflightContext uses slots=True, so we can't mock methods directly.
    Instead, we test that the facade returns the same results as the context.
    """

    def test_resolve_cdp_credentials_delegates(self) -> None:
        """_resolve_cdp_credentials should return same as context."""
        check = PreflightCheck()

        # Should return the same result as context method
        result = check._resolve_cdp_credentials()
        expected = check.context.resolve_cdp_credentials()

        assert result == expected

    def test_has_real_cdp_credentials_delegates(self) -> None:
        """_has_real_cdp_credentials should return same as context."""
        check = PreflightCheck()

        result = check._has_real_cdp_credentials()
        expected = check.context.has_real_cdp_credentials()

        assert result == expected

    def test_should_skip_remote_checks_delegates(self) -> None:
        """_should_skip_remote_checks should return same as context."""
        check = PreflightCheck()

        result = check._should_skip_remote_checks()
        expected = check.context.should_skip_remote_checks()

        assert result == expected

    def test_expected_env_defaults_delegates(self) -> None:
        """_expected_env_defaults should return same as context."""
        check = PreflightCheck()

        result = check._expected_env_defaults()
        expected = check.context.expected_env_defaults()

        assert result == expected
