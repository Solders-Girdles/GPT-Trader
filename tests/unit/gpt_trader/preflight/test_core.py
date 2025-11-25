"""Tests for PreflightCheck - the facade over preflight checks."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gpt_trader.preflight.core import PreflightCheck


class TestPreflightCheckInit:
    """Test PreflightCheck initialization."""

    def test_default_initialization(self) -> None:
        """PreflightCheck should initialize with defaults."""
        check = PreflightCheck()

        assert check.verbose is False
        assert check.profile == "canary"
        assert check.context is not None
        assert check.context.profile == "canary"

    def test_custom_initialization(self) -> None:
        """PreflightCheck should accept custom values."""
        check = PreflightCheck(verbose=True, profile="prod")

        assert check.verbose is True
        assert check.profile == "prod"
        assert check.context.verbose is True
        assert check.context.profile == "prod"


class TestPreflightCheckCompatibilityMirrors:
    """Test backwards-compatibility properties that mirror context."""

    def test_errors_mirrors_context(self) -> None:
        """errors property should mirror context.errors."""
        check = PreflightCheck()
        check.context.errors.append("test error")

        assert check.errors == ["test error"]
        assert check.errors is check.context.errors

    def test_warnings_mirrors_context(self) -> None:
        """warnings property should mirror context.warnings."""
        check = PreflightCheck()
        check.context.warnings.append("test warning")

        assert check.warnings == ["test warning"]

    def test_successes_mirrors_context(self) -> None:
        """successes property should mirror context.successes."""
        check = PreflightCheck()
        check.context.successes.append("test success")

        assert check.successes == ["test success"]

    def test_config_mirrors_context(self) -> None:
        """config property should mirror context.config."""
        check = PreflightCheck()
        check.context.config["key"] = "value"

        assert check.config == {"key": "value"}


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


class TestPreflightCheckDelegations:
    """Test that check methods delegate to module functions."""

    def test_check_python_version_delegates(self) -> None:
        """check_python_version should delegate to module function."""
        check = PreflightCheck()

        with patch("gpt_trader.preflight.core.check_python_version") as mock:
            mock.return_value = True
            result = check.check_python_version()

            mock.assert_called_once_with(check)
            assert result is True

    def test_check_dependencies_delegates(self) -> None:
        """check_dependencies should delegate to module function."""
        check = PreflightCheck()

        with patch("gpt_trader.preflight.core.check_dependencies") as mock:
            mock.return_value = True
            result = check.check_dependencies()

            mock.assert_called_once_with(check)
            assert result is True

    def test_check_environment_variables_delegates(self) -> None:
        """check_environment_variables should delegate to module function."""
        check = PreflightCheck()

        with patch("gpt_trader.preflight.core.check_environment_variables") as mock:
            mock.return_value = False
            result = check.check_environment_variables()

            mock.assert_called_once_with(check)
            assert result is False

    def test_check_api_connectivity_delegates(self) -> None:
        """check_api_connectivity should delegate to module function."""
        check = PreflightCheck()

        with patch("gpt_trader.preflight.core.check_api_connectivity") as mock:
            mock.return_value = True
            check.check_api_connectivity()

            mock.assert_called_once_with(check)

    def test_check_key_permissions_delegates(self) -> None:
        """check_key_permissions should delegate to module function."""
        check = PreflightCheck()

        with patch("gpt_trader.preflight.core.check_key_permissions") as mock:
            mock.return_value = True
            check.check_key_permissions()

            mock.assert_called_once_with(check)

    def test_check_risk_configuration_delegates(self) -> None:
        """check_risk_configuration should delegate to module function."""
        check = PreflightCheck()

        with patch("gpt_trader.preflight.core.check_risk_configuration") as mock:
            mock.return_value = True
            check.check_risk_configuration()

            mock.assert_called_once_with(check)

    def test_check_test_suite_delegates(self) -> None:
        """check_test_suite should delegate to module function."""
        check = PreflightCheck()

        with patch("gpt_trader.preflight.core.check_test_suite") as mock:
            mock.return_value = True
            check.check_test_suite()

            mock.assert_called_once_with(check)

    def test_check_profile_configuration_delegates(self) -> None:
        """check_profile_configuration should delegate to module function."""
        check = PreflightCheck()

        with patch("gpt_trader.preflight.core.check_profile_configuration") as mock:
            mock.return_value = True
            check.check_profile_configuration()

            mock.assert_called_once_with(check)

    def test_check_system_time_delegates(self) -> None:
        """check_system_time should delegate to module function."""
        check = PreflightCheck()

        with patch("gpt_trader.preflight.core.check_system_time") as mock:
            mock.return_value = True
            check.check_system_time()

            mock.assert_called_once_with(check)

    def test_check_disk_space_delegates(self) -> None:
        """check_disk_space should delegate to module function."""
        check = PreflightCheck()

        with patch("gpt_trader.preflight.core.check_disk_space") as mock:
            mock.return_value = True
            check.check_disk_space()

            mock.assert_called_once_with(check)

    def test_simulate_dry_run_delegates(self) -> None:
        """simulate_dry_run should delegate to module function."""
        check = PreflightCheck()

        with patch("gpt_trader.preflight.core.simulate_dry_run") as mock:
            mock.return_value = True
            check.simulate_dry_run()

            mock.assert_called_once_with(check)

    def test_generate_report_delegates(self) -> None:
        """generate_report should delegate to module function."""
        check = PreflightCheck()

        with patch("gpt_trader.preflight.core.generate_report") as mock:
            mock.return_value = (True, "READY")
            result = check.generate_report()

            mock.assert_called_once_with(check)
            assert result == (True, "READY")
