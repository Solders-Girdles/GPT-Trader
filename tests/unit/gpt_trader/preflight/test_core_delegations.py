"""Tests for PreflightCheck method delegation wiring."""

from __future__ import annotations

from unittest.mock import patch

from gpt_trader.preflight.core import PreflightCheck


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

    def test_check_readiness_report_delegates(self) -> None:
        """check_readiness_report should delegate to module function."""
        check = PreflightCheck()

        with patch("gpt_trader.preflight.core.check_readiness_report") as mock:
            mock.return_value = True
            check.check_readiness_report()

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
