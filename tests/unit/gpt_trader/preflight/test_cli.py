"""Tests for preflight CLI."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.preflight.cli import _header, main


class TestHeader:
    """Test CLI header function."""

    def test_prints_header_with_profile(self, capsys: pytest.CaptureFixture) -> None:
        """Should print header with profile name."""
        _header("canary")

        captured = capsys.readouterr()
        assert "GPT-TRADER PRODUCTION PREFLIGHT CHECK" in captured.out
        assert "canary" in captured.out

    def test_prints_timestamp(self, capsys: pytest.CaptureFixture) -> None:
        """Should print timestamp in header."""
        _header("prod")

        captured = capsys.readouterr()
        assert "UTC" in captured.out

    def test_prints_separators(self, capsys: pytest.CaptureFixture) -> None:
        """Should print separator lines."""
        _header("dev")

        captured = capsys.readouterr()
        assert "=" * 70 in captured.out


class TestMain:
    """Test main CLI entry point."""

    def test_returns_zero_on_success(self) -> None:
        """Should return 0 when all checks pass."""
        mock_checker = MagicMock()
        mock_checker.generate_report.return_value = (True, "READY")

        with (
            patch("gpt_trader.preflight.cli.PreflightCheck", return_value=mock_checker),
            patch("gpt_trader.preflight.cli._header"),
        ):
            result = main(["--profile", "dev"])

        assert result == 0

    def test_returns_one_on_failure(self) -> None:
        """Should return 1 when checks fail."""
        mock_checker = MagicMock()
        mock_checker.generate_report.return_value = (False, "NOT_READY")

        with (
            patch("gpt_trader.preflight.cli.PreflightCheck", return_value=mock_checker),
            patch("gpt_trader.preflight.cli._header"),
        ):
            result = main(["--profile", "prod"])

        assert result == 1

    def test_passes_verbose_flag(self) -> None:
        """Should pass verbose flag to PreflightCheck."""
        mock_checker = MagicMock()
        mock_checker.generate_report.return_value = (True, "READY")

        with (
            patch(
                "gpt_trader.preflight.cli.PreflightCheck", return_value=mock_checker
            ) as mock_class,
            patch("gpt_trader.preflight.cli._header"),
        ):
            main(["--verbose"])

            mock_class.assert_called_once_with(verbose=True, profile="canary")

    def test_passes_profile_flag(self) -> None:
        """Should pass profile flag to PreflightCheck."""
        mock_checker = MagicMock()
        mock_checker.generate_report.return_value = (True, "READY")

        with (
            patch(
                "gpt_trader.preflight.cli.PreflightCheck", return_value=mock_checker
            ) as mock_class,
            patch("gpt_trader.preflight.cli._header"),
        ):
            main(["--profile", "prod"])

            mock_class.assert_called_once_with(verbose=False, profile="prod")

    def test_uses_canary_profile_by_default(self) -> None:
        """Should use canary profile when not specified."""
        mock_checker = MagicMock()
        mock_checker.generate_report.return_value = (True, "READY")

        with (
            patch(
                "gpt_trader.preflight.cli.PreflightCheck", return_value=mock_checker
            ) as mock_class,
            patch("gpt_trader.preflight.cli._header"),
        ):
            main([])

            mock_class.assert_called_once_with(verbose=False, profile="canary")

    def test_runs_all_check_functions(self) -> None:
        """Should call all check functions."""
        mock_checker = MagicMock()
        mock_checker.generate_report.return_value = (True, "READY")

        with (
            patch("gpt_trader.preflight.cli.PreflightCheck", return_value=mock_checker),
            patch("gpt_trader.preflight.cli._header"),
        ):
            main([])

            # Verify all checks were called
            mock_checker.check_python_version.assert_called_once()
            mock_checker.check_dependencies.assert_called_once()
            mock_checker.check_environment_variables.assert_called_once()
            mock_checker.check_api_connectivity.assert_called_once()
            mock_checker.check_key_permissions.assert_called_once()
            mock_checker.check_risk_configuration.assert_called_once()
            mock_checker.check_test_suite.assert_called_once()
            mock_checker.check_profile_configuration.assert_called_once()
            mock_checker.check_system_time.assert_called_once()
            mock_checker.check_disk_space.assert_called_once()
            mock_checker.simulate_dry_run.assert_called_once()
            mock_checker.check_readiness_report.assert_called_once()

    def test_handles_check_exception_gracefully(self) -> None:
        """Should handle exceptions from checks gracefully."""
        mock_checker = MagicMock()
        mock_checker.check_python_version.side_effect = Exception("Test exception")
        mock_checker.generate_report.return_value = (True, "READY")

        with (
            patch("gpt_trader.preflight.cli.PreflightCheck", return_value=mock_checker),
            patch("gpt_trader.preflight.cli._header"),
        ):
            # Should not raise, should handle gracefully
            result = main([])

            # Should still complete and return based on report
            assert result == 0

    def test_calls_header_with_profile(self) -> None:
        """Should call header with correct profile."""
        mock_checker = MagicMock()
        mock_checker.generate_report.return_value = (True, "READY")

        with (
            patch("gpt_trader.preflight.cli.PreflightCheck", return_value=mock_checker),
            patch("gpt_trader.preflight.cli._header") as mock_header,
        ):
            main(["--profile", "prod"])

            mock_header.assert_called_once_with("prod")

    def test_short_flags_work(self) -> None:
        """Should accept short flag versions."""
        mock_checker = MagicMock()
        mock_checker.generate_report.return_value = (True, "READY")

        with (
            patch(
                "gpt_trader.preflight.cli.PreflightCheck", return_value=mock_checker
            ) as mock_class,
            patch("gpt_trader.preflight.cli._header"),
        ):
            main(["-v", "-p", "prod"])

            mock_class.assert_called_once_with(verbose=True, profile="prod")
