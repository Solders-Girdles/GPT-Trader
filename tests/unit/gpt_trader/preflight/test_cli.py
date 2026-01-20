"""Tests for preflight CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from gpt_trader.preflight.cli import _header, main


@dataclass(frozen=True)
class CLIMocks:
    checker: MagicMock
    preflight_class: MagicMock
    header: MagicMock


@pytest.fixture
def cli_mocks(monkeypatch: pytest.MonkeyPatch) -> CLIMocks:
    checker = MagicMock()
    checker.generate_report.return_value = (True, "READY")
    preflight_class = MagicMock(return_value=checker)
    header = MagicMock()

    monkeypatch.setattr("gpt_trader.preflight.cli.PreflightCheck", preflight_class)
    monkeypatch.setattr("gpt_trader.preflight.cli._header", header)

    return CLIMocks(checker=checker, preflight_class=preflight_class, header=header)


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

    @pytest.fixture(autouse=True)
    def _clear_warn_only_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GPT_TRADER_PREFLIGHT_WARN_ONLY", raising=False)

    def test_returns_zero_on_success(self, cli_mocks: CLIMocks) -> None:
        """Should return 0 when all checks pass."""
        cli_mocks.checker.generate_report.return_value = (True, "READY")

        assert main(["--profile", "dev"]) == 0

    def test_returns_one_on_failure(self, cli_mocks: CLIMocks) -> None:
        """Should return 1 when checks fail."""
        cli_mocks.checker.generate_report.return_value = (False, "NOT_READY")

        assert main(["--profile", "prod"]) == 1

    def test_passes_verbose_flag(self, cli_mocks: CLIMocks) -> None:
        """Should pass verbose flag to PreflightCheck."""
        main(["--verbose"])

        cli_mocks.preflight_class.assert_called_once_with(verbose=True, profile="canary")

    def test_passes_profile_flag(self, cli_mocks: CLIMocks) -> None:
        """Should pass profile flag to PreflightCheck."""
        main(["--profile", "prod"])

        cli_mocks.preflight_class.assert_called_once_with(verbose=False, profile="prod")

    def test_uses_canary_profile_by_default(self, cli_mocks: CLIMocks) -> None:
        """Should use canary profile when not specified."""
        main([])

        cli_mocks.preflight_class.assert_called_once_with(verbose=False, profile="canary")

    def test_runs_all_check_functions(self, cli_mocks: CLIMocks) -> None:
        """Should call all check functions."""
        main([])

        # Verify all checks were called
        cli_mocks.checker.check_python_version.assert_called_once()
        cli_mocks.checker.check_dependencies.assert_called_once()
        cli_mocks.checker.check_environment_variables.assert_called_once()
        cli_mocks.checker.check_api_connectivity.assert_called_once()
        cli_mocks.checker.check_key_permissions.assert_called_once()
        cli_mocks.checker.check_risk_configuration.assert_called_once()
        cli_mocks.checker.check_pretrade_diagnostics.assert_called_once()
        cli_mocks.checker.check_test_suite.assert_called_once()
        cli_mocks.checker.check_profile_configuration.assert_called_once()
        cli_mocks.checker.check_system_time.assert_called_once()
        cli_mocks.checker.check_disk_space.assert_called_once()
        cli_mocks.checker.simulate_dry_run.assert_called_once()
        cli_mocks.checker.check_event_store_redaction.assert_called_once()
        cli_mocks.checker.check_readiness_report.assert_called_once()

    def test_handles_check_exception_gracefully(self, cli_mocks: CLIMocks) -> None:
        """Should handle exceptions from checks gracefully."""
        cli_mocks.checker.check_python_version.side_effect = Exception("Test exception")
        cli_mocks.checker.generate_report.return_value = (True, "READY")

        # Should not raise, should handle gracefully
        assert main([]) == 0
        cli_mocks.checker.log_error.assert_called_once()

    def test_calls_header_with_profile(self, cli_mocks: CLIMocks) -> None:
        """Should call header with correct profile."""
        cli_mocks.checker.generate_report.return_value = (True, "READY")

        main(["--profile", "prod"])

        cli_mocks.header.assert_called_once_with("prod")

    def test_short_flags_work(self, cli_mocks: CLIMocks) -> None:
        """Should accept short flag versions."""
        main(["-v", "-p", "prod"])

        cli_mocks.preflight_class.assert_called_once_with(verbose=True, profile="prod")

    def test_warn_only_sets_env_var(self, cli_mocks: CLIMocks) -> None:
        main(["--warn-only"])

        assert os.environ.get("GPT_TRADER_PREFLIGHT_WARN_ONLY") == "1"
