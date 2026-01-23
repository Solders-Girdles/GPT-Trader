"""Tests for environment configuration preflight checks."""

from __future__ import annotations

import os

import pytest

from gpt_trader.preflight.checks.environment import check_environment_variables
from gpt_trader.preflight.core import PreflightCheck


def _set_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str], *, clear: bool = True) -> None:
    if clear:
        for key in list(os.environ):
            monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


class TestCheckEnvironmentVariables:
    """Test environment variable validation."""

    def test_passes_with_correct_prod_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should pass with correct production configuration."""
        checker = PreflightCheck(profile="prod")

        env = {
            "BROKER": "coinbase",
            "COINBASE_SANDBOX": "0",
            "COINBASE_API_MODE": "advanced",
            "COINBASE_ENABLE_INTX_PERPS": "0",
            "COINBASE_CDP_API_KEY": "organizations/abc/apiKeys/xyz",
            "COINBASE_CDP_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
        }

        with monkeypatch.context() as mp:
            _set_env(mp, env, clear=True)
            result = check_environment_variables(checker)

        assert result is True
        assert any("CDP API key format valid" in s for s in checker.successes)
        assert any("source=" in s for s in checker.successes)
        assert any("CDP private key found" in s for s in checker.successes)

    def test_fails_with_wrong_broker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should fail when BROKER is not coinbase."""
        checker = PreflightCheck(profile="prod")

        env = {
            "BROKER": "kraken",  # Wrong broker
            "COINBASE_SANDBOX": "0",
            "COINBASE_API_MODE": "advanced",
            "COINBASE_ENABLE_INTX_PERPS": "0",
        }

        with monkeypatch.context() as mp:
            _set_env(mp, env, clear=True)
            result = check_environment_variables(checker)

        assert result is False
        assert any("BROKER=kraken" in e for e in checker.errors)

    def test_fails_with_sandbox_in_prod(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should fail when sandbox enabled in prod profile."""
        checker = PreflightCheck(profile="prod")

        env = {
            "BROKER": "coinbase",
            "COINBASE_SANDBOX": "1",  # Should be 0 in prod
            "COINBASE_API_MODE": "advanced",
            "COINBASE_ENABLE_INTX_PERPS": "0",
        }

        with monkeypatch.context() as mp:
            _set_env(mp, env, clear=True)
            result = check_environment_variables(checker)

        assert result is False
        assert any("COINBASE_SANDBOX" in e for e in checker.errors)

    def test_warns_with_wrong_config_in_dev(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should warn (not fail) for non-strict vars in dev profile."""
        checker = PreflightCheck(profile="dev")

        env = {
            "BROKER": "coinbase",
            "COINBASE_SANDBOX": "0",  # Expected "1" in dev but not strict
        }

        with monkeypatch.context() as mp:
            _set_env(mp, env, clear=True)
            check_environment_variables(checker)

        # Should still pass because COINBASE_SANDBOX is not strict in dev
        # But will fail for missing credentials unless skip_remote
        # Let's check warnings are logged
        assert any("COINBASE_SANDBOX" in w for w in checker.warnings)

    def test_validates_cdp_api_key_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should validate CDP API key format."""
        checker = PreflightCheck(profile="prod")

        env = {
            "BROKER": "coinbase",
            "COINBASE_SANDBOX": "0",
            "COINBASE_API_MODE": "advanced",
            "COINBASE_ENABLE_INTX_PERPS": "0",
            "COINBASE_CDP_API_KEY": "invalid_format",  # Invalid format
            "COINBASE_CDP_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
        }

        with monkeypatch.context() as mp:
            _set_env(mp, env, clear=True)
            result = check_environment_variables(checker)

        assert result is False
        assert any("Invalid CDP API key format" in e for e in checker.errors)

    def test_validates_private_key_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should validate CDP private key format."""
        checker = PreflightCheck(profile="prod")

        env = {
            "BROKER": "coinbase",
            "COINBASE_SANDBOX": "0",
            "COINBASE_API_MODE": "advanced",
            "COINBASE_ENABLE_INTX_PERPS": "0",
            "COINBASE_CDP_API_KEY": "organizations/abc/apiKeys/xyz",
            "COINBASE_CDP_PRIVATE_KEY": "not-a-valid-key",  # Invalid format
        }

        with monkeypatch.context() as mp:
            _set_env(mp, env, clear=True)
            result = check_environment_variables(checker)

        assert result is False
        assert any("Invalid private key format" in e for e in checker.errors)

    def test_warns_about_missing_credentials_in_dev(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should warn (not fail) about missing credentials in dev profile."""
        checker = PreflightCheck(profile="dev")

        env = {
            "BROKER": "coinbase",
            # No CDP credentials
        }

        with monkeypatch.context() as mp:
            _set_env(mp, env, clear=True)
            check_environment_variables(checker)

        # In dev without credentials, should warn but not fail
        assert any("remote connectivity checks will be skipped" in w for w in checker.warnings)

    def test_validates_risk_variables_in_range(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Should validate risk variables are in acceptable range."""
        checker = PreflightCheck(profile="dev", verbose=True)

        env = {
            "BROKER": "coinbase",
            "RISK_MAX_LEVERAGE": "5",  # Within 1-10
            "RISK_DAILY_LOSS_LIMIT": "500",  # Within 10-10000
            "RISK_MAX_POSITION_PCT_PER_SYMBOL": "0.1",  # Within 0.01-0.5
        }

        with monkeypatch.context() as mp:
            _set_env(mp, env, clear=True)
            check_environment_variables(checker)

        captured = capsys.readouterr()
        assert "within safe range" in captured.out

    def test_warns_about_out_of_range_risk_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should warn about risk variables outside recommended range."""
        checker = PreflightCheck(profile="dev")

        env = {
            "BROKER": "coinbase",
            "RISK_MAX_LEVERAGE": "20",  # Outside 1-10
        }

        with monkeypatch.context() as mp:
            _set_env(mp, env, clear=True)
            check_environment_variables(checker)

        assert any("outside recommended range" in w for w in checker.warnings)

    def test_fails_on_invalid_risk_variable_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should fail on non-numeric risk variable."""
        checker = PreflightCheck(profile="dev")

        env = {
            "BROKER": "coinbase",
            "RISK_MAX_LEVERAGE": "not-a-number",
        }

        with monkeypatch.context() as mp:
            _set_env(mp, env, clear=True)
            result = check_environment_variables(checker)

        assert result is False
        assert any("not a valid number" in e for e in checker.errors)

    def test_warns_about_missing_risk_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should warn about missing risk variables."""
        checker = PreflightCheck(profile="dev")

        env = {
            "BROKER": "coinbase",
            # No risk variables set
        }

        with monkeypatch.context() as mp:
            _set_env(mp, env, clear=True)
            check_environment_variables(checker)

        assert any("using defaults" in w for w in checker.warnings)

    def test_prints_section_header(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Should print section header."""
        checker = PreflightCheck(profile="dev")

        with monkeypatch.context() as mp:
            _set_env(mp, {"BROKER": "coinbase"}, clear=True)
            check_environment_variables(checker)

        captured = capsys.readouterr()
        assert "ENVIRONMENT CONFIGURATION" in captured.out
