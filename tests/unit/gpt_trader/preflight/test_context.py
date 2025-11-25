"""Tests for PreflightContext - the shared state and logging utilities."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from gpt_trader.preflight.context import Colors, PreflightContext


class TestColors:
    """Test terminal color constants."""

    def test_colors_are_ansi_escape_sequences(self) -> None:
        """All color constants should be ANSI escape sequences."""
        assert Colors.RED.startswith("\033[")
        assert Colors.GREEN.startswith("\033[")
        assert Colors.YELLOW.startswith("\033[")
        assert Colors.BLUE.startswith("\033[")
        assert Colors.RESET.startswith("\033[")
        assert Colors.BOLD.startswith("\033[")

    def test_reset_terminates_formatting(self) -> None:
        """RESET should end all formatting."""
        assert Colors.RESET == "\033[0m"


class TestPreflightContextInit:
    """Test PreflightContext initialization."""

    def test_default_initialization(self) -> None:
        """Context should initialize with sensible defaults."""
        ctx = PreflightContext()

        assert ctx.verbose is False
        assert ctx.profile == "canary"
        assert ctx.errors == []
        assert ctx.warnings == []
        assert ctx.successes == []
        assert ctx.config == {}

    def test_custom_initialization(self) -> None:
        """Context should accept custom values."""
        ctx = PreflightContext(verbose=True, profile="prod")

        assert ctx.verbose is True
        assert ctx.profile == "prod"

    def test_lists_are_independent(self) -> None:
        """Each context should have independent lists."""
        ctx1 = PreflightContext()
        ctx2 = PreflightContext()

        ctx1.errors.append("error1")
        ctx2.warnings.append("warning1")

        assert ctx1.errors == ["error1"]
        assert ctx2.errors == []
        assert ctx1.warnings == []
        assert ctx2.warnings == ["warning1"]


class TestPreflightContextLogging:
    """Test logging methods."""

    def test_log_success_appends_to_list(self, capsys: pytest.CaptureFixture) -> None:
        """log_success should append to successes list and print."""
        ctx = PreflightContext()
        ctx.log_success("Test passed")

        assert "Test passed" in ctx.successes
        captured = capsys.readouterr()
        assert "Test passed" in captured.out
        assert Colors.GREEN in captured.out

    def test_log_warning_appends_to_list(self, capsys: pytest.CaptureFixture) -> None:
        """log_warning should append to warnings list and print."""
        ctx = PreflightContext()
        ctx.log_warning("Potential issue")

        assert "Potential issue" in ctx.warnings
        captured = capsys.readouterr()
        assert "Potential issue" in captured.out
        assert Colors.YELLOW in captured.out

    def test_log_error_appends_to_list(self, capsys: pytest.CaptureFixture) -> None:
        """log_error should append to errors list and print."""
        ctx = PreflightContext()
        ctx.log_error("Critical failure")

        assert "Critical failure" in ctx.errors
        captured = capsys.readouterr()
        assert "Critical failure" in captured.out
        assert Colors.RED in captured.out

    def test_log_info_only_prints_when_verbose(self, capsys: pytest.CaptureFixture) -> None:
        """log_info should only print when verbose=True."""
        ctx_quiet = PreflightContext(verbose=False)
        ctx_verbose = PreflightContext(verbose=True)

        ctx_quiet.log_info("Info message")
        captured = capsys.readouterr()
        assert "Info message" not in captured.out

        ctx_verbose.log_info("Info message")
        captured = capsys.readouterr()
        assert "Info message" in captured.out
        assert Colors.CYAN in captured.out

    def test_section_header_prints_formatted(self, capsys: pytest.CaptureFixture) -> None:
        """section_header should print a formatted header."""
        ctx = PreflightContext()
        ctx.section_header("Test Section")

        captured = capsys.readouterr()
        assert "Test Section" in captured.out
        assert "=" in captured.out
        assert Colors.BLUE in captured.out
        assert Colors.BOLD in captured.out


class TestPreflightContextCredentials:
    """Test CDP credential resolution."""

    def test_resolve_cdp_credentials_from_prod_env(self) -> None:
        """Should prefer PROD env vars."""
        with patch.dict(
            os.environ,
            {
                "COINBASE_PROD_CDP_API_KEY": "prod_key",
                "COINBASE_PROD_CDP_PRIVATE_KEY": "prod_private",
                "COINBASE_CDP_API_KEY": "fallback_key",
                "COINBASE_CDP_PRIVATE_KEY": "fallback_private",
            },
            clear=False,
        ):
            ctx = PreflightContext()
            api_key, private_key = ctx.resolve_cdp_credentials()

            assert api_key == "prod_key"
            assert private_key == "prod_private"

    def test_resolve_cdp_credentials_fallback(self) -> None:
        """Should fall back to non-PROD env vars."""
        with patch.dict(
            os.environ,
            {
                "COINBASE_CDP_API_KEY": "fallback_key",
                "COINBASE_CDP_PRIVATE_KEY": "fallback_private",
            },
            clear=True,
        ):
            ctx = PreflightContext()
            api_key, private_key = ctx.resolve_cdp_credentials()

            assert api_key == "fallback_key"
            assert private_key == "fallback_private"

    def test_resolve_cdp_credentials_returns_none_when_missing(self) -> None:
        """Should return None when credentials not set."""
        with patch.dict(os.environ, {}, clear=True):
            ctx = PreflightContext()
            api_key, private_key = ctx.resolve_cdp_credentials()

            assert api_key is None
            assert private_key is None

    def test_has_real_cdp_credentials_valid(self) -> None:
        """Should return True for valid CDP credentials."""
        with patch.dict(
            os.environ,
            {
                "COINBASE_CDP_API_KEY": "organizations/abc/apiKeys/xyz",
                "COINBASE_CDP_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
            },
            clear=True,
        ):
            ctx = PreflightContext()
            assert ctx.has_real_cdp_credentials() is True

    def test_has_real_cdp_credentials_invalid_key_format(self) -> None:
        """Should return False for invalid key format."""
        with patch.dict(
            os.environ,
            {
                "COINBASE_CDP_API_KEY": "invalid_format",
                "COINBASE_CDP_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
            },
            clear=True,
        ):
            ctx = PreflightContext()
            assert ctx.has_real_cdp_credentials() is False

    def test_has_real_cdp_credentials_invalid_private_key_format(self) -> None:
        """Should return False for invalid private key format."""
        with patch.dict(
            os.environ,
            {
                "COINBASE_CDP_API_KEY": "organizations/abc/apiKeys/xyz",
                "COINBASE_CDP_PRIVATE_KEY": "not-a-valid-key",
            },
            clear=True,
        ):
            ctx = PreflightContext()
            assert ctx.has_real_cdp_credentials() is False

    def test_has_real_cdp_credentials_missing(self) -> None:
        """Should return False when credentials missing."""
        with patch.dict(os.environ, {}, clear=True):
            ctx = PreflightContext()
            assert ctx.has_real_cdp_credentials() is False


class TestPreflightContextSkipRemoteChecks:
    """Test remote check skip logic."""

    def test_force_remote_overrides_skip(self) -> None:
        """COINBASE_PREFLIGHT_FORCE_REMOTE=1 should force remote checks."""
        with patch.dict(
            os.environ,
            {"COINBASE_PREFLIGHT_FORCE_REMOTE": "1"},
            clear=True,
        ):
            ctx = PreflightContext(profile="dev")
            assert ctx.should_skip_remote_checks() is False

    def test_skip_remote_env_var(self) -> None:
        """COINBASE_PREFLIGHT_SKIP_REMOTE=1 should skip remote checks."""
        with patch.dict(
            os.environ,
            {"COINBASE_PREFLIGHT_SKIP_REMOTE": "1"},
            clear=True,
        ):
            ctx = PreflightContext()
            assert ctx.should_skip_remote_checks() is True

    def test_dev_profile_without_credentials_skips(self) -> None:
        """Dev profile without real credentials should skip remote checks."""
        with patch.dict(os.environ, {}, clear=True):
            ctx = PreflightContext(profile="dev")
            assert ctx.should_skip_remote_checks() is True

    def test_prod_profile_without_credentials_does_not_skip(self) -> None:
        """Prod profile without credentials should NOT skip (will fail later)."""
        with patch.dict(os.environ, {}, clear=True):
            ctx = PreflightContext(profile="prod")
            assert ctx.should_skip_remote_checks() is False


class TestPreflightContextEnvDefaults:
    """Test expected environment defaults by profile."""

    def test_dev_profile_defaults(self) -> None:
        """Dev profile should have relaxed defaults."""
        ctx = PreflightContext(profile="dev")
        defaults = ctx.expected_env_defaults()

        assert defaults["BROKER"] == ("coinbase", True)
        assert defaults["COINBASE_SANDBOX"] == ("1", False)  # Not strict
        assert defaults["COINBASE_ENABLE_DERIVATIVES"] == ("0", False)  # Not strict

    def test_prod_profile_defaults(self) -> None:
        """Prod profile should have strict defaults."""
        ctx = PreflightContext(profile="prod")
        defaults = ctx.expected_env_defaults()

        assert defaults["BROKER"] == ("coinbase", True)
        assert defaults["COINBASE_SANDBOX"] == ("0", True)  # Strict
        assert defaults["COINBASE_ENABLE_DERIVATIVES"] == ("1", True)  # Strict

    def test_canary_profile_uses_prod_defaults(self) -> None:
        """Canary profile should use prod-like defaults."""
        ctx = PreflightContext(profile="canary")
        defaults = ctx.expected_env_defaults()

        # Canary is not dev, so should get production defaults
        assert defaults["COINBASE_SANDBOX"] == ("0", True)
