"""Tests for PreflightContext remote-check decisions and env defaults."""

from __future__ import annotations

import os
from unittest.mock import patch

from gpt_trader.preflight.context import PreflightContext


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
