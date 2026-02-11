"""Tests for PreflightContext remote-check decisions and env defaults."""

from __future__ import annotations

import pytest

from gpt_trader.preflight.context import PreflightContext


class TestPreflightContextSkipRemoteChecks:
    """Test remote check skip logic."""

    def test_force_remote_overrides_skip(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """COINBASE_PREFLIGHT_FORCE_REMOTE=1 should force remote checks."""
        monkeypatch.delenv("COINBASE_PREFLIGHT_SKIP_REMOTE", raising=False)
        monkeypatch.setenv("COINBASE_PREFLIGHT_FORCE_REMOTE", "1")

        ctx = PreflightContext(profile="dev")
        assert ctx.should_skip_remote_checks() is False

    def test_skip_remote_env_var(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """COINBASE_PREFLIGHT_SKIP_REMOTE=1 should skip remote checks."""
        monkeypatch.delenv("COINBASE_PREFLIGHT_FORCE_REMOTE", raising=False)
        monkeypatch.setenv("COINBASE_PREFLIGHT_SKIP_REMOTE", "1")

        ctx = PreflightContext()
        assert ctx.should_skip_remote_checks() is True

    def test_dev_profile_without_credentials_skips(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dev profile without real credentials should skip remote checks."""
        monkeypatch.delenv("COINBASE_PREFLIGHT_FORCE_REMOTE", raising=False)
        monkeypatch.delenv("COINBASE_PREFLIGHT_SKIP_REMOTE", raising=False)
        monkeypatch.delenv("COINBASE_CDP_API_KEY", raising=False)
        monkeypatch.delenv("COINBASE_CDP_PRIVATE_KEY", raising=False)

        ctx = PreflightContext(profile="dev")
        assert ctx.should_skip_remote_checks() is True

    def test_prod_profile_without_credentials_does_not_skip(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Prod profile without credentials should NOT skip (will fail later)."""
        monkeypatch.delenv("COINBASE_PREFLIGHT_FORCE_REMOTE", raising=False)
        monkeypatch.delenv("COINBASE_PREFLIGHT_SKIP_REMOTE", raising=False)
        monkeypatch.delenv("COINBASE_CDP_API_KEY", raising=False)
        monkeypatch.delenv("COINBASE_CDP_PRIVATE_KEY", raising=False)

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
        assert defaults["COINBASE_ENABLE_INTX_PERPS"] == ("0", False)  # Not strict

    def test_prod_profile_defaults(self) -> None:
        """Prod profile should have strict defaults."""
        ctx = PreflightContext(profile="prod")
        defaults = ctx.expected_env_defaults()

        assert defaults["BROKER"] == ("coinbase", True)
        assert defaults["COINBASE_SANDBOX"] == ("0", True)  # Strict
        assert defaults["COINBASE_ENABLE_INTX_PERPS"] == ("0", True)  # Strict

    def test_canary_profile_uses_prod_defaults(self) -> None:
        """Canary profile should use prod-like defaults."""
        ctx = PreflightContext(profile="canary")
        defaults = ctx.expected_env_defaults()

        # Canary is not dev, so should get production defaults
        assert defaults["COINBASE_SANDBOX"] == ("0", True)


class TestPreflightContextTradingIntent:
    """Test real-order intent and permission decisions."""

    def test_trading_modes_defaults_to_spot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TRADING_MODES", raising=False)

        ctx = PreflightContext()

        assert ctx.trading_modes() == ["spot"]

    def test_trading_modes_ignores_empty_entries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRADING_MODES", "  ,  ,SPOT, cfm ,,")

        ctx = PreflightContext()

        assert ctx.trading_modes() == ["spot", "cfm"]

    @pytest.mark.parametrize(
        "env_overrides",
        [
            {"DRY_RUN": "1"},
            {"PAPER_MODE": "1"},
            {"PERPS_PAPER": "1"},
            {"COINBASE_SANDBOX": "1"},
        ],
    )
    def test_intends_real_orders_is_false_when_safety_mode_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_overrides: dict[str, str],
    ) -> None:
        for key in ("DRY_RUN", "PAPER_MODE", "PERPS_PAPER", "COINBASE_SANDBOX"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env_overrides.items():
            monkeypatch.setenv(key, value)

        ctx = PreflightContext()

        assert ctx.intends_real_orders() is False

    def test_requires_trade_permission_for_spot_and_cfm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DRY_RUN", raising=False)
        monkeypatch.delenv("PAPER_MODE", raising=False)
        monkeypatch.delenv("PERPS_PAPER", raising=False)
        monkeypatch.delenv("COINBASE_SANDBOX", raising=False)
        monkeypatch.setenv("TRADING_MODES", "cfm")
        monkeypatch.setenv("COINBASE_ENABLE_INTX_PERPS", "0")

        ctx = PreflightContext()

        assert ctx.requires_trade_permission() is True

    def test_requires_trade_permission_for_intx_only_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DRY_RUN", raising=False)
        monkeypatch.delenv("PAPER_MODE", raising=False)
        monkeypatch.delenv("PERPS_PAPER", raising=False)
        monkeypatch.delenv("COINBASE_SANDBOX", raising=False)
        monkeypatch.setenv("TRADING_MODES", "intx_perps")
        monkeypatch.setenv("COINBASE_ENABLE_INTX_PERPS", "1")

        ctx = PreflightContext()

        assert ctx.requires_trade_permission() is True

    def test_requires_trade_permission_false_when_intx_only_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DRY_RUN", raising=False)
        monkeypatch.delenv("PAPER_MODE", raising=False)
        monkeypatch.delenv("PERPS_PAPER", raising=False)
        monkeypatch.delenv("COINBASE_SANDBOX", raising=False)
        monkeypatch.setenv("TRADING_MODES", "intx_perps")
        monkeypatch.setenv("COINBASE_ENABLE_INTX_PERPS", "0")

        ctx = PreflightContext()

        assert ctx.requires_trade_permission() is False
