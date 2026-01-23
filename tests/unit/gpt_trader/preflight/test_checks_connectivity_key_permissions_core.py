"""Tests for key permissions preflight checks (core scenarios)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from gpt_trader.preflight.checks.connectivity import check_key_permissions
from gpt_trader.preflight.core import PreflightCheck


def _set_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str], *, clear: bool = True) -> None:
    if clear:
        for key in list(os.environ):
            monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


class TestCheckKeyPermissionsCore:
    """Test key permissions checks (core scenarios)."""

    def test_skips_when_remote_checks_bypassed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should skip and succeed when remote checks are bypassed."""
        checker = PreflightCheck(profile="dev")

        env = {"COINBASE_PREFLIGHT_SKIP_REMOTE": "1"}
        with monkeypatch.context() as mp:
            _set_env(mp, env, clear=True)
            result = check_key_permissions(checker)

        assert result is True
        assert any("bypassed" in s for s in checker.successes)

    def test_fails_when_client_build_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should fail when CDP client cannot be built."""
        checker = PreflightCheck(profile="prod")

        env = {"COINBASE_PREFLIGHT_FORCE_REMOTE": "1"}
        with monkeypatch.context() as mp:
            _set_env(mp, env, clear=True)
            result = check_key_permissions(checker)

        assert result is False

    def test_passes_with_full_permissions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should pass when key has trade and view permissions."""
        checker = PreflightCheck(profile="prod")

        mock_client = MagicMock()
        mock_auth = MagicMock()
        mock_client.get_key_permissions.return_value = {
            "can_trade": True,
            "can_view": True,
            "portfolio_uuid": "uuid-123",
        }

        env = {"COINBASE_PREFLIGHT_FORCE_REMOTE": "1"}
        with monkeypatch.context() as mp:
            _set_env(mp, env, clear=True)
            mp.setattr(
                checker,
                "_build_cdp_client",
                MagicMock(return_value=(mock_client, mock_auth)),
            )
            result = check_key_permissions(checker)

        assert result is True
        assert any("view permission" in s for s in checker.successes)
        assert any("trade permission" in s for s in checker.successes)

    def test_fails_without_trade_permission_when_derivatives_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should fail when key lacks trade permission and derivatives are enabled."""
        checker = PreflightCheck(profile="prod")

        mock_client = MagicMock()
        mock_auth = MagicMock()
        mock_client.get_key_permissions.return_value = {
            "can_trade": False,
            "can_view": True,
            "portfolio_type": "INTX",
        }

        env = {
            "COINBASE_PREFLIGHT_FORCE_REMOTE": "1",
            "COINBASE_ENABLE_INTX_PERPS": "1",
        }
        with monkeypatch.context() as mp:
            _set_env(mp, env, clear=True)
            mp.setattr(
                checker,
                "_build_cdp_client",
                MagicMock(return_value=(mock_client, mock_auth)),
            )
            result = check_key_permissions(checker)

        assert result is False
        assert any("missing trade permission" in e for e in checker.errors)

    def test_fails_without_view_permission(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should fail when key lacks view permission."""
        checker = PreflightCheck(profile="prod")

        mock_client = MagicMock()
        mock_auth = MagicMock()
        mock_client.get_key_permissions.return_value = {
            "can_trade": True,
            "can_view": False,
        }

        env = {"COINBASE_PREFLIGHT_FORCE_REMOTE": "1"}
        with monkeypatch.context() as mp:
            _set_env(mp, env, clear=True)
            mp.setattr(
                checker,
                "_build_cdp_client",
                MagicMock(return_value=(mock_client, mock_auth)),
            )
            result = check_key_permissions(checker)

        assert result is False
        assert any("missing portfolio view permission" in e for e in checker.errors)

    def test_passes_with_view_only_key_when_live_orders_not_intended(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Should pass with view-only key when no live orders are intended."""
        checker = PreflightCheck(profile="prod", verbose=True)

        mock_client = MagicMock()
        mock_auth = MagicMock()
        mock_client.get_key_permissions.return_value = {
            "can_trade": False,
            "can_view": True,
            "portfolio_uuid": "uuid-123",
        }

        env = {
            "COINBASE_PREFLIGHT_FORCE_REMOTE": "1",
            "COINBASE_ENABLE_INTX_PERPS": "0",
            "DRY_RUN": "1",
        }
        with monkeypatch.context() as mp:
            _set_env(mp, env, clear=True)
            mp.setattr(
                checker,
                "_build_cdp_client",
                MagicMock(return_value=(mock_client, mock_auth)),
            )
            result = check_key_permissions(checker)

        assert result is True
        assert any("view permission" in s for s in checker.successes)
        # Trade permission missing is logged as info, not error
        captured = capsys.readouterr()
        assert "view-only" in captured.out

    def test_fails_with_view_only_key_when_live_spot_orders_intended(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should fail with view-only key when live spot orders are intended."""
        checker = PreflightCheck(profile="prod")

        mock_client = MagicMock()
        mock_auth = MagicMock()
        mock_client.get_key_permissions.return_value = {
            "can_trade": False,
            "can_view": True,
        }

        env = {
            "COINBASE_PREFLIGHT_FORCE_REMOTE": "1",
            "COINBASE_ENABLE_INTX_PERPS": "0",
            "TRADING_MODES": "spot",
        }
        with monkeypatch.context() as mp:
            _set_env(mp, env, clear=True)
            mp.setattr(
                checker,
                "_build_cdp_client",
                MagicMock(return_value=(mock_client, mock_auth)),
            )
            result = check_key_permissions(checker)

        assert result is False
        assert any("missing trade permission" in e for e in checker.errors)

    def test_warns_when_portfolio_uuid_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should warn when portfolio UUID is not returned."""
        checker = PreflightCheck(profile="prod")

        mock_client = MagicMock()
        mock_auth = MagicMock()
        mock_client.get_key_permissions.return_value = {
            "can_trade": True,
            "can_view": True,
            # No portfolio_uuid
        }

        env = {"COINBASE_PREFLIGHT_FORCE_REMOTE": "1"}
        with monkeypatch.context() as mp:
            _set_env(mp, env, clear=True)
            mp.setattr(
                checker,
                "_build_cdp_client",
                MagicMock(return_value=(mock_client, mock_auth)),
            )
            result = check_key_permissions(checker)

        assert result is True
        assert any("Portfolio UUID not returned" in w for w in checker.warnings)

    def test_prints_section_header(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Should print section header."""
        checker = PreflightCheck(profile="dev")

        env = {"COINBASE_PREFLIGHT_SKIP_REMOTE": "1"}
        with monkeypatch.context() as mp:
            _set_env(mp, env, clear=True)
            check_key_permissions(checker)

        captured = capsys.readouterr()
        assert "KEY PERMISSIONS" in captured.out
