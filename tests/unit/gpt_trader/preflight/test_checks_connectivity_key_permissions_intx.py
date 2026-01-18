"""Tests for key permissions preflight checks (INTX gating)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.preflight.checks.connectivity import check_key_permissions
from gpt_trader.preflight.core import PreflightCheck


class TestCheckKeyPermissionsIntx:
    def test_passes_intx_check_when_derivatives_enabled(self) -> None:
        """Should pass INTX check when derivatives enabled and portfolio is INTX."""
        checker = PreflightCheck(profile="prod")

        mock_client = MagicMock()
        mock_auth = MagicMock()
        mock_client.get_key_permissions.return_value = {
            "can_trade": True,
            "can_view": True,
            "portfolio_type": "INTX",
        }

        env = {
            "COINBASE_PREFLIGHT_FORCE_REMOTE": "1",
            "COINBASE_ENABLE_DERIVATIVES": "1",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, mock_auth)),
        ):
            result = check_key_permissions(checker)

        assert result is True
        assert any("INTX portfolio detected" in s for s in checker.successes)

    def test_fails_intx_check_when_wrong_portfolio_type(self) -> None:
        """Should fail when derivatives enabled but not INTX portfolio."""
        checker = PreflightCheck(profile="prod")

        mock_client = MagicMock()
        mock_auth = MagicMock()
        mock_client.get_key_permissions.return_value = {
            "can_trade": True,
            "can_view": True,
            "portfolio_type": "SPOT",
        }

        env = {
            "COINBASE_PREFLIGHT_FORCE_REMOTE": "1",
            "COINBASE_ENABLE_DERIVATIVES": "1",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, mock_auth)),
        ):
            result = check_key_permissions(checker)

        assert result is False
        assert any("INTX gating check failed" in e for e in checker.errors)

    def test_logs_intx_available_when_disabled(self, capsys: pytest.CaptureFixture) -> None:
        """Should log INTX available when derivatives disabled but INTX portfolio."""
        checker = PreflightCheck(profile="prod", verbose=True)

        mock_client = MagicMock()
        mock_auth = MagicMock()
        mock_client.get_key_permissions.return_value = {
            "can_trade": True,
            "can_view": True,
            "portfolio_type": "INTX",
        }

        env = {
            "COINBASE_PREFLIGHT_FORCE_REMOTE": "1",
            "COINBASE_ENABLE_DERIVATIVES": "0",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, mock_auth)),
        ):
            result = check_key_permissions(checker)

        assert result is True
        captured = capsys.readouterr()
        assert "INTX portfolio available" in captured.out
