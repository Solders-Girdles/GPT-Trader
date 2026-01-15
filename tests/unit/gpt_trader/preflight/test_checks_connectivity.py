"""Tests for API connectivity preflight checks."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from gpt_trader.preflight.checks.connectivity import (
    check_api_connectivity,
    check_key_permissions,
)
from gpt_trader.preflight.core import PreflightCheck


class TestCheckApiConnectivity:
    """Test API connectivity checks."""

    def test_skips_when_remote_checks_bypassed(self) -> None:
        """Should skip and succeed when remote checks are bypassed."""
        checker = PreflightCheck(profile="dev")

        # Use env var to skip remote checks
        env = {"COINBASE_PREFLIGHT_SKIP_REMOTE": "1"}
        with patch.dict(os.environ, env, clear=True):
            result = check_api_connectivity(checker)

        assert result is True
        assert any("bypassed" in s for s in checker.successes)

    def test_fails_when_client_build_fails(self) -> None:
        """Should fail when CDP client cannot be built."""
        checker = PreflightCheck(profile="prod")

        # Force remote checks but no credentials
        env = {
            "COINBASE_PREFLIGHT_FORCE_REMOTE": "1",
            "COINBASE_ENABLE_DERIVATIVES": "1",
        }
        with patch.dict(os.environ, env, clear=True):
            result = check_api_connectivity(checker)

        assert result is False

    def test_fails_on_jwt_generation_error(self) -> None:
        """Should fail when JWT generation fails."""
        checker = PreflightCheck(profile="prod")

        mock_client = MagicMock()
        mock_auth = MagicMock()
        mock_auth.generate_jwt.side_effect = Exception("JWT error")

        env = {
            "COINBASE_PREFLIGHT_FORCE_REMOTE": "1",
            "COINBASE_ENABLE_DERIVATIVES": "1",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, mock_auth)),
        ):
            result = check_api_connectivity(checker)

        assert result is False
        assert any("JWT generation failed" in e for e in checker.errors)

    def test_passes_with_successful_api_calls(self) -> None:
        """Should pass when all API calls succeed."""
        checker = PreflightCheck(profile="prod")

        mock_client = MagicMock()
        mock_auth = MagicMock()
        mock_client.get_time.return_value = {"iso": "2024-01-01T00:00:00Z"}
        mock_client.get_accounts.return_value = [{"id": "acc1"}]
        mock_client.list_products.return_value = [{"product_id": "BTC-PERP"}]

        env = {
            "COINBASE_PREFLIGHT_FORCE_REMOTE": "1",
            "COINBASE_ENABLE_DERIVATIVES": "1",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, mock_auth)),
        ):
            result = check_api_connectivity(checker)

        assert result is True
        assert any("Server time: OK" in s for s in checker.successes)
        assert any("Accounts: OK" in s for s in checker.successes)
        assert any("Products: OK" in s for s in checker.successes)

    def test_warns_on_empty_response(self) -> None:
        """Should warn when API returns empty response."""
        checker = PreflightCheck(profile="prod")

        mock_client = MagicMock()
        mock_auth = MagicMock()
        mock_client.get_time.return_value = None  # Empty response
        mock_client.get_accounts.return_value = [{"id": "acc1"}]
        mock_client.list_products.return_value = [{"product_id": "BTC-PERP"}]

        env = {
            "COINBASE_PREFLIGHT_FORCE_REMOTE": "1",
            "COINBASE_ENABLE_DERIVATIVES": "1",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, mock_auth)),
        ):
            check_api_connectivity(checker)

        assert any("Empty response" in w for w in checker.warnings)

    def test_fails_on_api_error(self) -> None:
        """Should fail when API call raises exception."""
        checker = PreflightCheck(profile="prod")

        mock_client = MagicMock()
        mock_auth = MagicMock()
        mock_client.get_time.side_effect = Exception("Network error")
        mock_client.get_accounts.return_value = [{"id": "acc1"}]
        mock_client.list_products.return_value = [{"product_id": "BTC-PERP"}]

        env = {"COINBASE_PREFLIGHT_FORCE_REMOTE": "1"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, mock_auth)),
        ):
            result = check_api_connectivity(checker)

        assert result is False

    def test_fails_when_no_perps_found(self) -> None:
        """Should fail when no perpetual products found."""
        checker = PreflightCheck(profile="prod")

        mock_client = MagicMock()
        mock_auth = MagicMock()
        mock_client.get_time.return_value = {"iso": "2024-01-01T00:00:00Z"}
        mock_client.get_accounts.return_value = [{"id": "acc1"}]
        mock_client.list_products.return_value = [
            {"product_id": "BTC-USD"},  # Not a PERP
            {"product_id": "ETH-USD"},
        ]

        env = {
            "COINBASE_PREFLIGHT_FORCE_REMOTE": "1",
            "COINBASE_ENABLE_DERIVATIVES": "1",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, mock_auth)),
        ):
            result = check_api_connectivity(checker)

        assert result is False
        assert any("No perpetual products" in e for e in checker.errors)

    def test_logs_perp_count_when_found(self) -> None:
        """Should log count of perpetual products found."""
        checker = PreflightCheck(profile="prod")

        mock_client = MagicMock()
        mock_auth = MagicMock()
        mock_client.get_time.return_value = {"iso": "2024-01-01T00:00:00Z"}
        mock_client.get_accounts.return_value = [{"id": "acc1"}]
        mock_client.list_products.return_value = [
            {"product_id": "BTC-PERP"},
            {"product_id": "ETH-PERP"},
            {"product_id": "SOL-PERP"},
        ]

        env = {"COINBASE_PREFLIGHT_FORCE_REMOTE": "1"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, mock_auth)),
        ):
            result = check_api_connectivity(checker)

        assert result is True
        assert any("3 perpetual products" in s for s in checker.successes)

    def test_logs_perp_names_when_verbose(self, capsys: pytest.CaptureFixture) -> None:
        """Should log perpetual product names when verbose."""
        checker = PreflightCheck(profile="prod", verbose=True)

        mock_client = MagicMock()
        mock_auth = MagicMock()
        mock_client.get_time.return_value = {"iso": "2024-01-01T00:00:00Z"}
        mock_client.get_accounts.return_value = [{"id": "acc1"}]
        mock_client.list_products.return_value = [
            {"product_id": "BTC-PERP"},
            {"product_id": "ETH-PERP"},
        ]

        env = {"COINBASE_PREFLIGHT_FORCE_REMOTE": "1"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, mock_auth)),
        ):
            check_api_connectivity(checker)

        captured = capsys.readouterr()
        assert "BTC-PERP" in captured.out

    def test_prints_section_header(self, capsys: pytest.CaptureFixture) -> None:
        """Should print section header."""
        checker = PreflightCheck(profile="dev")

        env = {"COINBASE_PREFLIGHT_SKIP_REMOTE": "1"}
        with patch.dict(os.environ, env, clear=True):
            check_api_connectivity(checker)

        captured = capsys.readouterr()
        assert "API CONNECTIVITY" in captured.out


class TestCheckKeyPermissions:
    """Test key permissions checks."""

    def test_skips_when_remote_checks_bypassed(self) -> None:
        """Should skip and succeed when remote checks are bypassed."""
        checker = PreflightCheck(profile="dev")

        env = {"COINBASE_PREFLIGHT_SKIP_REMOTE": "1"}
        with patch.dict(os.environ, env, clear=True):
            result = check_key_permissions(checker)

        assert result is True
        assert any("bypassed" in s for s in checker.successes)

    def test_fails_when_client_build_fails(self) -> None:
        """Should fail when CDP client cannot be built."""
        checker = PreflightCheck(profile="prod")

        env = {"COINBASE_PREFLIGHT_FORCE_REMOTE": "1"}
        with patch.dict(os.environ, env, clear=True):
            result = check_key_permissions(checker)

        assert result is False

    def test_passes_with_full_permissions(self) -> None:
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
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, mock_auth)),
        ):
            result = check_key_permissions(checker)

        assert result is True
        assert any("view permission" in s for s in checker.successes)
        assert any("trade permission" in s for s in checker.successes)

    def test_fails_without_trade_permission_when_derivatives_enabled(self) -> None:
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
            "COINBASE_ENABLE_DERIVATIVES": "1",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, mock_auth)),
        ):
            result = check_key_permissions(checker)

        assert result is False
        assert any("missing trade permission" in e for e in checker.errors)

    def test_fails_without_view_permission(self) -> None:
        """Should fail when key lacks view permission."""
        checker = PreflightCheck(profile="prod")

        mock_client = MagicMock()
        mock_auth = MagicMock()
        mock_client.get_key_permissions.return_value = {
            "can_trade": True,
            "can_view": False,
        }

        env = {"COINBASE_PREFLIGHT_FORCE_REMOTE": "1"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, mock_auth)),
        ):
            result = check_key_permissions(checker)

        assert result is False
        assert any("missing portfolio view permission" in e for e in checker.errors)

    def test_passes_with_view_only_key_when_live_orders_not_intended(
        self, capsys: pytest.CaptureFixture
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
            "COINBASE_ENABLE_DERIVATIVES": "0",
            "DRY_RUN": "1",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, mock_auth)),
        ):
            result = check_key_permissions(checker)

        assert result is True
        assert any("view permission" in s for s in checker.successes)
        # Trade permission missing is logged as info, not error
        captured = capsys.readouterr()
        assert "view-only" in captured.out

    def test_fails_with_view_only_key_when_live_spot_orders_intended(self) -> None:
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
            "COINBASE_ENABLE_DERIVATIVES": "0",
            "TRADING_MODES": "spot",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, mock_auth)),
        ):
            result = check_key_permissions(checker)

        assert result is False
        assert any("missing trade permission" in e for e in checker.errors)

    def test_warns_when_portfolio_uuid_missing(self) -> None:
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
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, mock_auth)),
        ):
            result = check_key_permissions(checker)

        assert result is True
        assert any("Portfolio UUID not returned" in w for w in checker.warnings)

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

    def test_retries_on_transient_errors(self) -> None:
        """Should retry on transient network errors."""
        checker = PreflightCheck(profile="prod")

        mock_client = MagicMock()
        mock_auth = MagicMock()
        # Fail first two times, succeed on third
        mock_client.get_key_permissions.side_effect = [
            URLError("Network error"),
            TimeoutError("Timeout"),
            {"can_trade": True, "can_view": True},
        ]

        env = {"COINBASE_PREFLIGHT_FORCE_REMOTE": "1"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, mock_auth)),
            patch("time.sleep"),  # Skip actual sleep
        ):
            result = check_key_permissions(checker)

        assert result is True
        assert mock_client.get_key_permissions.call_count == 3

    def test_fails_after_max_retries(self) -> None:
        """Should fail after max retry attempts."""
        checker = PreflightCheck(profile="prod")

        mock_client = MagicMock()
        mock_auth = MagicMock()
        mock_client.get_key_permissions.side_effect = URLError("Network error")

        env = {"COINBASE_PREFLIGHT_FORCE_REMOTE": "1"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, mock_auth)),
            patch("time.sleep"),
        ):
            result = check_key_permissions(checker)

        assert result is False
        assert any("failed after retries" in e for e in checker.errors)

    def test_fails_on_non_transient_error(self) -> None:
        """Should fail immediately on non-transient errors."""
        checker = PreflightCheck(profile="prod")

        mock_client = MagicMock()
        mock_auth = MagicMock()
        mock_client.get_key_permissions.side_effect = ValueError("Bad data")

        env = {"COINBASE_PREFLIGHT_FORCE_REMOTE": "1"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, mock_auth)),
        ):
            result = check_key_permissions(checker)

        assert result is False
        assert any("Failed to fetch" in e for e in checker.errors)
        # Should not retry on non-transient errors
        assert mock_client.get_key_permissions.call_count == 1

    def test_fails_on_empty_permissions_response(self) -> None:
        """Should fail when permissions response is empty/None."""
        checker = PreflightCheck(profile="prod")

        mock_client = MagicMock()
        mock_auth = MagicMock()
        # When get_key_permissions returns None, it becomes {} via `or {}`
        # and can_trade/can_view will be False
        mock_client.get_key_permissions.return_value = None

        env = {"COINBASE_PREFLIGHT_FORCE_REMOTE": "1"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(checker, "_build_cdp_client", return_value=(mock_client, mock_auth)),
        ):
            result = check_key_permissions(checker)

        assert result is False
        # Empty response leads to missing view permission error
        assert any("missing portfolio view permission" in e for e in checker.errors)

    def test_prints_section_header(self, capsys: pytest.CaptureFixture) -> None:
        """Should print section header."""
        checker = PreflightCheck(profile="dev")

        env = {"COINBASE_PREFLIGHT_SKIP_REMOTE": "1"}
        with patch.dict(os.environ, env, clear=True):
            check_key_permissions(checker)

        captured = capsys.readouterr()
        assert "KEY PERMISSIONS" in captured.out
