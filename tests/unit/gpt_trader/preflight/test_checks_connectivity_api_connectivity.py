"""Tests for API connectivity preflight checks."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.preflight.checks.connectivity import check_api_connectivity
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
