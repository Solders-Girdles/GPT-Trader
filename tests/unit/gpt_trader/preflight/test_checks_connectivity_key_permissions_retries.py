"""Tests for key permissions preflight checks (retry and error handling)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from gpt_trader.preflight.checks.connectivity import check_key_permissions
from gpt_trader.preflight.core import PreflightCheck


class TestCheckKeyPermissionsRetries:
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
