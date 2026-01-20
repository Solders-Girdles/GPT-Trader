from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import gpt_trader.features.brokerages.coinbase.client as client_module
from gpt_trader.tui.services.credential_validator import CredentialValidator


class TestCredentialValidatorConnectivity:
    """Tests for connectivity validation (mocked)."""

    @pytest.mark.asyncio
    async def test_connectivity_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Successful connectivity should set connectivity_ok."""
        monkeypatch.setenv("COINBASE_CDP_API_KEY", "organizations/123/apiKeys/456")
        monkeypatch.setenv(
            "COINBASE_CDP_PRIVATE_KEY",
            "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
        )

        validator = CredentialValidator()

        mock_auth_instance = MagicMock()
        mock_auth = MagicMock(return_value=mock_auth_instance)
        monkeypatch.setattr(client_module, "create_cdp_jwt_auth", mock_auth)

        mock_client = MagicMock()
        mock_client.get_time.return_value = {"iso": "2024-12-06T00:00:00Z"}
        mock_client.get_key_permissions.return_value = {
            "can_trade": True,
            "can_view": True,
            "portfolio_type": "SPOT",
            "portfolio_uuid": "uuid-123",
        }
        mock_client_class = MagicMock(return_value=mock_client)
        monkeypatch.setattr(client_module, "CoinbaseClient", mock_client_class)

        result = await validator.validate_for_mode("paper")

        assert result.connectivity_ok is True
        assert result.jwt_generation_ok is True
        assert result.valid_for_mode is True

    @pytest.mark.asyncio
    async def test_jwt_generation_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """JWT generation failure should set connectivity_ok to False."""
        monkeypatch.setenv("COINBASE_CDP_API_KEY", "organizations/123/apiKeys/456")
        monkeypatch.setenv(
            "COINBASE_CDP_PRIVATE_KEY",
            "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
        )

        validator = CredentialValidator()

        mock_auth = MagicMock(side_effect=Exception("Invalid private key"))
        monkeypatch.setattr(client_module, "create_cdp_jwt_auth", mock_auth)

        from gpt_trader.preflight.validation_result import CredentialValidationResult

        result = CredentialValidationResult(mode="paper")
        result.credentials_configured = True
        result.key_format_valid = True

        await validator._validate_connectivity(result)

        assert result.jwt_generation_ok is False
        assert result.connectivity_ok is False
        assert result.has_errors is True
