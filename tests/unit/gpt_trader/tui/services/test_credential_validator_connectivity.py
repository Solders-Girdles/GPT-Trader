from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.tui.services.credential_validator import CredentialValidator


class TestCredentialValidatorConnectivity:
    """Tests for connectivity validation (mocked)."""

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {
            "COINBASE_CDP_API_KEY": "organizations/123/apiKeys/456",
            "COINBASE_CDP_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
        },
        clear=True,
    )
    async def test_connectivity_success(self) -> None:
        """Successful connectivity should set connectivity_ok."""
        validator = CredentialValidator()

        with (
            patch(
                "gpt_trader.features.brokerages.coinbase.client.CoinbaseClient"
            ) as mock_client_class,
            patch(
                "gpt_trader.features.brokerages.coinbase.client.create_cdp_jwt_auth"
            ) as mock_auth,
        ):
            mock_auth_instance = MagicMock()
            mock_auth.return_value = mock_auth_instance

            mock_client = MagicMock()
            mock_client.get_time.return_value = {"iso": "2024-12-06T00:00:00Z"}
            mock_client.get_key_permissions.return_value = {
                "can_trade": True,
                "can_view": True,
                "portfolio_type": "SPOT",
                "portfolio_uuid": "uuid-123",
            }
            mock_client_class.return_value = mock_client

            result = await validator.validate_for_mode("paper")

            assert result.connectivity_ok is True
            assert result.jwt_generation_ok is True
            assert result.valid_for_mode is True

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {
            "COINBASE_CDP_API_KEY": "organizations/123/apiKeys/456",
            "COINBASE_CDP_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
        },
        clear=True,
    )
    async def test_jwt_generation_failure(self) -> None:
        """JWT generation failure should set connectivity_ok to False."""
        validator = CredentialValidator()

        with patch(
            "gpt_trader.features.brokerages.coinbase.client.create_cdp_jwt_auth"
        ) as mock_auth:
            mock_auth.side_effect = Exception("Invalid private key")

            from gpt_trader.preflight.validation_result import CredentialValidationResult

            result = CredentialValidationResult(mode="paper")
            result.credentials_configured = True
            result.key_format_valid = True

            await validator._validate_connectivity(result)

            assert result.jwt_generation_ok is False
            assert result.connectivity_ok is False
            assert result.has_errors is True
