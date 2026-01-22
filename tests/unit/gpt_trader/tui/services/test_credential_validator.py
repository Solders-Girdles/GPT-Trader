from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import gpt_trader.features.brokerages.coinbase.client as client_module
from gpt_trader.preflight.validation_result import (
    CredentialValidationResult,
    ValidationSeverity,
)
from gpt_trader.tui.services.credential_validator import CredentialValidator


class TestCredentialValidatorKeyFormatAndConfiguration:
    """Tests for credential configuration and key format validation."""

    @pytest.mark.asyncio
    async def test_demo_mode_skips_validation(self) -> None:
        """Demo mode should pass without any credentials."""
        validator = CredentialValidator()
        result = await validator.validate_for_mode("demo")

        assert result.valid_for_mode is True
        assert result.mode == "demo"
        assert len(result.findings) == 1
        assert result.findings[0].severity == ValidationSeverity.SUCCESS
        assert "does not require" in result.findings[0].message

    @pytest.mark.asyncio
    async def test_missing_credentials_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Modes requiring credentials should fail if none configured."""
        monkeypatch.delenv("COINBASE_CDP_API_KEY", raising=False)
        monkeypatch.delenv("COINBASE_CDP_PRIVATE_KEY", raising=False)

        validator = CredentialValidator()
        result = await validator.validate_for_mode("paper")

        assert result.valid_for_mode is False
        assert result.credentials_configured is False
        assert result.has_errors is True

    def test_valid_key_format(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Valid key format should pass format validation."""
        monkeypatch.setenv("COINBASE_CDP_API_KEY", "organizations/123/apiKeys/456")
        monkeypatch.setenv(
            "COINBASE_CDP_PRIVATE_KEY",
            "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
        )

        validator = CredentialValidator()

        direct_result = CredentialValidationResult(mode="paper")
        validator._validate_key_format(direct_result)

        assert direct_result.credentials_configured is True
        assert direct_result.key_format_valid is True
        success_findings = [
            f for f in direct_result.findings if f.severity == ValidationSeverity.SUCCESS
        ]
        assert len(success_findings) == 2

    def test_invalid_key_format(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Invalid key format should fail validation with legacy detection."""
        monkeypatch.setenv("COINBASE_CDP_API_KEY", "invalid-key-format")
        monkeypatch.setenv(
            "COINBASE_CDP_PRIVATE_KEY",
            "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
        )

        validator = CredentialValidator()

        result = CredentialValidationResult(mode="paper")
        validator._validate_key_format(result)

        assert result.credentials_configured is True
        assert result.key_format_valid is False
        error_findings = [f for f in result.findings if f.severity == ValidationSeverity.ERROR]
        assert len(error_findings) == 1
        assert "Legacy API key format detected" in error_findings[0].message

    def test_invalid_private_key_format(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Invalid private key format should fail validation with specific error."""
        monkeypatch.setenv("COINBASE_CDP_API_KEY", "organizations/123/apiKeys/456")
        monkeypatch.setenv("COINBASE_CDP_PRIVATE_KEY", "invalid-private-key")

        validator = CredentialValidator()

        result = CredentialValidationResult(mode="paper")
        validator._validate_key_format(result)

        error_findings = [f for f in result.findings if f.severity == ValidationSeverity.ERROR]
        assert len(error_findings) == 1
        assert "Private key not in PEM format" in error_findings[0].message


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

        result = CredentialValidationResult(mode="paper")
        result.credentials_configured = True
        result.key_format_valid = True

        await validator._validate_connectivity(result)

        assert result.jwt_generation_ok is False
        assert result.connectivity_ok is False
        assert result.has_errors is True
