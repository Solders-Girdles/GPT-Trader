"""Tests for CredentialValidator service."""

import os
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.preflight.validation_result import (
    ValidationSeverity,
)
from gpt_trader.tui.services.credential_validator import (
    MODE_REQUIREMENTS,
    CredentialValidator,
)


class TestModeRequirements:
    """Tests for MODE_REQUIREMENTS configuration."""

    def test_demo_requires_nothing(self):
        """Demo mode should not require any credentials."""
        req = MODE_REQUIREMENTS["demo"]
        assert req["requires_credentials"] is False
        assert req["requires_view"] is False
        assert req["requires_trade"] is False

    def test_read_only_requires_view(self):
        """Read-only mode requires view but not trade."""
        req = MODE_REQUIREMENTS["read_only"]
        assert req["requires_credentials"] is True
        assert req["requires_view"] is True
        assert req["requires_trade"] is False

    def test_paper_requires_view(self):
        """Paper mode requires view but not trade."""
        req = MODE_REQUIREMENTS["paper"]
        assert req["requires_credentials"] is True
        assert req["requires_view"] is True
        assert req["requires_trade"] is False

    def test_live_requires_trade(self):
        """Live mode requires both view and trade."""
        req = MODE_REQUIREMENTS["live"]
        assert req["requires_credentials"] is True
        assert req["requires_view"] is True
        assert req["requires_trade"] is True


class TestCredentialValidator:
    """Tests for CredentialValidator class."""

    @pytest.mark.asyncio
    async def test_demo_mode_skips_validation(self):
        """Demo mode should pass without any credentials."""
        validator = CredentialValidator()
        result = await validator.validate_for_mode("demo")

        assert result.valid_for_mode is True
        assert result.mode == "demo"
        assert len(result.findings) == 1
        assert result.findings[0].severity == ValidationSeverity.SUCCESS
        assert "does not require" in result.findings[0].message

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    async def test_missing_credentials_fails(self):
        """Modes requiring credentials should fail if none configured."""
        # Clear any existing credential env vars
        for key in list(os.environ.keys()):
            if "COINBASE" in key:
                del os.environ[key]

        validator = CredentialValidator()
        result = await validator.validate_for_mode("paper")

        assert result.valid_for_mode is False
        assert result.credentials_configured is False
        assert result.has_errors is True

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {
            "COINBASE_CDP_API_KEY": "organizations/123/apiKeys/456",
            "COINBASE_CDP_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
        },
        clear=True,
    )
    async def test_valid_key_format(self):
        """Valid key format should pass format validation."""
        validator = CredentialValidator()
        await validator.validate_for_mode("demo")  # Demo mode, validates format check path

        # Demo skips validation, but let's test format directly
        from gpt_trader.preflight.validation_result import CredentialValidationResult

        direct_result = CredentialValidationResult(mode="paper")
        validator._validate_key_format(direct_result)

        assert direct_result.credentials_configured is True
        assert direct_result.key_format_valid is True
        # Should have 2 success findings (api key + private key)
        success_findings = [
            f for f in direct_result.findings if f.severity == ValidationSeverity.SUCCESS
        ]
        assert len(success_findings) == 2

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {
            "COINBASE_CDP_API_KEY": "invalid-key-format",
            "COINBASE_CDP_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
        },
        clear=True,
    )
    async def test_invalid_key_format(self):
        """Invalid key format should fail format validation with legacy detection."""
        validator = CredentialValidator()

        from gpt_trader.preflight.validation_result import CredentialValidationResult

        result = CredentialValidationResult(mode="paper")
        validator._validate_key_format(result)

        assert result.credentials_configured is True
        assert result.key_format_valid is False
        # Should have error for API key format - now detects legacy format
        error_findings = [f for f in result.findings if f.severity == ValidationSeverity.ERROR]
        assert len(error_findings) == 1
        assert "Legacy API key format detected" in error_findings[0].message

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {
            "COINBASE_CDP_API_KEY": "organizations/123/apiKeys/456",
            "COINBASE_CDP_PRIVATE_KEY": "invalid-private-key",
        },
        clear=True,
    )
    async def test_invalid_private_key_format(self):
        """Invalid private key format should fail validation with specific error."""
        validator = CredentialValidator()

        from gpt_trader.preflight.validation_result import CredentialValidationResult

        result = CredentialValidationResult(mode="paper")
        validator._validate_key_format(result)

        # Should have error for private key format - now detects non-PEM format
        error_findings = [f for f in result.findings if f.severity == ValidationSeverity.ERROR]
        assert len(error_findings) == 1
        assert "Private key not in PEM format" in error_findings[0].message

    def test_evaluate_mode_compatibility_live_without_trade(self):
        """Live mode should fail without trade permission."""
        validator = CredentialValidator()

        from gpt_trader.preflight.validation_result import (
            CredentialValidationResult,
            PermissionDetails,
        )

        result = CredentialValidationResult(mode="live")
        result.connectivity_ok = True
        result.permissions = PermissionDetails(
            can_trade=False,
            can_view=True,
        )

        validator._evaluate_mode_compatibility(result, MODE_REQUIREMENTS["live"])

        assert result.valid_for_mode is False
        assert result.has_errors is True
        error_messages = [f.message for f in result.blocking_issues]
        assert any("trade permission" in msg.lower() for msg in error_messages)

    def test_evaluate_mode_compatibility_paper_with_view_only(self):
        """Paper mode should pass with view-only key."""
        validator = CredentialValidator()

        from gpt_trader.preflight.validation_result import (
            CredentialValidationResult,
            PermissionDetails,
        )

        result = CredentialValidationResult(mode="paper")
        result.connectivity_ok = True
        result.permissions = PermissionDetails(
            can_trade=False,
            can_view=True,
        )

        validator._evaluate_mode_compatibility(result, MODE_REQUIREMENTS["paper"])

        assert result.valid_for_mode is True
        assert result.has_errors is False

    def test_evaluate_mode_compatibility_read_only_with_view(self):
        """Read-only mode should pass with view permission."""
        validator = CredentialValidator()

        from gpt_trader.preflight.validation_result import (
            CredentialValidationResult,
            PermissionDetails,
        )

        result = CredentialValidationResult(mode="read_only")
        result.connectivity_ok = True
        result.permissions = PermissionDetails(
            can_trade=False,
            can_view=True,
        )

        validator._evaluate_mode_compatibility(result, MODE_REQUIREMENTS["read_only"])

        assert result.valid_for_mode is True

    def test_evaluate_mode_compatibility_without_connectivity(self):
        """Mode compatibility should fail without connectivity."""
        validator = CredentialValidator()

        from gpt_trader.preflight.validation_result import (
            CredentialValidationResult,
            PermissionDetails,
        )

        result = CredentialValidationResult(mode="paper")
        result.connectivity_ok = False
        result.permissions = PermissionDetails(
            can_trade=True,
            can_view=True,
        )

        validator._evaluate_mode_compatibility(result, MODE_REQUIREMENTS["paper"])

        # Without connectivity, should not be valid even with permissions
        assert result.valid_for_mode is False

    def test_get_mode_requirements(self):
        """get_mode_requirements should return correct requirements."""
        assert CredentialValidator.get_mode_requirements("demo") == MODE_REQUIREMENTS["demo"]
        assert CredentialValidator.get_mode_requirements("live") == MODE_REQUIREMENTS["live"]
        # Unknown mode should default to demo
        assert CredentialValidator.get_mode_requirements("unknown") == MODE_REQUIREMENTS["demo"]


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
    async def test_connectivity_success(self):
        """Successful connectivity should set connectivity_ok."""
        validator = CredentialValidator()

        # Mock the client creation and API calls - patch where imported from
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
    async def test_jwt_generation_failure(self):
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


class TestKeyTypeDetection:
    """Tests for key type detection."""

    def test_detect_cdp_es256_key(self):
        """Valid CDP ES256 key should be detected correctly."""
        validator = CredentialValidator()
        result = validator._detect_key_type(
            "organizations/abc123/apiKeys/xyz789",
            "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
        )
        assert result["key_type"] == "cdp"
        assert result["algorithm"] == "ES256"
        assert result["is_cdp_format"] is True
        assert result["is_legacy"] is False
        assert result["is_ed25519"] is False
        assert len(result["issues"]) == 0

    def test_detect_legacy_uuid_key(self):
        """Legacy UUID format key should be detected."""
        validator = CredentialValidator()
        result = validator._detect_key_type(
            "12345678-1234-1234-1234-123456789abc",
            "some-api-secret",
        )
        assert result["key_type"] == "legacy_uuid"
        assert result["is_legacy"] is True
        assert result["is_cdp_format"] is False
        assert len(result["issues"]) >= 1
        assert "Legacy UUID-format" in result["issues"][0]

    def test_detect_legacy_short_key(self):
        """Legacy short format key should be detected."""
        validator = CredentialValidator()
        result = validator._detect_key_type(
            "abc123shortkey",
            "some-api-secret",
        )
        assert result["key_type"] == "legacy_short"
        assert result["is_legacy"] is True
        assert result["is_cdp_format"] is False
        assert "Legacy short-format" in result["issues"][0]

    def test_detect_ed25519_key(self):
        """Ed25519 key should be detected and flagged."""
        validator = CredentialValidator()
        result = validator._detect_key_type(
            "organizations/abc/apiKeys/xyz",
            "-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----",
        )
        assert result["algorithm"] == "Ed25519"
        assert result["is_ed25519"] is True
        assert result["is_cdp_format"] is True
        assert "Ed25519" in result["issues"][0]
        assert "ES256" in result["suggestions"][0]

    def test_detect_openssh_ed25519_key(self):
        """OpenSSH Ed25519 key should be detected."""
        validator = CredentialValidator()
        result = validator._detect_key_type(
            "organizations/abc/apiKeys/xyz",
            "-----BEGIN OPENSSH PRIVATE KEY-----\ntest\n-----END OPENSSH PRIVATE KEY-----",
        )
        assert result["algorithm"] == "Ed25519_OpenSSH"
        assert result["is_ed25519"] is True
        assert "OpenSSH Ed25519" in result["issues"][0]

    def test_detect_rsa_key(self):
        """RSA key should be flagged as wrong algorithm."""
        validator = CredentialValidator()
        result = validator._detect_key_type(
            "organizations/abc/apiKeys/xyz",
            "-----BEGIN RSA PRIVATE KEY-----\ntest\n-----END RSA PRIVATE KEY-----",
        )
        assert result["algorithm"] == "RSA"
        assert result["is_ed25519"] is False
        assert "RSA" in result["issues"][0]

    def test_detect_non_pem_secret(self):
        """Non-PEM secret should be flagged."""
        validator = CredentialValidator()
        result = validator._detect_key_type(
            "organizations/abc/apiKeys/xyz",
            "not-a-pem-formatted-key",
        )
        assert result["algorithm"] == "none_pem"
        assert "not in PEM format" in result["issues"][0]

    def test_detect_empty_keys(self):
        """Empty keys should return unknown type."""
        validator = CredentialValidator()
        result = validator._detect_key_type("", "")
        assert result["key_type"] == "unknown"
        assert result["algorithm"] == "unknown"
        assert result["is_cdp_format"] is False
        assert result["is_legacy"] is False
