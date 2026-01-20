from __future__ import annotations

import pytest

from gpt_trader.preflight.validation_result import ValidationSeverity
from gpt_trader.tui.services.credential_validator import (
    MODE_REQUIREMENTS,
    CredentialValidator,
)


class TestCredentialValidatorKeyFormatAndCompatibility:
    """Tests for CredentialValidator format validation and mode compatibility."""

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

        from gpt_trader.preflight.validation_result import CredentialValidationResult

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

        from gpt_trader.preflight.validation_result import CredentialValidationResult

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

        from gpt_trader.preflight.validation_result import CredentialValidationResult

        result = CredentialValidationResult(mode="paper")
        validator._validate_key_format(result)

        error_findings = [f for f in result.findings if f.severity == ValidationSeverity.ERROR]
        assert len(error_findings) == 1
        assert "Private key not in PEM format" in error_findings[0].message

    def test_evaluate_mode_compatibility_live_without_trade(self) -> None:
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

    def test_evaluate_mode_compatibility_paper_with_view_only(self) -> None:
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

    def test_evaluate_mode_compatibility_read_only_with_view(self) -> None:
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

    def test_evaluate_mode_compatibility_without_connectivity(self) -> None:
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

        assert result.valid_for_mode is False

    def test_get_mode_requirements(self) -> None:
        """get_mode_requirements should return correct requirements."""
        assert CredentialValidator.get_mode_requirements("demo") == MODE_REQUIREMENTS["demo"]
        assert CredentialValidator.get_mode_requirements("live") == MODE_REQUIREMENTS["live"]
        assert CredentialValidator.get_mode_requirements("unknown") == MODE_REQUIREMENTS["demo"]
