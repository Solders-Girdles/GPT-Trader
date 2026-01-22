from __future__ import annotations

from gpt_trader.preflight.validation_result import (
    CredentialValidationResult,
    PermissionDetails,
)
from gpt_trader.tui.services.credential_validator import (
    MODE_REQUIREMENTS,
    CredentialValidator,
)


class TestModeRequirements:
    """Tests for MODE_REQUIREMENTS configuration."""

    def test_demo_requires_nothing(self) -> None:
        """Demo mode should not require any credentials."""
        req = MODE_REQUIREMENTS["demo"]
        assert req["requires_credentials"] is False
        assert req["requires_view"] is False
        assert req["requires_trade"] is False

    def test_read_only_requires_view(self) -> None:
        """Read-only mode requires view but not trade."""
        req = MODE_REQUIREMENTS["read_only"]
        assert req["requires_credentials"] is True
        assert req["requires_view"] is True
        assert req["requires_trade"] is False

    def test_paper_requires_view(self) -> None:
        """Paper mode requires view but not trade."""
        req = MODE_REQUIREMENTS["paper"]
        assert req["requires_credentials"] is True
        assert req["requires_view"] is True
        assert req["requires_trade"] is False

    def test_live_requires_trade(self) -> None:
        """Live mode requires both view and trade."""
        req = MODE_REQUIREMENTS["live"]
        assert req["requires_credentials"] is True
        assert req["requires_view"] is True
        assert req["requires_trade"] is True


class TestModeCompatibility:
    """Tests for mode compatibility evaluation."""

    def test_evaluate_mode_compatibility_live_without_trade(self) -> None:
        """Live mode should fail without trade permission."""
        validator = CredentialValidator()

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


class TestKeyTypeDetection:
    """Tests for key type detection."""

    def test_detect_cdp_es256_key(self) -> None:
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

    def test_detect_legacy_uuid_key(self) -> None:
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

    def test_detect_legacy_short_key(self) -> None:
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

    def test_detect_ed25519_key(self) -> None:
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

    def test_detect_openssh_ed25519_key(self) -> None:
        """OpenSSH Ed25519 key should be detected."""
        validator = CredentialValidator()
        result = validator._detect_key_type(
            "organizations/abc/apiKeys/xyz",
            "-----BEGIN OPENSSH PRIVATE KEY-----\ntest\n-----END OPENSSH PRIVATE KEY-----",
        )
        assert result["algorithm"] == "Ed25519_OpenSSH"
        assert result["is_ed25519"] is True
        assert "OpenSSH Ed25519" in result["issues"][0]

    def test_detect_rsa_key(self) -> None:
        """RSA key should be flagged as wrong algorithm."""
        validator = CredentialValidator()
        result = validator._detect_key_type(
            "organizations/abc/apiKeys/xyz",
            "-----BEGIN RSA PRIVATE KEY-----\ntest\n-----END RSA PRIVATE KEY-----",
        )
        assert result["algorithm"] == "RSA"
        assert result["is_ed25519"] is False
        assert "RSA" in result["issues"][0]

    def test_detect_non_pem_secret(self) -> None:
        """Non-PEM secret should be flagged."""
        validator = CredentialValidator()
        result = validator._detect_key_type(
            "organizations/abc/apiKeys/xyz",
            "not-a-pem-formatted-key",
        )
        assert result["algorithm"] == "none_pem"
        assert "not in PEM format" in result["issues"][0]

    def test_detect_empty_keys(self) -> None:
        """Empty keys should return unknown type."""
        validator = CredentialValidator()
        result = validator._detect_key_type("", "")
        assert result["key_type"] == "unknown"
        assert result["algorithm"] == "unknown"
        assert result["is_cdp_format"] is False
        assert result["is_legacy"] is False
