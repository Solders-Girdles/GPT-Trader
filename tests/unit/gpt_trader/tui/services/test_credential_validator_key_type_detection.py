from __future__ import annotations

from gpt_trader.tui.services.credential_validator import CredentialValidator


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
