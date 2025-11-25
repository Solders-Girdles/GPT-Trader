"""Tests for security/simple_auth.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gpt_trader.security.simple_auth import APIKey, SimpleAuth

# ============================================================
# Test: APIKey dataclass
# ============================================================


class TestAPIKeyDataclass:
    """Tests for APIKey dataclass."""

    def test_api_key_stores_fields(self) -> None:
        """Test that APIKey stores name and private_key correctly."""
        key = APIKey(name="test-key-name", private_key="test-private-key")

        assert key.name == "test-key-name"
        assert key.private_key == "test-private-key"

    def test_api_key_equality(self) -> None:
        """Test that APIKey instances with same values are equal."""
        key1 = APIKey(name="key", private_key="secret")
        key2 = APIKey(name="key", private_key="secret")

        assert key1 == key2

    def test_api_key_repr(self) -> None:
        """Test that APIKey has a useful repr."""
        key = APIKey(name="my-key", private_key="secret")

        assert "my-key" in repr(key)


# ============================================================
# Test: get_coinbase_credentials
# ============================================================


class TestGetCoinbaseCredentials:
    """Tests for SimpleAuth.get_coinbase_credentials method."""

    @patch.dict(
        "os.environ",
        {
            "COINBASE_API_KEY_NAME": "test-api-key",
            "COINBASE_PRIVATE_KEY": "test-private-key",
        },
    )
    def test_get_coinbase_credentials_success(self) -> None:
        """Test successful credential retrieval."""
        result = SimpleAuth.get_coinbase_credentials()

        assert isinstance(result, APIKey)
        assert result.name == "test-api-key"
        assert result.private_key == "test-private-key"

    @patch.dict("os.environ", {"COINBASE_PRIVATE_KEY": "test-key"}, clear=True)
    def test_get_coinbase_credentials_missing_name(self) -> None:
        """Test that missing API key name raises ValueError."""
        with pytest.raises(ValueError, match="COINBASE_API_KEY_NAME"):
            SimpleAuth.get_coinbase_credentials()

    @patch.dict("os.environ", {"COINBASE_API_KEY_NAME": "test-name"}, clear=True)
    def test_get_coinbase_credentials_missing_key(self) -> None:
        """Test that missing private key raises ValueError."""
        with pytest.raises(ValueError, match="COINBASE_PRIVATE_KEY"):
            SimpleAuth.get_coinbase_credentials()

    @patch.dict("os.environ", {}, clear=True)
    def test_get_coinbase_credentials_both_missing(self) -> None:
        """Test that missing both credentials raises ValueError."""
        with pytest.raises(ValueError):
            SimpleAuth.get_coinbase_credentials()

    @patch.dict(
        "os.environ",
        {
            "COINBASE_API_KEY_NAME": "test-api-key",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\\nMIHcAgEB\\n-----END EC PRIVATE KEY-----",
        },
    )
    def test_get_coinbase_credentials_newline_escaping(self) -> None:
        """Test that escaped newlines in PEM key are converted to real newlines."""
        result = SimpleAuth.get_coinbase_credentials()

        # The \\n should be converted to \n
        assert "\\n" not in result.private_key
        assert "\n" in result.private_key
        assert (
            "-----BEGIN EC PRIVATE KEY-----\nMIHcAgEB\n-----END EC PRIVATE KEY-----"
            == result.private_key
        )

    @patch.dict(
        "os.environ",
        {
            "COINBASE_API_KEY_NAME": "test-api-key",
            "COINBASE_PRIVATE_KEY": "simple-key-no-escapes",
        },
    )
    def test_get_coinbase_credentials_no_newlines(self) -> None:
        """Test credentials without escaped newlines work correctly."""
        result = SimpleAuth.get_coinbase_credentials()

        assert result.private_key == "simple-key-no-escapes"

    @patch.dict(
        "os.environ",
        {
            "COINBASE_API_KEY_NAME": "",
            "COINBASE_PRIVATE_KEY": "test-key",
        },
    )
    def test_get_coinbase_credentials_empty_name(self) -> None:
        """Test that empty API key name raises ValueError."""
        with pytest.raises(ValueError):
            SimpleAuth.get_coinbase_credentials()

    @patch.dict(
        "os.environ",
        {
            "COINBASE_API_KEY_NAME": "test-name",
            "COINBASE_PRIVATE_KEY": "",
        },
    )
    def test_get_coinbase_credentials_empty_key(self) -> None:
        """Test that empty private key raises ValueError."""
        with pytest.raises(ValueError):
            SimpleAuth.get_coinbase_credentials()


# ============================================================
# Test: get_auth_headers
# ============================================================


class TestGetAuthHeaders:
    """Tests for SimpleAuth.get_auth_headers method."""

    def test_get_auth_headers_returns_none(self) -> None:
        """Test that get_auth_headers currently returns None (unimplemented)."""
        result = SimpleAuth.get_auth_headers("GET", "/api/v1/accounts")

        # Current implementation just has 'pass', so returns None
        assert result is None

    def test_get_auth_headers_with_body(self) -> None:
        """Test get_auth_headers with body parameter."""
        result = SimpleAuth.get_auth_headers("POST", "/api/v1/orders", body='{"symbol": "BTC"}')

        assert result is None

    def test_get_auth_headers_default_body(self) -> None:
        """Test get_auth_headers uses empty string as default body."""
        result = SimpleAuth.get_auth_headers("DELETE", "/api/v1/orders/123")

        assert result is None
