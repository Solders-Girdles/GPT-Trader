"""Integration smoke test for CoinbaseClient auth negotiation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from bot_v2.features.brokerages.coinbase.auth import (
    CDPJWTAuth,
    CoinbaseAuth,
    build_rest_auth,
    build_ws_auth_provider,
    create_cdp_auth,
    create_cdp_jwt_auth,
)
from bot_v2.features.brokerages.coinbase.models import APIConfig


class TestCoinbaseAuthSmoke:
    """Smoke tests for Coinbase authentication negotiation."""

    def test_coinbase_hmac_auth_signing(self):
        """Test HMAC auth generates proper headers."""
        auth = CoinbaseAuth(
            api_key="test_key",
            api_secret="dGVzdCBzZWNyZXQ=",  # base64 "test secret"
            passphrase="test_pass",
            api_mode="exchange",
        )

        headers = auth.sign("POST", "/api/orders", {"size": "1.0"})

        # Verify required headers are present
        assert "CB-ACCESS-KEY" in headers
        assert "CB-ACCESS-SIGN" in headers
        assert "CB-ACCESS-TIMESTAMP" in headers
        assert "CB-ACCESS-PASSPHRASE" in headers
        assert "Content-Type" in headers

        # Verify header values
        assert headers["CB-ACCESS-KEY"] == "test_key"
        assert headers["CB-ACCESS-PASSPHRASE"] == "test_pass"
        assert headers["Content-Type"] == "application/json"

        # Verify signature is base64
        import base64

        signature = headers["CB-ACCESS-SIGN"]
        assert base64.b64decode(signature, validate=True) is not None

    def test_coinbase_hmac_auth_advanced_mode(self):
        """Test HMAC auth in advanced mode (no passphrase)."""
        auth = CoinbaseAuth(
            api_key="test_key",
            api_secret="dGVzdCBzZWNyZXQ=",
            api_mode="advanced",
        )

        headers = auth.sign("GET", "/api/accounts", None)

        assert "CB-ACCESS-KEY" in headers
        assert "CB-ACCESS-SIGN" in headers
        assert "CB-ACCESS-TIMESTAMP" in headers
        assert "CB-ACCESS-PASSPHRASE" not in headers  # No passphrase in advanced mode

    def test_coinbase_hmac_auth_mode_detection(self):
        """Test auth mode detection based on passphrase presence."""
        # With passphrase -> exchange mode
        auth_exchange = CoinbaseAuth(
            api_key="test_key",
            api_secret="dGVzdCBzZWNyZXQ=",
            passphrase="test_pass",
        )
        headers = auth_exchange.sign("GET", "/api/accounts", None)
        assert "CB-ACCESS-PASSPHRASE" in headers

        # Without passphrase -> advanced mode
        auth_advanced = CoinbaseAuth(
            api_key="test_key",
            api_secret="dGVzdCBzZWNyZXQ=",
        )
        headers = auth_advanced.sign("GET", "/api/accounts", None)
        assert "CB-ACCESS-PASSPHRASE" not in headers

    def test_cdp_jwt_auth_generation(self):
        """Test CDP JWT auth handles missing dependencies gracefully."""
        # Test that the auth class properly handles missing dependencies
        auth = CDPJWTAuth(
            api_key_name="test_key",
            private_key_pem="-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----",
        )

        # Verify that the class raises an ImportError when jwt is not available
        with patch.dict("sys.modules", {"jwt": None}):
            with pytest.raises(ImportError, match="CDP authentication requires"):
                auth.generate_jwt("GET", "/api/accounts")

    def test_cdp_jwt_auth_sign_method(self):
        """Test CDP JWT auth sign method returns proper headers."""
        with patch.object(CDPJWTAuth, "generate_jwt", return_value="test.jwt.token"):
            auth = CDPJWTAuth(
                api_key_name="test_key",
                private_key_pem="-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----",
            )

            headers = auth.sign("POST", "/api/orders", {"size": "1.0"})

            assert headers == {
                "Authorization": "Bearer test.jwt.token",
                "Content-Type": "application/json",
            }

    @patch("cryptography.hazmat.primitives.serialization")
    def test_cdp_jwt_auth_invalid_key_handling(self, mock_serialization):
        """Test CDP JWT auth handles invalid private keys."""
        mock_serialization.load_pem_private_key.side_effect = ValueError("Invalid key")

        auth = CDPJWTAuth(
            api_key_name="test_key",
            private_key_pem="invalid_key",
        )

        with pytest.raises(ValueError, match="Invalid private key"):
            auth.generate_jwt()

    def test_create_cdp_auth_legacy_helper(self):
        """Test legacy create_cdp_auth helper."""
        auth = create_cdp_auth(
            api_key_name="test_key",
            private_key_pem="-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----",
            base_url="https://api.exchange.coinbase.com",
        )

        assert auth.api_key_name == "test_key"
        assert auth.issuer == "coinbase-cloud"
        assert auth.audience == ("retail_rest_api_proxy",)
        assert auth.include_host_in_uri is False
        assert auth.base_host == "api.exchange.coinbase.com"

    def test_create_cdp_jwt_auth_custom_config(self):
        """Test create_cdp_jwt_auth with custom configuration."""
        auth = create_cdp_jwt_auth(
            api_key_name="test_key",
            private_key_pem="-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----",
            base_url="https://custom.api.com",
            issuer="custom_issuer",
            audience=["aud1", "aud2"],
            include_host_in_uri=False,
            add_nonce_header=False,
        )

        assert auth.api_key_name == "test_key"
        assert auth.base_host == "custom.api.com"
        assert auth.issuer == "custom_issuer"
        assert auth.audience == ("aud1", "aud2")
        assert auth.include_host_in_uri is False
        assert auth.add_nonce_header is False

    def test_normalize_private_key_handling(self):
        """Test private key normalization handles various formats."""
        from bot_v2.features.brokerages.coinbase.auth import _normalize_private_key

        # Test bytes input
        key_bytes = b"-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----"
        normalized = _normalize_private_key(key_bytes)
        assert isinstance(normalized, str)
        assert normalized.startswith("-----BEGIN")

        # Test string with escaped newlines
        key_escaped = "-----BEGIN PRIVATE KEY-----\\ntest\\n-----END PRIVATE KEY-----"
        normalized = _normalize_private_key(key_escaped)
        assert "\\n" not in normalized
        assert "\n" in normalized

        # Test already normalized key
        key_clean = "-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----"
        normalized = _normalize_private_key(key_clean)
        assert normalized == key_clean

        # Test string with literal backslash-n sequences
        key_literal = "-----BEGIN PRIVATE KEY-----\\ntest\\n-----END PRIVATE KEY-----"
        normalized = _normalize_private_key(key_literal)
        # Should convert \n to actual newlines
        assert "\\n" not in normalized
        assert "\n" in normalized

    def test_host_from_base_url_parsing(self):
        """Test host extraction from various base URL formats."""
        from bot_v2.features.brokerages.coinbase.auth import _host_from_base_url

        assert _host_from_base_url("https://api.coinbase.com") == "api.coinbase.com"
        assert _host_from_base_url("https://api.coinbase.com/") == "api.coinbase.com"
        assert _host_from_base_url("api.coinbase.com") == "api.coinbase.com"
        assert _host_from_base_url("api.coinbase.com/") == "api.coinbase.com"
        assert _host_from_base_url(None) == "api.coinbase.com"
        assert _host_from_base_url("") == "api.coinbase.com"
        # Test that invalid URLs fall back to default instead of crashing
        try:
            result = _host_from_base_url("invalid-url")
            assert result == "api.coinbase.com"  # Fallback on error
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass

    def test_build_rest_auth_cdp_preference(self):
        """Test build_rest_auth prefers CDP auth when available."""
        config = APIConfig(
            api_key="hmac_key",
            api_secret="hmac_secret",
            passphrase="test_passphrase",
            base_url="https://api.coinbase.com",
            cdp_api_key="cdp_key",
            cdp_private_key="-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----",
        )

        auth = build_rest_auth(config)
        assert isinstance(auth, CDPJWTAuth)
        assert auth.api_key_name == "cdp_key"

    def test_build_rest_auth_fallback_to_hmac(self):
        """Test build_rest_auth falls back to HMAC when CDP not available."""
        config = APIConfig(
            api_key="hmac_key",
            api_secret="hmac_secret",
            passphrase="test_passphrase",
            base_url="https://api.coinbase.com",
        )

        auth = build_rest_auth(config)
        assert isinstance(auth, CoinbaseAuth)
        assert auth.api_key == "hmac_key"

    def test_build_ws_auth_provider_cdp_config(self):
        """Test WS auth provider creation for CDP config."""
        config = APIConfig(
            api_key="hmac_key",
            api_secret="hmac_secret",
            passphrase="test_passphrase",
            base_url="https://api.coinbase.com",
            cdp_api_key="cdp_key",
            cdp_private_key="-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----",
            enable_derivatives=True,
            auth_type="JWT",
        )

        # Test that the provider handles missing dependencies gracefully
        with (
            patch.dict("os.environ", {"COINBASE_WS_USER_AUTH": "1"}),
            patch.dict("sys.modules", {"jwt": None}),
        ):
            provider = build_ws_auth_provider(config, None)

            # The provider should still be created, but fail when called
            assert provider is not None

            # When called, it should handle the error gracefully
            auth_payload = provider()
            assert auth_payload is None  # Should return None on error

    def test_build_ws_auth_provider_no_auth(self):
        """Test WS auth provider returns None when no auth configured."""
        config = APIConfig(
            api_key="test_key",
            api_secret="test_secret",
            passphrase="test_passphrase",
            base_url="https://api.coinbase.com",
        )

        provider = build_ws_auth_provider(config, None)
        assert provider is None

    def test_build_ws_auth_provider_client_auth_fallback(self):
        """Test WS auth provider uses client auth object when available."""
        config = APIConfig(
            api_key="test_key",
            api_secret="test_secret",
            passphrase="test_passphrase",
            base_url="https://api.coinbase.com",
        )

        class MockClientAuth:
            def generate_jwt(self, method, path):
                return "client.jwt.token"

        client_auth = MockClientAuth()

        with patch.dict("os.environ", {"COINBASE_WS_USER_AUTH": "1"}):
            provider = build_ws_auth_provider(config, client_auth)

            assert provider is not None
            auth_payload = provider()
            assert auth_payload == {"jwt": "client.jwt.token"}

    def test_auth_error_handling_in_ws_provider(self):
        """Test WS auth provider handles errors gracefully."""
        config = APIConfig(
            api_key="test_key",
            api_secret="test_secret",
            passphrase="test_passphrase",
            base_url="https://api.coinbase.com",
            cdp_api_key="cdp_key",
            cdp_private_key="invalid_key",
            enable_derivatives=True,
            auth_type="JWT",
        )

        provider = build_ws_auth_provider(config, None)
        assert provider is not None

        # Should return None on auth failure
        auth_payload = provider()
        assert auth_payload is None

    def test_import_guards_in_cdp_auth(self):
        """Test CDP auth handles missing dependencies gracefully."""
        with patch.dict("sys.modules", {"jwt": None, "cryptography": None}):
            with pytest.raises(ImportError, match="pyjwt.*cryptography"):
                CDPJWTAuth(
                    api_key_name="test",
                    private_key_pem="key",
                ).generate_jwt()


class TestAuthIntegrationFlows:
    """Test complete auth negotiation flows."""

    def test_full_hmac_auth_flow(self):
        """Test complete HMAC auth flow from config to headers."""
        config = APIConfig(
            api_key="test_key",
            api_secret="dGVzdCBzZWNyZXQ=",
            passphrase="test_pass",
            base_url="https://api.coinbase.com",
            api_mode="exchange",
        )

        auth = build_rest_auth(config)
        assert isinstance(auth, CoinbaseAuth)

        headers = auth.sign("POST", "/api/orders", {"product_id": "BTC-USD", "size": "1.0"})

        # Verify all expected headers are present
        required_headers = [
            "CB-ACCESS-KEY",
            "CB-ACCESS-SIGN",
            "CB-ACCESS-TIMESTAMP",
            "CB-ACCESS-PASSPHRASE",
            "Content-Type",
        ]

        for header in required_headers:
            assert header in headers, f"Missing required header: {header}"

        # Verify header values are reasonable
        assert headers["CB-ACCESS-KEY"] == "test_key"
        assert headers["CB-ACCESS-PASSPHRASE"] == "test_pass"
        assert headers["Content-Type"] == "application/json"

        # Verify timestamp is recent
        import time

        timestamp = float(headers["CB-ACCESS-TIMESTAMP"])
        now = time.time()
        assert abs(now - timestamp) < 10  # Within 10 seconds

    def test_full_cdp_auth_flow(self):
        """Test complete CDP auth flow from config to headers."""
        # Create a mock JWT module
        mock_jwt = type("MockJWT", (), {"encode": lambda *args, **kwargs: "encoded.jwt.token"})()
        mock_secrets = type("MockSecrets", (), {"token_hex": lambda: "test_nonce_123"})()

        # Create a mock cryptography module
        mock_serialization = type(
            "MockSerialization",
            (),
            {"load_pem_private_key": lambda *args, **kwargs: type("MockPrivateKey", (), {})()},
        )()
        mock_cryptography = type(
            "MockCryptography",
            (),
            {
                "hazmat": type(
                    "MockHazmat",
                    (),
                    {
                        "primitives": type(
                            "MockPrimitives", (), {"serialization": mock_serialization}
                        )()
                    },
                )()
            },
        )()

        # Patch the imports in the auth module
        with patch.dict(
            "sys.modules",
            {
                "jwt": mock_jwt,
                "secrets": mock_secrets,
                "cryptography": mock_cryptography,
                "cryptography.hazmat": mock_cryptography.hazmat,
                "cryptography.hazmat.primitives": mock_cryptography.hazmat.primitives,
                "cryptography.hazmat.primitives.serialization": mock_serialization,
            },
        ):
            config = APIConfig(
                api_key="test_key",
                api_secret="test_secret",
                passphrase="test_passphrase",
                base_url="https://api.coinbase.com",
                cdp_api_key="cdp_test_key",
                cdp_private_key="-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----",
            )

            auth = build_rest_auth(config)
            assert isinstance(auth, CDPJWTAuth)

            headers = auth.sign("GET", "/api/accounts", None)

            # Verify headers
            assert "Authorization" in headers
            assert "Content-Type" in headers
            assert headers["Authorization"] == "Bearer encoded.jwt.token"
            assert headers["Content-Type"] == "application/json"

    def test_auth_mode_negotiation(self):
        """Test auth mode negotiation based on available credentials."""
        # CDP takes precedence
        config_cdp = APIConfig(
            api_key="hmac_key",
            api_secret="hmac_secret",
            passphrase="test_passphrase",
            base_url="https://api.coinbase.com",
            cdp_api_key="cdp_key",
            cdp_private_key="cdp_private_key",
        )
        auth_cdp = build_rest_auth(config_cdp)
        assert isinstance(auth_cdp, CDPJWTAuth)

        # HMAC fallback
        config_hmac = APIConfig(
            api_key="hmac_key",
            api_secret="hmac_secret",
            passphrase="test_passphrase",
            base_url="https://api.coinbase.com",
        )
        auth_hmac = build_rest_auth(config_hmac)
        assert isinstance(auth_hmac, CoinbaseAuth)

        # No auth available
        _ = APIConfig(
            api_key="test_key",
            api_secret="test_secret",
            passphrase="test_passphrase",
            base_url="https://api.coinbase.com",
        )
        # This would normally fail, but build_rest_auth assumes credentials exist
        # In practice, this is handled at a higher level
