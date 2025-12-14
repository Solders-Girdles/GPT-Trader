"""
Simple Authentication Module for Coinbase.
Provides JWT-based authentication for Coinbase Advanced Trade API.
"""

import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import jwt


@dataclass
class APIKey:
    name: str
    private_key: str


class CoinbaseAuth:
    def __init__(self, api_key: str, api_secret: str, passphrase: str | None = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase


class CDPJWTAuth(CoinbaseAuth):
    """CDP JWT Authentication for Coinbase Advanced Trade API."""

    def __init__(self, api_key: str, private_key: str):
        super().__init__(api_key, private_key)  # private_key used as api_secret
        # For JWT, api_key is actually key_name
        self.key_name = api_key
        self.private_key = self._normalize_key(private_key)

    def _normalize_key(self, key: str) -> str:
        return key.replace("\\n", "\n")

    def generate_jwt(self, method: str, path: str) -> str:
        """Generate a JWT token for the given HTTP method and path."""
        parsed = urlparse(path)
        request_path = parsed.path or path
        request_path = request_path if request_path.startswith("/") else f"/{request_path}"
        uri = f"{method} api.coinbase.com{request_path}"

        current_time = int(time.time())
        claims = {
            "sub": self.key_name,
            "iss": "cdp",
            "nbf": current_time,
            "exp": current_time + 120,
            "uri": uri,
        }

        headers = {"kid": self.key_name, "nonce": secrets.token_hex()}

        return jwt.encode(claims, self.private_key, algorithm="ES256", headers=headers)

    def get_headers(self, method: str, path: str, body: Any = None) -> dict[str, str]:
        """Generate HTTP headers with JWT authorization."""
        token = self.generate_jwt(method, path)
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


class SimpleAuth(CoinbaseAuth):
    def __init__(self, key_name: str, private_key: str):
        super().__init__(key_name, private_key)
        self.key_name = key_name
        self.private_key = self._normalize_key(private_key)

    def _normalize_key(self, key: str) -> str:
        return key.replace("\\n", "\n")

    def generate_jwt(self, method: str, path: str) -> str:
        parsed = urlparse(path)
        request_path = parsed.path or path
        request_path = request_path if request_path.startswith("/") else f"/{request_path}"
        uri = f"{method} api.coinbase.com{request_path}"

        current_time = int(time.time())
        claims = {
            "sub": self.key_name,
            "iss": "cdp",
            "nbf": current_time,
            "exp": current_time + 120,
            "uri": uri,
        }

        headers = {"kid": self.key_name, "nonce": secrets.token_hex()}

        return jwt.encode(claims, self.private_key, algorithm="ES256", headers=headers)

    def get_headers(self, method: str, path: str, body: Any = None) -> dict[str, str]:
        token = self.generate_jwt(method, path)
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def get_auth() -> SimpleAuth:
    """Get authentication from environment variables.

    Returns:
        SimpleAuth configured with credentials from environment.

    Raises:
        EnvironmentError: If credentials are missing in production mode.
            Set ENV=development or ENV=test to use test fallback credentials.
    """
    import os

    name = os.getenv("COINBASE_API_KEY_NAME")
    key = os.getenv("COINBASE_PRIVATE_KEY")
    if not name or not key:
        env_mode = os.getenv("ENV", "").lower()
        if env_mode in ("development", "test", "testing"):
            return SimpleAuth("test", "test_key")
        raise OSError(
            "Missing COINBASE_API_KEY_NAME or COINBASE_PRIVATE_KEY. "
            "Set these environment variables or use ENV=development for test mode."
        )
    return SimpleAuth(name, key)


def create_cdp_jwt_auth(api_key: str, private_key: str) -> CDPJWTAuth:
    """Factory function to create a CDP JWT auth instance."""
    return CDPJWTAuth(api_key, private_key)
