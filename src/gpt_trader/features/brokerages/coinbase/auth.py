"""
Simple Authentication Module for Coinbase.
Replaces the complex auth hierarchy with a direct JWT generator.
"""

import base64
import hashlib
import hmac  # Use hmac module
import json
import secrets
import time
from dataclasses import dataclass
from typing import Any  # Import Dict

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
    def __init__(self, api_key: str, private_key: str):  # Type hints added
        super().__init__(api_key, private_key)  # private_key used as api_secret
        # For JWT, api_key is actually key_name
        self.key_name = api_key
        self.private_key = self._normalize_key(private_key)

    def _normalize_key(self, key: str) -> str:  # Method was missing
        return key.replace("\\n", "\n")


class HMACAuth(CoinbaseAuth):
    def __init__(self, api_key: str, api_secret: str, passphrase: str | None = None):
        super().__init__(api_key, api_secret, passphrase)
        # Decode api_secret from base64 string to bytes for hmac key
        self._decoded_api_secret = base64.b64decode(self.api_secret)

    def sign(self, method: str, path: str, body: Any = None) -> dict[str, str]:
        timestamp = str(int(time.time()))
        message = timestamp + method + path
        if body is not None:
            message += json.dumps(body)

        signature = hmac.new(
            self._decoded_api_secret, message.encode("utf-8"), hashlib.sha256
        ).hexdigest()  # Corrected hmac usage

        return {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": self.passphrase or "",
            "Content-Type": "application/json",
        }


class SimpleAuth(CoinbaseAuth):  # SimpleAuth should inherit CDPJWTAuth if it's based on JWT
    def __init__(self, key_name: str, private_key: str):
        super().__init__(key_name, private_key)
        self.key_name = key_name
        self.private_key = self._normalize_key(private_key)

    def _normalize_key(self, key: str) -> str:
        return key.replace("\\n", "\n")

    def generate_jwt(self, method: str, path: str) -> str:
        request_path = path if path.startswith("/") else f"/{path}"
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
    import os

    name = os.getenv("COINBASE_API_KEY_NAME")
    key = os.getenv("COINBASE_PRIVATE_KEY")
    if not name or not key:
        # Fallback for tests or if not set, though ideally should raise
        return SimpleAuth("test", "test_key")
    return SimpleAuth(name, key)
