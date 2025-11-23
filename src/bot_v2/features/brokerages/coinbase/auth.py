"""
Simple Authentication Module for Coinbase.
Replaces the complex auth hierarchy with a direct JWT generator.
"""
import time
import secrets
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import jwt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

@dataclass
class APIKey:
    name: str
    private_key: str

class SimpleAuth:
    def __init__(self, key_name: str, private_key: str):
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

        headers = {
            "kid": self.key_name,
            "nonce": secrets.token_hex()
        }

        return jwt.encode(
            claims,
            self.private_key,
            algorithm="ES256",
            headers=headers
        )

    def get_headers(self, method: str, path: str) -> dict[str, str]:
        token = self.generate_jwt(method, path)
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

def get_auth() -> SimpleAuth:
    import os
    name = os.getenv("COINBASE_API_KEY_NAME")
    key = os.getenv("COINBASE_PRIVATE_KEY")
    if not name or not key:
        # Fallback for tests or if not set, though ideally should raise
        return SimpleAuth("test", "test_key")
    return SimpleAuth(name, key)
