"""
Simple Authentication Module
Reads API keys from environment variables.
"""
import os
from dataclasses import dataclass

@dataclass
class APIKey:
    name: str
    private_key: str

class SimpleAuth:
    @staticmethod
    def get_coinbase_credentials() -> APIKey:
        """Retrieves Coinbase credentials from environment variables."""
        name = os.getenv("COINBASE_API_KEY_NAME")
        key = os.getenv("COINBASE_PRIVATE_KEY")

        if not name or not key:
            raise ValueError("COINBASE_API_KEY_NAME and COINBASE_PRIVATE_KEY must be set in environment.")

        # Handle newlines in PEM key if they are escaped
        key = key.replace("\\n", "\n")

        return APIKey(name=name, private_key=key)

    @staticmethod
    def get_auth_headers(method: str, path: str, body: str = "") -> dict[str, str]:
        """
        Generates authentication headers for Coinbase Advanced Trade API.
        This uses the CDP SDK or manual JWT generation if we want to avoid the SDK.
        For now, we'll assume we use the official logic or a simplified version.
        """
        # Note: In a real "no abstraction" world, we might put the JWT generation here.
        # But `coinbase.auth` might still exist. Let's check if we can reuse the existing simple auth logic
        # or if we need to implement a tiny JWT generator here.
        pass
