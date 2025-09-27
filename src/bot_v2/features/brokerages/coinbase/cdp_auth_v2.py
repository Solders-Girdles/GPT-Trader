"""
Coinbase Developer Platform (CDP) JWT authentication V2.

Aligns JWT format with Coinbase's official Python SDK (coinbase-advanced-py):
- iss: "cdp"
- sub: full API key resource name (e.g., organizations/.../apiKeys/...)
- uri: "<METHOD> <HOST><PATH>" (e.g., "GET api.coinbase.com/api/v3/brokerage/accounts")
- headers: kid = full API key resource name, nonce = random hex

Notes:
- No audience claim is required.
- Host in the uri claim must include the hostname, without scheme.
"""

from __future__ import annotations

import json
import time
import hashlib
import secrets
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import Dict, Optional, Any

import logging

logger = logging.getLogger(__name__)


@dataclass
class CDPAuthV2:
    """CDP API authentication using JWT with EC private key - SDK compatible version."""
    
    api_key_name: str  # Format: organizations/.../apiKeys/...
    private_key_pem: str  # EC private key in PEM format
    base_host: str = "api.coinbase.com"  # Host used in JWT uri claim (no scheme)
    
    def generate_jwt(self, method: str, path: str) -> str:
        """Generate JWT token matching official SDK format.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., /api/v3/brokerage/accounts)
            
        Returns:
            JWT token string
        """
        try:
            import jwt
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.backends import default_backend
        except ImportError:
            raise ImportError(
                "CDP authentication requires 'pyjwt' and 'cryptography' packages. "
                "Install with: pip install pyjwt cryptography"
            )
        
        # Load the EC private key
        try:
            private_key = serialization.load_pem_private_key(
                self.private_key_pem.encode() if isinstance(self.private_key_pem, str) else self.private_key_pem,
                password=None,
                backend=default_backend()
            )
        except Exception as e:
            logger.error("Failed to load private key", exc_info=True)
            raise ValueError("Invalid private key") from e
        
        # Create JWT claims matching SDK format
        current_time = int(time.time())

        # Build uri with hostname (no scheme), e.g.:
        #   "GET api.coinbase.com/api/v3/brokerage/accounts"
        path_only = path if path.startswith("/") else f"/{path}"
        uri = f"{method.upper()} {self.base_host}{path_only}"

        claims = {
            "sub": self.api_key_name,
            "iss": "cdp",
            "nbf": current_time,
            "exp": current_time + 120,  # Token expires in 2 minutes
            "uri": uri,
        }

        # JWT headers matching SDK
        # Generate random nonce as hex string (SDK format)
        nonce = secrets.token_hex()

        # KID should be the full API key resource name (match SDK)
        headers = {
            "kid": self.api_key_name,
            "nonce": nonce,
        }
        
        # Generate and return JWT
        token = jwt.encode(
            claims,
            private_key,
            algorithm="ES256",
            headers=headers
        )
        
        # jwt.encode returns string in newer versions, bytes in older
        if isinstance(token, bytes):
            token = token.decode('utf-8')
            
        fingerprint = hashlib.sha256(token.encode('utf-8')).hexdigest()[:8]
        logger.debug(
            "Generated JWT (SDK-compatible) for %s (fingerprint=%s)",
            path_only,
            fingerprint,
        )
        logger.debug("  URI claim: %s", uri)

        return token
    
    def sign(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Generate authorization headers for CDP API request.
        
        Args:
            method: HTTP method
            path: API path
            body: Request body (not used in JWT, but kept for compatibility)
            
        Returns:
            Dictionary of headers including Authorization
        """
        jwt_token = self.generate_jwt(method, path)
        
        return {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json",
        }


def create_cdp_auth_v2(api_key_name: str, private_key_pem: str, base_url: str | None = None) -> CDPAuthV2:
    """Factory function to create CDP auth instance with SDK-compatible format.
    
    Args:
        api_key_name: Full API key name from CDP
        private_key_pem: EC private key in PEM format
        base_url: Optional base URL (e.g., "https://api.coinbase.com").
                  If provided, its hostname will be used in the JWT uri claim.
                  Defaults to "api.coinbase.com".
        
        Returns:
            CDPAuthV2 instance
        """
    # Clean up private key format if needed
    # Normalize common formatting issues
    if isinstance(private_key_pem, bytes):
        private_key_pem = private_key_pem.decode()
    private_key_pem = private_key_pem.strip()
    private_key_pem = private_key_pem.replace("\r\n", "\n").replace("\r", "\n")
    if not private_key_pem.startswith("-----BEGIN"):
        # If it's escaped, unescape it
        private_key_pem = private_key_pem.replace("\\n", "\n")
    
    # Determine host for uri claim
    host = "api.coinbase.com"
    if base_url:
        try:
            parsed = urlparse(base_url)
            if parsed.netloc:
                host = parsed.netloc
            else:
                # If a bare host like "api.coinbase.com" is provided
                host = base_url.replace("https://", "").replace("http://", "").strip("/")
        except Exception:
            pass

    return CDPAuthV2(
        api_key_name=api_key_name,
        private_key_pem=private_key_pem,
        base_host=host,
    )
