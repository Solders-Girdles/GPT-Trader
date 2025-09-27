"""
Coinbase Developer Platform (CDP) JWT authentication.

This module handles JWT-based authentication for the newer CDP API,
which uses EC private keys instead of HMAC signatures.
"""

from __future__ import annotations

import json
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Any

import logging

logger = logging.getLogger(__name__)


@dataclass
class CDPAuth:
    """CDP API authentication using JWT with EC private key."""
    
    api_key_name: str  # Format: organizations/.../apiKeys/...
    private_key_pem: str  # EC private key in PEM format
    
    def generate_jwt(self, method: str, path: str) -> str:
        """Generate JWT token for CDP API authentication.
        
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
        
        # Create JWT claims
        current_time = int(time.time())
        claims = {
            "sub": self.api_key_name,
            "iss": "coinbase-cloud",
            "nbf": current_time,
            "exp": current_time + 120,  # Token expires in 2 minutes
            "aud": ["retail_rest_api_proxy"],
            "uri": f"{method.upper()} {path}"
        }
        
        # JWT headers
        headers = {
            "alg": "ES256",
            "kid": self.api_key_name,
            "typ": "JWT",
            "nonce": str(int(time.time() * 1000))
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
        logger.debug("Generated JWT for %s %s (fingerprint=%s)", method, path, fingerprint)
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


def create_cdp_auth(api_key_name: str, private_key_pem: str) -> CDPAuth:
    """Factory function to create CDP auth instance.
    
    Args:
        api_key_name: Full API key name from CDP
        private_key_pem: EC private key in PEM format
        
    Returns:
        CDPAuth instance
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
    
    return CDPAuth(
        api_key_name=api_key_name,
        private_key_pem=private_key_pem
    )
