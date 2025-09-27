"""
Enhanced CDP JWT authentication with file path support.

This module extends CDP authentication to support both:
1. Private key as string (environment variable)
2. Private key from file path (more secure)
"""

from __future__ import annotations

import os
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Any, Union

import logging
from bot_v2.features.monitor import get_logger

logger = logging.getLogger(__name__)


@dataclass
class EnhancedCDPAuth:
    """Enhanced CDP authentication with file path support."""
    
    api_key_name: str  # Format: organizations/.../apiKeys/...
    private_key_source: Union[str, Path]  # PEM string or file path
    
    def __post_init__(self):
        """Load private key from file if path provided."""
        self._private_key_pem = self._load_private_key()
    
    def _load_private_key(self) -> str:
        """Load private key from string or file.
        
        Returns:
            Private key in PEM format
        """
        # Check if it's a file path
        if isinstance(self.private_key_source, (str, Path)):
            # Convert to Path object
            path = Path(self.private_key_source) if isinstance(self.private_key_source, str) else self.private_key_source
            
            # Check if it looks like a file path (not PEM content)
            if not str(self.private_key_source).startswith("-----BEGIN"):
                # Try to read from file
                if path.exists() and path.is_file():
                    logger.info(f"Loading private key from file: {path}")
                    
                    # Check file permissions (should be 400 or 600)
                    stat = path.stat()
                    mode = oct(stat.st_mode)[-3:]
                    if mode not in ['400', '600']:
                        logger.warning(f"Private key file has loose permissions: {mode} (should be 400 or 600)")
                    
                    # Read the key
                    with open(path, 'r') as f:
                        key_content = f.read()
                    
                    # Validate it looks like a PEM key
                    if "-----BEGIN" in key_content and "-----END" in key_content:
                        return key_content
                    else:
                        raise ValueError(f"File {path} does not contain valid PEM key")
                else:
                    # File doesn't exist, check environment variable
                    env_key = os.getenv("COINBASE_CDP_PRIVATE_KEY")
                    if env_key:
                        logger.warning("Using private key from environment (less secure)")
                        return env_key
                    else:
                        raise ValueError(f"Private key file not found: {path}")
        
        # It's already PEM content
        if "-----BEGIN" in str(self.private_key_source):
            logger.debug("Using private key from string")
            return str(self.private_key_source)
        
        # Last resort: check if it's an environment variable name
        env_value = os.getenv(str(self.private_key_source))
        if env_value:
            logger.info("Loaded private key from environment variable")
            return env_value
        
        raise ValueError("Could not load private key from any source")
    
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
                self._private_key_pem.encode() if isinstance(self._private_key_pem, str) else self._private_key_pem,
                password=None,
                backend=default_backend()
            )
        except Exception as e:
            logger.error("Failed to load private key", exc_info=True)
            try:
                get_logger().log_auth_event(
                    action="jwt_generate",
                    provider="coinbase_cdp",
                    success=False,
                    error_code="PRIVATE_KEY_LOAD_FAILED",
                    error="private_key_load_failed",
                )
            except Exception:
                pass
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
        try:
            get_logger().log_auth_event(
                action="jwt_generate",
                provider="coinbase_cdp",
                success=True,
                method=method.upper(),
                path=path,
                exp=claims.get("exp"),
            )
        except Exception:
            pass
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


def create_enhanced_cdp_auth(
    api_key_name: Optional[str] = None,
    private_key: Optional[str] = None
) -> EnhancedCDPAuth:
    """Factory function to create enhanced CDP auth instance.
    
    Supports multiple configuration methods:
    1. Explicit parameters
    2. Environment variables
    3. File paths
    
    Args:
        api_key_name: API key name (or from COINBASE_CDP_API_KEY env)
        private_key: Private key PEM or path (or from COINBASE_CDP_PRIVATE_KEY_PATH env)
        
    Returns:
        EnhancedCDPAuth instance
    """
    # Get API key name
    if not api_key_name:
        api_key_name = os.getenv("COINBASE_CDP_API_KEY")
        if not api_key_name:
            try:
                get_logger().log_auth_event(
                    action="config_missing",
                    provider="coinbase_cdp",
                    success=False,
                    error_code="MISSING_API_KEY",
                )
            except Exception:
                pass
            raise ValueError("CDP API key not provided or in environment")
    
    # Get private key source (prefer file path over direct key)
    if not private_key:
        # Check for file path first (more secure)
        private_key = os.getenv("COINBASE_CDP_PRIVATE_KEY_PATH")
        
        if not private_key:
            # Fall back to direct key (less secure)
            private_key = os.getenv("COINBASE_CDP_PRIVATE_KEY")
            
            if private_key:
                logger.warning("Using private key from COINBASE_CDP_PRIVATE_KEY environment variable")
                logger.warning("Consider using COINBASE_CDP_PRIVATE_KEY_PATH for better security")
        
        if not private_key:
            # Try common locations
            common_paths = [
                Path.home() / ".coinbase" / "keys" / "cdp_private_key.pem",
                Path("/secure/keys/coinbase_cdp_private_key.pem"),
                Path("./keys/cdp_private_key.pem"),
            ]
            
            for path in common_paths:
                if path.exists():
                    logger.info(f"Found private key at: {path}")
                    private_key = str(path)
                    break
            
            if not private_key:
                try:
                    get_logger().log_auth_event(
                        action="config_missing",
                        provider="coinbase_cdp",
                        success=False,
                        error_code="MISSING_PRIVATE_KEY",
                    )
                except Exception:
                    pass
                raise ValueError(
                    "CDP private key not provided. Set COINBASE_CDP_PRIVATE_KEY_PATH "
                    "or COINBASE_CDP_PRIVATE_KEY environment variable"
                )
    
    return EnhancedCDPAuth(
        api_key_name=api_key_name,
        private_key_source=private_key
    )


# Convenience function for backward compatibility
def get_cdp_auth() -> EnhancedCDPAuth:
    """Get CDP auth instance from environment configuration.
    
    Returns:
        Configured EnhancedCDPAuth instance
    """
    return create_enhanced_cdp_auth()
