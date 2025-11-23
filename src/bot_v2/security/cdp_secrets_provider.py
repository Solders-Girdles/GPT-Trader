"""
CDP Secrets Provider for Bot V2 Trading System

Provides secure storage and automatic rotation of Coinbase Developer Platform (CDP)
API keys with short-lived JWT generation. Integrates with SecretsManager for
1Password/HashiCorp Vault backend support.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from bot_v2.features.brokerages.coinbase.auth import CDPJWTAuth, create_cdp_jwt_auth
from bot_v2.security.secrets_manager import SecretsManager, get_secrets_manager
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="cdp_secrets")


@dataclass
class CDPCredentials:
    """CDP API credentials with metadata"""

    api_key_name: str
    private_key_pem: str
    created_at: datetime
    last_rotated_at: datetime | None = None
    rotation_policy_days: int = 90
    allowed_ips: list[str] | None = None  # IP allowlist for this key


@dataclass
class CDPJWTToken:
    """Short-lived CDP JWT token with expiration tracking"""

    token: str
    created_at: datetime
    expires_at: datetime
    method: str
    path: str

    @property
    def is_expired(self) -> bool:
        """Check if token is expired"""
        return datetime.now(UTC) >= self.expires_at

    @property
    def seconds_until_expiry(self) -> float:
        """Get seconds until token expires"""
        delta = self.expires_at - datetime.now(UTC)
        return max(0.0, delta.total_seconds())


class CDPSecretsProvider:
    """
    Manages CDP API credentials with automatic rotation and short-lived JWT generation.

    Features:
    - Secure storage via SecretsManager (1Password/HashiCorp Vault)
    - Short-lived JWT generation (120 seconds per Coinbase specs)
    - IP allowlisting enforcement
    - Automatic key rotation policies
    - Thread-safe token caching
    """

    def __init__(
        self,
        secrets_manager: SecretsManager | None = None,
        default_rotation_days: int = 90,
    ) -> None:
        self._secrets_manager = secrets_manager or get_secrets_manager()
        self._default_rotation_days = default_rotation_days
        self._lock = threading.RLock()
        self._credentials_cache: dict[str, CDPCredentials] = {}
        self._auth_cache: dict[str, CDPJWTAuth] = {}

    def store_cdp_credentials(
        self,
        service_name: str,
        api_key_name: str,
        private_key_pem: str,
        *,
        rotation_policy_days: int | None = None,
        allowed_ips: list[str] | None = None,
    ) -> bool:
        """
        Store CDP credentials securely.

        Args:
            service_name: Service identifier (e.g., 'coinbase_production', 'coinbase_intx')
            api_key_name: CDP API key name
            private_key_pem: EC private key in PEM format
            rotation_policy_days: Days before credential rotation required
            allowed_ips: IP addresses allowed to use this key

        Returns:
            Success status
        """
        with self._lock:
            now = datetime.now(UTC)
            credentials = CDPCredentials(
                api_key_name=api_key_name,
                private_key_pem=private_key_pem,
                created_at=now,
                last_rotated_at=None,
                rotation_policy_days=rotation_policy_days or self._default_rotation_days,
                allowed_ips=allowed_ips,
            )

            # Store in vault
            secret_data = {
                "api_key_name": credentials.api_key_name,
                "private_key_pem": credentials.private_key_pem,
                "created_at": credentials.created_at.isoformat(),
                "last_rotated_at": (
                    credentials.last_rotated_at.isoformat() if credentials.last_rotated_at else None
                ),
                "rotation_policy_days": credentials.rotation_policy_days,
                "allowed_ips": credentials.allowed_ips or [],
            }

            path = f"cdp_credentials/{service_name}"
            success = self._secrets_manager.store_secret(path, secret_data)

            if success:
                self._credentials_cache[service_name] = credentials
                logger.info(
                    f"Stored CDP credentials for {service_name}",
                    operation="store_credentials",
                    service=service_name,
                    status="success",
                )
            else:
                logger.error(
                    f"Failed to store CDP credentials for {service_name}",
                    operation="store_credentials",
                    service=service_name,
                    status="failed",
                )

            return success

    def get_cdp_credentials(self, service_name: str) -> CDPCredentials | None:
        """
        Retrieve CDP credentials.

        Args:
            service_name: Service identifier

        Returns:
            CDPCredentials if found, None otherwise
        """
        with self._lock:
            # Check cache first
            if service_name in self._credentials_cache:
                return self._credentials_cache[service_name]

            # Retrieve from vault
            path = f"cdp_credentials/{service_name}"
            secret_data = self._secrets_manager.get_secret(path)

            if not secret_data:
                logger.warning(
                    f"CDP credentials not found for {service_name}",
                    operation="get_credentials",
                    service=service_name,
                    status="not_found",
                )
                return None

            # Reconstruct credentials
            credentials = CDPCredentials(
                api_key_name=secret_data["api_key_name"],
                private_key_pem=secret_data["private_key_pem"],
                created_at=datetime.fromisoformat(secret_data["created_at"]),
                last_rotated_at=(
                    datetime.fromisoformat(secret_data["last_rotated_at"])
                    if secret_data.get("last_rotated_at")
                    else None
                ),
                rotation_policy_days=secret_data.get(
                    "rotation_policy_days", self._default_rotation_days
                ),
                allowed_ips=secret_data.get("allowed_ips"),
            )

            self._credentials_cache[service_name] = credentials
            return credentials

    def generate_short_lived_jwt(
        self,
        service_name: str,
        method: str,
        path: str,
        *,
        client_ip: str | None = None,
        base_url: str | None = None,
    ) -> CDPJWTToken | None:
        """
        Generate a short-lived JWT token for CDP API request.

        Per Coinbase specs:
        - Tokens expire after 120 seconds (2 minutes)
        - Each request should have a unique JWT
        - Uses ES256 algorithm (ECDSA with SHA-256)

        Args:
            service_name: Service identifier
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path
            client_ip: Client IP address for allowlist validation
            base_url: Base URL for the API

        Returns:
            CDPJWTToken if successful, None otherwise
        """
        with self._lock:
            credentials = self.get_cdp_credentials(service_name)
            if not credentials:
                return None

            # Validate IP allowlist
            if credentials.allowed_ips and client_ip:
                if not self._validate_ip_allowlist(client_ip, credentials.allowed_ips):
                    logger.warning(
                        f"IP {client_ip} not in allowlist for {service_name}",
                        operation="generate_jwt",
                        service=service_name,
                        client_ip=client_ip,
                        status="ip_rejected",
                    )
                    return None

            # Check if credentials need rotation
            if self._needs_rotation(credentials):
                logger.warning(
                    f"CDP credentials for {service_name} need rotation",
                    operation="generate_jwt",
                    service=service_name,
                    status="rotation_needed",
                )

            # Get or create auth instance
            auth = self._get_auth_instance(service_name, credentials, base_url)

            # Generate JWT
            try:
                token_str = auth.generate_jwt(method, path)
                now = datetime.now(UTC)
                expires_at = now + timedelta(seconds=120)  # CDP tokens expire in 2 minutes

                token = CDPJWTToken(
                    token=token_str,
                    created_at=now,
                    expires_at=expires_at,
                    method=method,
                    path=path,
                )

                logger.debug(
                    f"Generated short-lived JWT for {service_name}",
                    operation="generate_jwt",
                    service=service_name,
                    method=method,
                    path=path,
                    expires_in_seconds=120,
                )

                return token

            except Exception as e:
                logger.error(
                    f"Failed to generate JWT for {service_name}: {e}",
                    operation="generate_jwt",
                    service=service_name,
                    status="error",
                    exc_info=True,
                )
                return None

    def _get_auth_instance(
        self,
        service_name: str,
        credentials: CDPCredentials,
        base_url: str | None,
    ) -> CDPJWTAuth:
        """Get or create cached auth instance"""
        cache_key = f"{service_name}:{base_url or 'default'}"

        if cache_key not in self._auth_cache:
            self._auth_cache[cache_key] = create_cdp_jwt_auth(
                api_key_name=credentials.api_key_name,
                private_key_pem=credentials.private_key_pem,
                base_url=base_url,
            )

        return self._auth_cache[cache_key]

    def _validate_ip_allowlist(self, client_ip: str, allowed_ips: list[str]) -> bool:
        """
        Validate client IP against allowlist.

        Supports:
        - Exact IP matching (e.g., "192.168.1.1")
        - CIDR notation (e.g., "192.168.1.0/24")
        """
        import ipaddress

        try:
            client_addr = ipaddress.ip_address(client_ip)

            for allowed in allowed_ips:
                # Try exact match first
                if allowed == client_ip:
                    return True

                # Try CIDR notation
                try:
                    network = ipaddress.ip_network(allowed, strict=False)
                    if client_addr in network:
                        return True
                except ValueError:
                    # Not a valid CIDR, skip
                    continue

            return False

        except ValueError:
            logger.warning(
                f"Invalid IP address format: {client_ip}",
                operation="validate_ip",
                client_ip=client_ip,
            )
            return False

    def _needs_rotation(self, credentials: CDPCredentials) -> bool:
        """Check if credentials need rotation based on policy"""
        if credentials.rotation_policy_days <= 0:
            return False

        last_rotation = credentials.last_rotated_at or credentials.created_at
        age_days = (datetime.now(UTC) - last_rotation).days

        return age_days >= credentials.rotation_policy_days

    def rotate_credentials(
        self,
        service_name: str,
        new_api_key_name: str,
        new_private_key_pem: str,
    ) -> bool:
        """
        Rotate CDP credentials.

        Args:
            service_name: Service identifier
            new_api_key_name: New CDP API key name
            new_private_key_pem: New EC private key in PEM format

        Returns:
            Success status
        """
        with self._lock:
            old_credentials = self.get_cdp_credentials(service_name)
            if not old_credentials:
                logger.error(
                    f"Cannot rotate - credentials not found for {service_name}",
                    operation="rotate_credentials",
                    service=service_name,
                    status="not_found",
                )
                return False

            # Store new credentials with rotation timestamp
            now = datetime.now(UTC)
            new_credentials = CDPCredentials(
                api_key_name=new_api_key_name,
                private_key_pem=new_private_key_pem,
                created_at=old_credentials.created_at,  # Keep original creation time
                last_rotated_at=now,
                rotation_policy_days=old_credentials.rotation_policy_days,
                allowed_ips=old_credentials.allowed_ips,
            )

            secret_data = {
                "api_key_name": new_credentials.api_key_name,
                "private_key_pem": new_credentials.private_key_pem,
                "created_at": new_credentials.created_at.isoformat(),
                "last_rotated_at": new_credentials.last_rotated_at.isoformat(),
                "rotation_policy_days": new_credentials.rotation_policy_days,
                "allowed_ips": new_credentials.allowed_ips or [],
            }

            path = f"cdp_credentials/{service_name}"
            success = self._secrets_manager.store_secret(path, secret_data)

            if success:
                self._credentials_cache[service_name] = new_credentials
                # Clear auth cache to force new auth instance
                self._auth_cache = {
                    k: v
                    for k, v in self._auth_cache.items()
                    if not k.startswith(f"{service_name}:")
                }

                logger.info(
                    f"Rotated CDP credentials for {service_name}",
                    operation="rotate_credentials",
                    service=service_name,
                    status="success",
                )
            else:
                logger.error(
                    f"Failed to rotate CDP credentials for {service_name}",
                    operation="rotate_credentials",
                    service=service_name,
                    status="failed",
                )

            return success

    def update_ip_allowlist(self, service_name: str, allowed_ips: list[str]) -> bool:
        """
        Update IP allowlist for CDP credentials.

        Args:
            service_name: Service identifier
            allowed_ips: New list of allowed IP addresses/CIDR blocks

        Returns:
            Success status
        """
        with self._lock:
            credentials = self.get_cdp_credentials(service_name)
            if not credentials:
                return False

            credentials.allowed_ips = allowed_ips

            secret_data = {
                "api_key_name": credentials.api_key_name,
                "private_key_pem": credentials.private_key_pem,
                "created_at": credentials.created_at.isoformat(),
                "last_rotated_at": (
                    credentials.last_rotated_at.isoformat() if credentials.last_rotated_at else None
                ),
                "rotation_policy_days": credentials.rotation_policy_days,
                "allowed_ips": credentials.allowed_ips,
            }

            path = f"cdp_credentials/{service_name}"
            success = self._secrets_manager.store_secret(path, secret_data)

            if success:
                logger.info(
                    f"Updated IP allowlist for {service_name}",
                    operation="update_ip_allowlist",
                    service=service_name,
                    allowed_ips_count=len(allowed_ips),
                    status="success",
                )

            return success

    def list_credentials(self) -> list[str]:
        """List all stored CDP credential service names"""
        all_secrets = self._secrets_manager.list_secrets()
        return [
            s.replace("cdp_credentials/", "")
            for s in all_secrets
            if s.startswith("cdp_credentials/")
        ]


# Global instance
_cdp_secrets_provider: CDPSecretsProvider | None = None


def get_cdp_secrets_provider() -> CDPSecretsProvider:
    """Get the global CDP secrets provider instance"""
    global _cdp_secrets_provider
    if _cdp_secrets_provider is None:
        _cdp_secrets_provider = CDPSecretsProvider()
    return _cdp_secrets_provider


# Convenience functions
def store_cdp_credentials(
    service_name: str,
    api_key_name: str,
    private_key_pem: str,
    *,
    rotation_policy_days: int | None = None,
    allowed_ips: list[str] | None = None,
) -> bool:
    """Store CDP credentials"""
    return get_cdp_secrets_provider().store_cdp_credentials(
        service_name,
        api_key_name,
        private_key_pem,
        rotation_policy_days=rotation_policy_days,
        allowed_ips=allowed_ips,
    )


def generate_cdp_jwt(
    service_name: str,
    method: str,
    path: str,
    *,
    client_ip: str | None = None,
    base_url: str | None = None,
) -> CDPJWTToken | None:
    """Generate short-lived CDP JWT token"""
    return get_cdp_secrets_provider().generate_short_lived_jwt(
        service_name, method, path, client_ip=client_ip, base_url=base_url
    )
