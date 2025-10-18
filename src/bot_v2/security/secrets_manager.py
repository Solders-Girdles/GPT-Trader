"""
Secrets Manager for Bot V2 Trading System

Provides secure storage and retrieval of sensitive data including API keys,
database credentials, and encryption keys. Supports HashiCorp Vault integration
with encrypted file fallback.
"""

# Removed unused imports - Fernet handles encryption directly
import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cryptography.fernet import Fernet

from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="security")

if TYPE_CHECKING:  # pragma: no cover
    from hvac import Client as HvacClient
else:
    HvacClient = Any  # type: ignore[misc]


@dataclass
class EncryptedData:
    """Container for encrypted data with metadata"""

    ciphertext: bytes
    timestamp: datetime
    version: int = 1


class SecretsManager:
    """
    Hierarchical secrets manager with encryption and vault support.
    Implements secure storage patterns for production trading systems.
    """

    def __init__(
        self, vault_enabled: bool = True, *, settings: RuntimeSettings | None = None
    ) -> None:
        self._lock = threading.RLock()
        self._cipher_suite: Fernet | None = None
        self._secrets_cache: dict[str, dict[str, Any]] = {}
        self._vault_client: HvacClient | None = None
        self._vault_enabled = vault_enabled
        self._static_settings = settings is not None
        self._settings = settings or load_runtime_settings()
        self._initialize_encryption()

        if vault_enabled:
            self._initialize_vault()

    def _initialize_encryption(self) -> None:
        """Initialize encryption using environment key or generate new"""
        env_map = (
            self._settings.raw_env if self._static_settings else load_runtime_settings().raw_env
        )
        encryption_key = env_map.get("BOT_V2_ENCRYPTION_KEY")

        if not encryption_key:
            # Generate new key for development
            environment = (env_map.get("ENV") or "development").lower()
            if environment == "development":
                encryption_key = Fernet.generate_key().decode()
                logger.warning(
                    "Generated new encryption key for development",
                    operation="encryption_init",
                    status="generated",
                )
            else:
                raise ValueError("ENCRYPTION_KEY must be set in production")

        # Validate and set cipher
        try:
            self._cipher_suite = Fernet(
                encryption_key.encode() if isinstance(encryption_key, str) else encryption_key
            )
        except Exception as e:
            raise ValueError(f"Invalid encryption key: {e}")

    def _require_cipher(self) -> Fernet:
        if self._cipher_suite is None:
            raise RuntimeError("Encryption subsystem not initialised")
        return self._cipher_suite

    @staticmethod
    def _as_secret_dict(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return dict(payload)
        raise ValueError("Secret payload must be a mapping")

    def _initialize_vault(self) -> None:
        """Initialize HashiCorp Vault connection"""
        try:
            import hvac

            env_map = (
                self._settings.raw_env if self._static_settings else load_runtime_settings().raw_env
            )
            vault_addr = env_map.get("VAULT_ADDR", "http://localhost:8200")
            vault_token = env_map.get("VAULT_TOKEN")

            if vault_token:
                self._vault_client = hvac.Client(url=vault_addr, token=vault_token)

                if self._vault_client.is_authenticated():
                    logger.info(
                        "Vault connection established",
                        operation="vault_init",
                        status="connected",
                    )
                else:
                    logger.warning(
                        "Vault authentication failed, using file fallback",
                        operation="vault_init",
                        status="unauthenticated",
                    )
                    self._vault_enabled = False
            else:
                logger.info(
                    "Vault token not found, using file storage",
                    operation="vault_init",
                    status="token_missing",
                )
                self._vault_enabled = False

        except ImportError:
            logger.warning(
                "hvac not installed, using file storage",
                operation="vault_init",
                status="dependency_missing",
            )
            self._vault_enabled = False
        except Exception as e:
            logger.error(
                f"Vault initialization failed: {e}",
                operation="vault_init",
                status="error",
            )
            self._vault_enabled = False

    def store_secret(self, path: str, secret: dict[str, Any]) -> bool:
        """
        Store a secret at the specified path.

        Args:
            path: Hierarchical path (e.g., 'brokers/coinbase' or 'brokers/{name}')
            secret: Secret data to store

        Returns:
            Success status
        """
        with self._lock:
            try:
                if self._vault_enabled and self._vault_client:
                    # Store in Vault
                    self._vault_client.secrets.kv.v2.create_or_update_secret(
                        path=path, secret=secret
                    )
                    logger.info(
                        f"Secret stored in Vault: {path}",
                        operation="secret_store",
                        status="success",
                    )
                else:
                    # Fallback to encrypted file
                    self._store_to_file(path, secret)
                    logger.info(
                        f"Secret stored in encrypted file: {path}",
                        operation="secret_store",
                        status="success",
                    )

                # Update cache
                self._secrets_cache[path] = secret
                return True

            except Exception as e:
                logger.error(
                    f"Failed to store secret: {e}",
                    operation="secret_store",
                    status="error",
                )
                return False

    def get_secret(self, path: str) -> dict[str, Any] | None:
        """
        Retrieve a secret from the specified path.

        Args:
            path: Hierarchical path to secret

        Returns:
            Secret data or None if not found
        """
        with self._lock:
            # Check cache first
            if path in self._secrets_cache:
                return self._secrets_cache[path]

            try:
                if self._vault_enabled and self._vault_client is not None:
                    # Retrieve from Vault
                    response = self._vault_client.secrets.kv.v2.read_secret_version(path=path)
                    secret_payload = response.get("data", {}).get("data")
                    secret = (
                        self._as_secret_dict(secret_payload) if secret_payload is not None else None
                    )
                else:
                    # Retrieve from encrypted file
                    secret = self._load_from_file(path)

                if secret:
                    self._secrets_cache[path] = secret
                    return secret

            except Exception as e:
                logger.error(
                    f"Failed to retrieve secret: {e}",
                    operation="secret_retrieve",
                    status="error",
                )

            return None

    def _store_to_file(self, path: str, secret: dict[str, Any]) -> None:
        """Store secret to encrypted file"""
        secrets_dir = Path.home() / ".bot_v2" / "secrets"
        secrets_dir.mkdir(parents=True, exist_ok=True)

        file_path = secrets_dir / f"{path.replace('/', '_')}.enc"

        # Encrypt data
        cipher = self._require_cipher()
        data = json.dumps(secret).encode()
        encrypted = cipher.encrypt(data)

        # Write to file
        file_path.write_bytes(encrypted)

    def _load_from_file(self, path: str) -> dict[str, Any] | None:
        """Load secret from encrypted file"""
        secrets_dir = Path.home() / ".bot_v2" / "secrets"
        file_path = secrets_dir / f"{path.replace('/', '_')}.enc"

        if not file_path.exists():
            return None

        try:
            # Read and decrypt
            cipher = self._require_cipher()
            encrypted = file_path.read_bytes()
            decrypted = cipher.decrypt(encrypted)
            payload = json.loads(decrypted.decode())
            return self._as_secret_dict(payload)
        except ValueError as exc:
            logger.error("Invalid secret payload for %s: %s", path, exc)
            return None
        except Exception as e:
            logger.error(f"Failed to decrypt file: {e}")
            return None

    def rotate_key(self, path: str) -> bool:
        """
        Rotate the encryption key for a secret.

        Args:
            path: Path to secret

        Returns:
            Success status
        """
        with self._lock:
            # Get current secret
            secret = self.get_secret(path)
            if not secret:
                return False

            # Generate new key if needed
            if "_key" in secret:
                import secrets

                secret["_key"] = secrets.token_urlsafe(32)
                secret["_rotated_at"] = datetime.now().isoformat()

                # Store with new key
                return self.store_secret(path, secret)

            return True

    def delete_secret(self, path: str) -> bool:
        """Delete a secret"""
        with self._lock:
            try:
                if self._vault_enabled and self._vault_client:
                    self._vault_client.secrets.kv.v2.delete_metadata_and_all_versions(path=path)
                else:
                    secrets_dir = Path.home() / ".bot_v2" / "secrets"
                    file_path = secrets_dir / f"{path.replace('/', '_')}.enc"
                    if file_path.exists():
                        file_path.unlink()

                # Remove from cache
                self._secrets_cache.pop(path, None)
                return True

            except Exception as e:
                logger.error(f"Failed to delete secret: {e}")
                return False

    def list_secrets(self) -> list[str]:
        """List all available secret paths"""
        paths = []

        if self._vault_enabled and self._vault_client:
            try:
                response = self._vault_client.secrets.kv.v2.list_secrets(path="")
                paths = response["data"]["keys"]
            except Exception as exc:
                logger.warning(
                    "Unable to list secrets from vault: %s",
                    exc,
                    exc_info=True,
                )
        else:
            secrets_dir = Path.home() / ".bot_v2" / "secrets"
            if secrets_dir.exists():
                paths = [f.stem.replace("_", "/") for f in secrets_dir.glob("*.enc")]

        return paths

    def clear_cache(self) -> None:
        """Clear the secrets cache"""
        with self._lock:
            self._secrets_cache.clear()


# Global instance
_secrets_manager: SecretsManager | None = None


def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance"""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


# Convenience functions
def store_secret(path: str, secret: dict[str, Any]) -> bool:
    """Store a secret"""
    return get_secrets_manager().store_secret(path, secret)


def get_secret(path: str) -> dict[str, Any] | None:
    """Get a secret"""
    return get_secrets_manager().get_secret(path)


def get_api_key(service: str) -> str | None:
    """Get API key for a specific service"""
    secret = get_secret(f"api_keys/{service}")
    return secret.get("key") if secret else None
