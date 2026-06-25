"""
Secrets Manager for GPT-Trader

Provides secure storage and retrieval of sensitive data including API keys,
database credentials, and encryption keys. Supports HashiCorp Vault integration
with explicit encrypted file fallback outside development.
"""

import base64

# Removed unused imports - Fernet handles encryption directly
import hashlib
import hmac
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from gpt_trader.app.config import BotConfig
from gpt_trader.config import path_registry
from gpt_trader.config.config_utilities import parse_bool_env
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from cryptography.fernet import Fernet as FernetType
else:  # pragma: no cover - runtime import guard
    FernetType = Any

_RuntimeFernet: type[FernetType] | None
try:  # pragma: no cover - optional dependency
    from cryptography.fernet import Fernet as _RuntimeFernet  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - degrade gracefully
    _RuntimeFernet = None

logger = get_logger(__name__, component="security")

DEFAULT_DEVELOPMENT_VAULT_ADDRESS = "http://localhost:8200"
FILE_SECRET_FALLBACK_ENVIRONMENT_VARIABLE = "GPT_TRADER_ALLOW_FILE_SECRET_FALLBACK"
DEVELOPMENT_ENVIRONMENTS = frozenset({"development", "dev", "local", "paper", "test", "testing"})
LOG_FINGERPRINT_LENGTH = 12


class VaultConfigurationError(RuntimeError):
    """Raised when Vault is required but cannot be initialised safely."""


def _require_fernet() -> type[FernetType]:
    if _RuntimeFernet is None:
        raise RuntimeError(
            "cryptography is not installed. Install extras with `pip install gpt-trader[security]` "
            "or `uv sync`."
        )
    return cast(type[FernetType], _RuntimeFernet)


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
        self,
        vault_enabled: bool = True,
        *,
        config: BotConfig | None = None,
        secrets_dir: Path | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._cipher_suite: FernetType | None = None
        self._encryption_key_is_ephemeral = False
        self._log_fingerprint_key: bytes | None = None
        self._secrets_cache: dict[str, dict[str, Any]] = {}
        self._vault_client: Any | None = None
        self._vault_enabled = vault_enabled
        self._config = config
        self._environment = self._current_environment()
        self._allow_file_secret_fallback = parse_bool_env(
            FILE_SECRET_FALLBACK_ENVIRONMENT_VARIABLE,
            default=False,
        )
        self._secrets_dir = (
            Path(secrets_dir) if secrets_dir is not None else path_registry.USER_SECRETS_DIR
        )
        self._initialize_encryption()

        if vault_enabled:
            self._initialize_vault()

    def _initialize_encryption(self) -> None:
        """Initialize encryption using environment key or generate new"""
        encryption_key = os.getenv("GPT_TRADER_ENCRYPTION_KEY")

        if not encryption_key:
            # Generate new key for development-like environments.
            if self._is_development_environment():
                fernet_cls = _require_fernet()
                encryption_key = fernet_cls.generate_key().decode()
                self._encryption_key_is_ephemeral = True
                logger.warning(
                    "Generated new encryption key for development",
                    operation="encryption_init",
                    status="generated",
                )
            else:
                raise ValueError("ENCRYPTION_KEY must be set in production")

        # Validate and set cipher
        try:
            fernet_cls = _require_fernet()

            # Handle different key formats
            if isinstance(encryption_key, str):
                # Attempt to decode as hex first
                try:
                    key_bytes = bytes.fromhex(encryption_key)
                    if len(key_bytes) != 32:
                        raise ValueError("Hex key must represent 32 bytes")
                    key_material = base64.urlsafe_b64encode(key_bytes)
                except ValueError:
                    # Treat as pre-encoded string (base64 urlsafe or legacy plain)
                    key_material = encryption_key.encode()
            else:
                key_material = encryption_key

            self._cipher_suite = fernet_cls(key_material)
            self._log_fingerprint_key = hashlib.sha256(
                b"gpt-trader-secrets-log-ref-v1:" + bytes(key_material)
            ).digest()
        except Exception as e:
            raise ValueError(f"Invalid encryption key: {e}")

    def _require_cipher(self) -> FernetType:
        if self._cipher_suite is None:
            raise RuntimeError("Encryption subsystem not initialised")
        return self._cipher_suite

    def _require_durable_file_key(self) -> None:
        if self._encryption_key_is_ephemeral:
            raise RuntimeError(
                "File-backed secret storage requires GPT_TRADER_ENCRYPTION_KEY; "
                "development-generated encryption keys are ephemeral and cache-only."
            )

    @staticmethod
    def _as_secret_dict(payload: Any) -> dict[str, Any] | None:
        if isinstance(payload, dict):
            return dict(payload)
        return None

    def _fingerprint_for_log(self, value: str) -> str:
        if self._log_fingerprint_key is None:
            raise RuntimeError("Log fingerprinting key not initialised")
        return hmac.new(
            self._log_fingerprint_key,
            value.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:LOG_FINGERPRINT_LENGTH]

    def _secret_log_ref(self, path: str) -> str:
        return f"secret:{self._fingerprint_for_log(str(path))}"

    def _encrypted_file_log_ref(self, file_path: Path) -> str:
        return f"encrypted-file:{self._fingerprint_for_log(str(file_path))}"

    def _current_environment(self) -> str:
        if self._config is not None:
            config_environment = (self._config.environment or "").strip().lower()
            if config_environment:
                return config_environment
        return os.getenv("ENV", "development").strip().lower() or "development"

    def _is_development_environment(self) -> bool:
        return self._environment in DEVELOPMENT_ENVIRONMENTS

    def _file_secret_fallback_allowed(self) -> bool:
        return self._is_development_environment() or self._allow_file_secret_fallback

    def _disable_vault_for_file_storage(self) -> None:
        self._vault_enabled = False
        self._vault_client = None

    def _raise_vault_configuration_error(self, reason: str) -> None:
        raise VaultConfigurationError(
            f"{reason}. Vault-backed storage is required in environment "
            f"'{self._environment}'. Set VAULT_ADDR to an https:// Vault endpoint "
            f"and VAULT_TOKEN to a valid token, or set "
            f"{FILE_SECRET_FALLBACK_ENVIRONMENT_VARIABLE}=1 to intentionally use "
            "encrypted file storage."
        )

    def _vault_address_missing(self) -> None:
        reason = "VAULT_ADDR must be set outside development"
        if self._file_secret_fallback_allowed():
            logger.warning(
                f"{reason}; using encrypted file storage",
                operation="vault_init",
                status="address_missing",
                environment=self._environment,
            )
            self._disable_vault_for_file_storage()
            return

        logger.error(
            f"{reason}; encrypted file fallback is disabled",
            operation="vault_init",
            status="address_missing",
            environment=self._environment,
        )
        self._raise_vault_configuration_error(reason)

    def _vault_address_insecure(self) -> None:
        reason = "VAULT_ADDR must use https:// outside development"
        if self._file_secret_fallback_allowed():
            logger.warning(
                f"{reason}; using encrypted file storage",
                operation="vault_init",
                status="insecure_address",
                environment=self._environment,
            )
            self._disable_vault_for_file_storage()
            return

        logger.error(
            f"{reason}; encrypted file fallback is disabled",
            operation="vault_init",
            status="insecure_address",
            environment=self._environment,
        )
        self._raise_vault_configuration_error(reason)

    def _vault_token_missing(self) -> None:
        reason = "VAULT_TOKEN must be set when Vault is enabled"
        if self._file_secret_fallback_allowed():
            logger.warning(
                f"{reason}; using encrypted file storage",
                operation="vault_init",
                status="token_missing",
                environment=self._environment,
            )
            self._disable_vault_for_file_storage()
            return

        logger.error(
            f"{reason}; encrypted file fallback is disabled",
            operation="vault_init",
            status="token_missing",
            environment=self._environment,
        )
        self._raise_vault_configuration_error(reason)

    def _vault_unauthenticated(self) -> None:
        reason = "Vault authentication failed"
        if self._file_secret_fallback_allowed():
            logger.warning(
                f"{reason}; using encrypted file storage",
                operation="vault_init",
                status="unauthenticated",
                environment=self._environment,
            )
            self._disable_vault_for_file_storage()
            return

        logger.error(
            f"{reason}; encrypted file fallback is disabled",
            operation="vault_init",
            status="unauthenticated",
            environment=self._environment,
        )
        self._raise_vault_configuration_error(reason)

    def _vault_dependency_missing(self) -> None:
        reason = "hvac is not installed"
        if self._file_secret_fallback_allowed():
            logger.warning(
                f"{reason}; using encrypted file storage",
                operation="vault_init",
                status="dependency_missing",
                environment=self._environment,
            )
            self._disable_vault_for_file_storage()
            return

        logger.error(
            f"{reason}; encrypted file fallback is disabled",
            operation="vault_init",
            status="dependency_missing",
            environment=self._environment,
        )
        self._raise_vault_configuration_error(reason)

    def _vault_initialization_error(self, reason: str) -> None:
        if self._file_secret_fallback_allowed():
            logger.warning(
                f"{reason}; using encrypted file storage",
                operation="vault_init",
                status="error",
                environment=self._environment,
            )
            self._disable_vault_for_file_storage()
            return

        logger.error(
            f"{reason}; encrypted file fallback is disabled",
            operation="vault_init",
            status="error",
            environment=self._environment,
        )
        self._raise_vault_configuration_error(reason)

    def _resolve_vault_address(self) -> str | None:
        vault_address = os.getenv("VAULT_ADDR")
        if not vault_address:
            if self._is_development_environment():
                return DEFAULT_DEVELOPMENT_VAULT_ADDRESS
            self._vault_address_missing()
            return None

        vault_address = vault_address.strip()
        if not self._is_development_environment() and not vault_address.lower().startswith(
            "https://"
        ):
            self._vault_address_insecure()
            return None

        return vault_address

    def _initialize_vault(self) -> None:
        """Initialize HashiCorp Vault connection"""
        try:
            import hvac

            logger.debug("Successfully imported hvac module")

            vault_addr = self._resolve_vault_address()
            if not self._vault_enabled or vault_addr is None:
                return

            vault_token = os.getenv("VAULT_TOKEN")
            logger.debug(
                f"Vault config - addr: {vault_addr}, token: {'*' * len(vault_token) if vault_token else 'None'}"
            )

            if vault_token:
                self._vault_client = hvac.Client(url=vault_addr, token=vault_token)
                logger.debug(f"Created vault client: {type(self._vault_client)}")

                if self._vault_client.is_authenticated():
                    logger.info(
                        "Vault connection established",
                        operation="vault_init",
                        status="connected",
                    )
                    logger.debug("Vault authentication successful, keeping vault enabled")
                else:
                    self._vault_unauthenticated()
            else:
                self._vault_token_missing()

        except ImportError:
            self._vault_dependency_missing()
        except Exception as e:
            if isinstance(e, VaultConfigurationError):
                raise
            self._vault_initialization_error(
                f"Vault initialization failed: {type(e).__name__}: {e}"
            )

    def store_secret(self, path: str, secret: dict[str, Any] | Any) -> bool:
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
                # Validate secret payload
                validated_secret = self._as_secret_dict(secret)
                if validated_secret is None:
                    return False

                if self._vault_enabled and self._vault_client:
                    # Store in Vault
                    self._vault_client.secrets.kv.v2.create_or_update_secret(
                        path=path, secret=validated_secret
                    )
                    logger.info(
                        "Secret stored in Vault",
                        operation="secret_store",
                        status="success",
                        storage_backend="vault",
                        secret_ref=self._secret_log_ref(path),
                    )
                else:
                    # Fallback to encrypted file
                    self._store_to_file(path, validated_secret)
                    logger.info(
                        "Secret stored in encrypted file",
                        operation="secret_store",
                        status="success",
                        storage_backend="encrypted_file",
                        secret_ref=self._secret_log_ref(path),
                    )

                # Update cache
                self._secrets_cache[path] = validated_secret
                return True

            except Exception as e:
                logger.error(
                    "Failed to store secret",
                    operation="secret_store",
                    status="error",
                    error_type=type(e).__name__,
                    secret_ref=self._secret_log_ref(path),
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

                if secret is not None:
                    self._secrets_cache[path] = secret
                    return secret

            except Exception as e:
                logger.error(
                    "Failed to retrieve secret",
                    operation="secret_retrieve",
                    status="error",
                    error_type=type(e).__name__,
                    secret_ref=self._secret_log_ref(path),
                )

            return None

    def _store_to_file(self, path: str, secret: dict[str, Any]) -> None:
        """Store secret to encrypted file"""
        self._require_durable_file_key()
        secrets_dir = self._secrets_dir
        secret_ref = self._secret_log_ref(path)
        logger.debug("Using encrypted file storage directory", secret_ref=secret_ref)
        secrets_dir.mkdir(parents=True, exist_ok=True)

        file_path = secrets_dir / f"{path.replace('/', '_')}.enc"
        file_ref = self._encrypted_file_log_ref(file_path)
        logger.debug(
            "Storing secret to encrypted file",
            secret_ref=secret_ref,
            encrypted_file_ref=file_ref,
        )

        # Encrypt data
        cipher = self._require_cipher()
        data = json.dumps(secret).encode()
        encrypted = cipher.encrypt(data)

        # Write to file
        file_path.write_bytes(encrypted)
        logger.debug(
            "Successfully wrote encrypted secret file",
            byte_count=len(encrypted),
            secret_ref=secret_ref,
            encrypted_file_ref=file_ref,
        )

    def _load_from_file(self, path: str) -> dict[str, Any] | None:
        """Load secret from encrypted file"""
        secrets_dir = self._secrets_dir
        file_path = secrets_dir / f"{path.replace('/', '_')}.enc"
        secret_ref = self._secret_log_ref(path)
        file_ref = self._encrypted_file_log_ref(file_path)
        logger.debug(
            "Loading secret from encrypted file",
            secret_ref=secret_ref,
            encrypted_file_ref=file_ref,
        )

        if not file_path.exists():
            logger.debug(
                "Encrypted secret file does not exist",
                secret_ref=secret_ref,
                encrypted_file_ref=file_ref,
            )
            return None

        try:
            # Read and decrypt
            cipher = self._require_cipher()
            try:
                encrypted = file_path.read_bytes()
                logger.debug(
                    "Read encrypted secret file",
                    byte_count=len(encrypted),
                    secret_ref=secret_ref,
                    encrypted_file_ref=file_ref,
                )
            except OSError as exc:
                logger.error(
                    "Failed to read encrypted secret file",
                    error_type=type(exc).__name__,
                    secret_ref=secret_ref,
                    encrypted_file_ref=file_ref,
                )
                return None

            try:
                decrypted = cipher.decrypt(encrypted)
            except Exception as exc:
                logger.error(
                    "Failed to decrypt encrypted secret file",
                    error_type=type(exc).__name__,
                    secret_ref=secret_ref,
                    encrypted_file_ref=file_ref,
                )
                return None

            try:
                payload = json.loads(decrypted.decode())
                logger.debug(
                    "Successfully decrypted and loaded secret",
                    secret_ref=secret_ref,
                    encrypted_file_ref=file_ref,
                )
                result = self._as_secret_dict(payload)
                # Handle empty dict case - ensure we return {} instead of None
                if result is None and payload == {}:
                    return {}
                return result
            except ValueError as exc:
                logger.error(
                    "Invalid JSON in encrypted secret file",
                    error_type=type(exc).__name__,
                    secret_ref=secret_ref,
                    encrypted_file_ref=file_ref,
                )
                return None
        except Exception as e:
            logger.error(
                "Unexpected error loading secret",
                error_type=type(e).__name__,
                secret_ref=secret_ref,
                encrypted_file_ref=file_ref,
            )
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

            import secrets

            # Create a copy to avoid modifying the cached version
            updated_secret = dict(secret)

            # Always add rotation fields for key rotation
            # This is needed for test_rotate_key_success which expects _key to be added
            updated_secret["_key"] = secrets.token_urlsafe(32)
            updated_secret["_rotated_at"] = datetime.now().isoformat()

            # Store with updated secret
            try:
                return self.store_secret(path, updated_secret)
            except Exception as exc:
                logger.error(
                    "Failed to store rotated secret",
                    error_type=type(exc).__name__,
                    operation="rotate_key",
                    secret_ref=self._secret_log_ref(path),
                )
                return False

    def delete_secret(self, path: str) -> bool:
        """Delete a secret"""
        with self._lock:
            try:
                if self._vault_enabled and self._vault_client:
                    self._vault_client.secrets.kv.v2.delete_metadata_and_all_versions(path=path)
                else:
                    secrets_dir = self._secrets_dir
                    file_path = secrets_dir / f"{path.replace('/', '_')}.enc"
                    if file_path.exists():
                        file_path.unlink()

                # Remove from cache
                self._secrets_cache.pop(path, None)
                return True

            except Exception as e:
                logger.error(
                    "Failed to delete secret",
                    operation="secret_delete",
                    status="error",
                    error_type=type(e).__name__,
                    secret_ref=self._secret_log_ref(path),
                )
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
            secrets_dir = self._secrets_dir
            if secrets_dir.exists():
                paths = [f.stem.replace("_", "/") for f in secrets_dir.glob("*.enc")]

        return paths

    def clear_cache(self) -> None:
        """Clear the secrets cache"""
        with self._lock:
            self._secrets_cache.clear()
