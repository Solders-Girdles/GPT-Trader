"""Encryption service for backup data.

Handles encryption key management and encrypt/decrypt operations.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional encryption support
try:
    from cryptography.fernet import Fernet

    ENCRYPTION_AVAILABLE = True
except ImportError:
    Fernet = None  # type: ignore[assignment]
    ENCRYPTION_AVAILABLE = False
    logger.warning("Cryptography library not available, encryption disabled")


class EncryptionService:
    """Service for encrypting and decrypting backup data."""

    def __init__(self, key_path: Path, enabled: bool = True) -> None:
        """
        Initialize encryption service.

        Args:
            key_path: Path to encryption key file
            enabled: Whether encryption is enabled
        """
        self.key_path = key_path
        self.enabled = enabled and ENCRYPTION_AVAILABLE
        self._cipher: Fernet | None = None  # type: ignore[assignment]
        self._encryption_key: bytes | None = None

        if self.enabled:
            self._init_encryption()

    def _init_encryption(self) -> None:
        """Initialize encryption key and cipher."""
        if not ENCRYPTION_AVAILABLE or Fernet is None:
            self.enabled = False
            self._cipher = None
            return

        try:
            if self.key_path.exists():
                # Load existing key
                with open(self.key_path, "rb") as f:
                    self._encryption_key = f.read()
            else:
                # Generate new key
                generated_key = Fernet.generate_key()
                if not isinstance(generated_key, (bytes, bytearray)):
                    generated_key = str(generated_key).encode()
                self._encryption_key = generated_key

                # Save key (in production, use proper key management)
                self.key_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.key_path, "wb") as f:
                    f.write(self._encryption_key)

                # Secure the key file
                os.chmod(self.key_path, 0o600)

            self._cipher = Fernet(self._encryption_key)
            logger.info("Encryption initialized")

        except Exception as e:
            logger.error(f"Encryption initialization failed: {e}")
            self.enabled = False
            self._cipher = None

    def encrypt(self, payload: bytes) -> tuple[bytes, str | None]:
        """
        Encrypt payload data.

        Args:
            payload: Data to encrypt

        Returns:
            Tuple of (encrypted_data, key_id)
        """
        if not self.enabled or not self._cipher:
            return payload, None

        try:
            encrypted = self._cipher.encrypt(payload)
            if isinstance(encrypted, (bytes, bytearray)):
                return bytes(encrypted), "primary"

            logger.warning("Cipher returned non-bytes payload; using original data")
            return payload, "primary"

        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return payload, None

    def decrypt(self, payload: bytes) -> bytes:
        """
        Decrypt payload data.

        Args:
            payload: Encrypted data

        Returns:
            Decrypted data

        Raises:
            ValueError: If decryption fails
        """
        if not self.enabled or not self._cipher:
            return payload

        try:
            decrypted = self._cipher.decrypt(payload)
            return decrypted.encode() if isinstance(decrypted, str) else decrypted

        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError(f"Decryption failed: {e}") from e

    @property
    def is_available(self) -> bool:
        """Check if encryption is available."""
        return ENCRYPTION_AVAILABLE

    @property
    def is_enabled(self) -> bool:
        """Check if encryption is enabled."""
        return self.enabled
