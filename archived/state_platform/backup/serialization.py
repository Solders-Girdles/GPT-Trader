"""Backup serialization, compression, and encryption"""

import logging
from pathlib import Path
from typing import Any

from bot_v2.state.backup.models import BackupConfig
from bot_v2.state.utils import compress_data, decompress_data, serialize_to_json

# Optional encryption support
try:
    from cryptography.fernet import Fernet

    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False
    logging.warning("Cryptography library not available, encryption disabled")

logger = logging.getLogger(__name__)


class BackupSerializer:
    """Handles backup serialization, compression, and encryption"""

    def __init__(self, config: BackupConfig) -> None:
        self.config = config
        self._encryption_key: bytes | None = None

        # Initialize encryption if enabled
        if self.config.enable_encryption and ENCRYPTION_AVAILABLE:
            self._init_encryption()

    def _init_encryption(self) -> None:
        """Initialize encryption key"""
        try:
            # Try to load existing key
            key_file = Path(self.config.backup_dir) / ".encryption_key"

            if key_file.exists():
                with open(key_file, "rb") as f:
                    self._encryption_key = f.read()
                logger.info("Loaded existing encryption key")
            else:
                # Generate new key
                self._encryption_key = Fernet.generate_key()
                with open(key_file, "wb") as f:
                    f.write(self._encryption_key)
                logger.info("Generated new encryption key")

        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            self._encryption_key = None

    def serialize_backup_data(self, backup_data: dict[str, Any]) -> bytes:
        """Serialize backup data to bytes"""
        # Use shared serialization utility
        return serialize_to_json(backup_data)

    def prepare_compressed_payload(self, serialized: bytes) -> tuple[bytes, int, int]:
        """Compress serialized data if enabled"""
        original_size = len(serialized)
        if self.config.enable_compression:
            # Use shared compression utility
            compressed = compress_data(serialized, compression_level=6)
            return compressed, original_size, len(compressed)
        return serialized, original_size, original_size

    def encrypt_payload(self, payload: bytes) -> tuple[bytes, str | None]:
        """Encrypt payload if encryption is enabled"""
        if self.config.enable_encryption and self._encryption_key and ENCRYPTION_AVAILABLE:
            fernet = Fernet(self._encryption_key)
            encrypted = fernet.encrypt(payload)
            encryption_algorithm = "Fernet"
            return encrypted, encryption_algorithm
        return payload, None

    def decrypt_payload(self, encrypted_data: bytes) -> bytes:
        """Decrypt backup data"""
        if not self._encryption_key or not ENCRYPTION_AVAILABLE:
            return encrypted_data

        try:
            fernet = Fernet(self._encryption_key)
            return fernet.decrypt(encrypted_data)
        except Exception as e:
            logger.error(f"Failed to decrypt backup: {e}")
            raise

    def decompress_payload(self, data: bytes) -> bytes:
        """Decompress backup data if needed"""
        try:
            # Use shared decompression utility
            return decompress_data(data)
        except Exception:
            # Data not compressed
            return data

    def calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum for data"""
        import hashlib

        return hashlib.sha256(data).hexdigest()

    def verify_checksum(self, data: bytes, expected_checksum: str) -> bool:
        """Verify data integrity using checksum"""
        actual_checksum = self.calculate_checksum(data)
        return actual_checksum == expected_checksum
