"""Compression service for backup data.

Handles compression and decompression of backup payloads.
"""

import gzip
import logging

logger = logging.getLogger(__name__)


class CompressionService:
    """Service for compressing and decompressing backup data."""

    def __init__(self, enabled: bool = True, compression_level: int = 6) -> None:
        """
        Initialize compression service.

        Args:
            enabled: Whether compression is enabled
            compression_level: Compression level (1-9, higher = better compression)
        """
        self.enabled = enabled
        self.compression_level = max(1, min(9, compression_level))

    def compress(self, data: bytes) -> tuple[bytes, int, int]:
        """
        Compress data.

        Args:
            data: Data to compress

        Returns:
            Tuple of (compressed_data, original_size, compressed_size)
        """
        original_size = len(data)

        if not self.enabled:
            return data, original_size, 0

        try:
            compressed = gzip.compress(data, compresslevel=self.compression_level)
            compressed_size = len(compressed)

            logger.debug(
                f"Compressed {original_size} -> {compressed_size} bytes "
                f"({100 * (1 - compressed_size / original_size):.1f}% reduction)"
            )

            return compressed, original_size, compressed_size

        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return data, original_size, 0

    def decompress(self, data: bytes) -> bytes:
        """
        Decompress data.

        Args:
            data: Compressed data

        Returns:
            Decompressed data

        Raises:
            ValueError: If decompression fails
        """
        if not self.enabled:
            return data

        try:
            return gzip.decompress(data)

        except (OSError, gzip.BadGzipFile) as e:
            logger.warning(f"Expected compressed data but found plain data or corruption: {e}")
            # Return original data - might be uncompressed
            return data

        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise ValueError(f"Decompression failed: {e}") from e

    @property
    def is_enabled(self) -> bool:
        """Check if compression is enabled."""
        return self.enabled
