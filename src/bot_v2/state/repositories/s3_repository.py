"""
S3 State Repository - COLD tier storage implementation

Provides long-term archival storage with lower access times.
"""

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from bot_v2.monitoring.metrics_collector import MetricsCollector

from bot_v2.state.utils.adapters import S3Adapter

logger = logging.getLogger(__name__)

__all__ = ["S3StateRepository"]


class S3StateRepository:
    """
    S3 repository for COLD tier state storage.

    Provides long-term archival storage with lower access times.
    """

    def __init__(
        self,
        adapter: S3Adapter,
        bucket: str,
        prefix: str = "cold/",
        metrics_collector: "MetricsCollector | None" = None,
    ) -> None:
        """
        Initialize S3 repository.

        Args:
            adapter: S3 adapter instance
            bucket: S3 bucket name
            prefix: Key prefix for cold storage objects
            metrics_collector: Optional metrics collector for telemetry
        """
        self.adapter = adapter
        self.bucket = bucket
        self.prefix = prefix
        self.metrics_collector = metrics_collector

    def _build_key(self, key: str) -> str:
        """Build full S3 key with prefix."""
        return f"{self.prefix}{key}"

    def _strip_prefix(self, key: str) -> str:
        """Strip prefix from S3 key."""
        return key.replace(self.prefix, "")

    async def fetch(self, key: str) -> Any | None:
        """
        Fetch state from S3.

        Args:
            key: State key to fetch

        Returns:
            Deserialized state value or None if not found
        """
        if self.metrics_collector:
            self.metrics_collector.record_counter("state.repository.s3.operations.fetch_total")

        try:
            response = self.adapter.get_object(bucket=self.bucket, key=self._build_key(key))
            data = response["Body"].read().decode("utf-8")
            return json.loads(data)
        except Exception as e:
            logger.debug(f"S3 fetch failed for {key}: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_counter("state.repository.s3.operations.errors_total")

        return None

    async def store(self, key: str, value: str, metadata: dict[str, Any]) -> bool:
        """
        Store state in S3.

        Args:
            key: State key
            value: Serialized state value
            metadata: Metadata dict containing 'checksum'

        Returns:
            True if successful, False otherwise
        """
        if self.metrics_collector:
            self.metrics_collector.record_counter("state.repository.s3.operations.store_total")

        try:
            checksum = metadata.get("checksum", "")
            self.adapter.put_object(
                bucket=self.bucket,
                key=self._build_key(key),
                body=value.encode(),
                storage_class="STANDARD_IA",
                metadata={"checksum": checksum},
            )
            return True
        except Exception as e:
            logger.error(f"S3 store failed for {key}: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_counter("state.repository.s3.operations.errors_total")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete state from S3.

        Args:
            key: State key to delete

        Returns:
            True if successful, False otherwise
        """
        if self.metrics_collector:
            self.metrics_collector.record_counter("state.repository.s3.operations.delete_total")

        try:
            self.adapter.delete_object(bucket=self.bucket, key=self._build_key(key))
            return True
        except Exception as e:
            logger.warning(f"S3 delete failed for {key}: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_counter("state.repository.s3.operations.errors_total")
            return False

    async def keys(self, pattern: str) -> list[str]:
        """
        Get S3 keys matching pattern.

        Args:
            pattern: Key pattern (limited support, uses prefix)

        Returns:
            List of matching keys
        """
        try:
            prefix = pattern.split("*")[0] if "*" in pattern else pattern
            response = self.adapter.list_objects_v2(
                bucket=self.bucket, prefix=self._build_key(prefix)
            )
            if "Contents" in response:
                return [self._strip_prefix(obj["Key"]) for obj in response["Contents"]]
        except Exception as e:
            logger.debug(f"S3 key lookup failed for {pattern}: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_counter("state.repository.s3.operations.errors_total")

        return []

    async def stats(self) -> dict[str, Any]:
        """
        Get S3 storage statistics.

        Returns:
            Dict containing 'key_count' and other stats
        """
        try:
            response = self.adapter.list_objects_v2(bucket=self.bucket, prefix=self.prefix)
            return {"key_count": response.get("KeyCount", 0)}
        except Exception as e:
            logger.debug(f"S3 stats collection failed: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_counter("state.repository.s3.operations.errors_total")

        return {"key_count": 0}

    # Batch operations
    async def store_many(self, items: dict[str, tuple[str, dict[str, Any]]]) -> set[str]:
        """
        Store multiple items in S3.

        Note: S3 doesn't have batch put, so this iterates sequentially.
        For true parallelism, consider using concurrent uploads at higher level.

        Args:
            items: Dict mapping keys to (value, metadata) tuples

        Returns:
            Set of keys that were successfully stored
        """
        if not items:
            return set()

        if self.metrics_collector:
            self.metrics_collector.record_counter("state.repository.s3.operations.store_many_total")
            self.metrics_collector.record_histogram(
                "state.repository.s3.operations.batch_size", float(len(items))
            )

        successful_keys = set()
        for key, (value, metadata) in items.items():
            try:
                checksum = metadata.get("checksum", "")
                self.adapter.put_object(
                    bucket=self.bucket,
                    key=self._build_key(key),
                    body=value.encode(),
                    storage_class="STANDARD_IA",
                    metadata={"checksum": checksum},
                )
                successful_keys.add(key)
            except Exception as e:
                logger.error(f"S3 store failed for {key}: {e}")
                if self.metrics_collector:
                    self.metrics_collector.record_counter(
                        "state.repository.s3.operations.errors_total"
                    )

        return successful_keys

    async def delete_many(self, keys: list[str]) -> int:
        """
        Delete multiple keys from S3 using batch delete.

        Args:
            keys: List of keys to delete

        Returns:
            Number of keys successfully deleted
        """
        if not keys:
            return 0

        if self.metrics_collector:
            self.metrics_collector.record_counter(
                "state.repository.s3.operations.delete_many_total"
            )
            self.metrics_collector.record_histogram(
                "state.repository.s3.operations.batch_size", float(len(keys))
            )

        try:
            # Build full S3 keys with prefix
            full_keys = [self._build_key(key) for key in keys]

            # S3 batch delete handles up to 1000 keys
            response = self.adapter.delete_objects(bucket=self.bucket, keys=full_keys)

            # Count successful deletions
            deleted = len(response.get("Deleted", []))
            errors = response.get("Errors", [])

            if errors:
                logger.warning(f"S3 delete_many had {len(errors)} errors")

            return deleted
        except Exception as e:
            logger.error(f"S3 delete_many failed: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_counter("state.repository.s3.operations.errors_total")
            return 0
