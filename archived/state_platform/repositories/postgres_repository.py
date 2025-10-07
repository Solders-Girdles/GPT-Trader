"""
PostgreSQL State Repository - WARM tier storage implementation

Provides recent data access with ~5s latency and transaction support.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from bot_v2.monitoring.metrics_collector import MetricsCollector

from bot_v2.state.utils.adapters import PostgresAdapter

logger = logging.getLogger(__name__)

__all__ = ["PostgresStateRepository"]


class PostgresStateRepository:
    """
    PostgreSQL repository for WARM tier state storage.

    Provides recent data access with ~5s latency.
    """

    def __init__(
        self, adapter: PostgresAdapter, metrics_collector: "MetricsCollector | None" = None
    ) -> None:
        """
        Initialize PostgreSQL repository.

        Args:
            adapter: PostgreSQL adapter instance
            metrics_collector: Optional metrics collector for telemetry
        """
        self.adapter = adapter
        self.metrics_collector = metrics_collector

    async def fetch(self, key: str) -> Any | None:
        """
        Fetch state from PostgreSQL.

        Args:
            key: State key to fetch

        Returns:
            Deserialized state value or None if not found
        """
        if self.metrics_collector:
            self.metrics_collector.record_counter(
                "state.repository.postgres.operations.fetch_total"
            )

        try:
            results = self.adapter.execute("SELECT data FROM state_warm WHERE key = %s", (key,))
            if results:
                result = results[0]
                # Update last accessed time
                self.adapter.execute(
                    "UPDATE state_warm SET last_accessed = %s WHERE key = %s",
                    (datetime.utcnow(), key),
                )
                self.adapter.commit()
                return result["data"]
        except Exception as e:
            logger.debug(f"PostgreSQL fetch failed for {key}: {e}")
            self.adapter.rollback()
            if self.metrics_collector:
                self.metrics_collector.record_counter(
                    "state.repository.postgres.operations.errors_total"
                )

        return None

    async def store(self, key: str, value: str, metadata: dict[str, Any]) -> bool:
        """
        Store state in PostgreSQL.

        Args:
            key: State key
            value: Serialized state value
            metadata: Metadata dict containing 'checksum' and 'size_bytes'

        Returns:
            True if successful, False otherwise
        """
        if self.metrics_collector:
            self.metrics_collector.record_counter(
                "state.repository.postgres.operations.store_total"
            )

        try:
            checksum = metadata.get("checksum", "")
            size_bytes = metadata.get("size_bytes", len(value.encode()))

            self.adapter.execute(
                """
                INSERT INTO state_warm (key, data, checksum, size_bytes)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (key) DO UPDATE SET
                    data = EXCLUDED.data,
                    checksum = EXCLUDED.checksum,
                    size_bytes = EXCLUDED.size_bytes,
                    last_accessed = CURRENT_TIMESTAMP,
                    version = state_warm.version + 1
            """,
                (key, value, checksum, size_bytes),
            )
            self.adapter.commit()
            return True
        except Exception as e:
            logger.error(f"PostgreSQL store failed for {key}: {e}")
            self.adapter.rollback()
            if self.metrics_collector:
                self.metrics_collector.record_counter(
                    "state.repository.postgres.operations.errors_total"
                )
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete state from PostgreSQL.

        Args:
            key: State key to delete

        Returns:
            True if successful, False otherwise
        """
        if self.metrics_collector:
            self.metrics_collector.record_counter(
                "state.repository.postgres.operations.delete_total"
            )

        try:
            self.adapter.execute("DELETE FROM state_warm WHERE key = %s", (key,))
            self.adapter.commit()
            return True
        except Exception as e:
            logger.warning(f"PostgreSQL delete failed for {key}: {e}")
            try:
                self.adapter.rollback()
            except Exception:
                logger.debug("PostgreSQL rollback failed after delete error", exc_info=True)
            if self.metrics_collector:
                self.metrics_collector.record_counter(
                    "state.repository.postgres.operations.errors_total"
                )
            return False

    async def keys(self, pattern: str) -> list[str]:
        """
        Get PostgreSQL keys matching pattern.

        Args:
            pattern: Key pattern (converts * to SQL %)

        Returns:
            List of matching keys
        """
        try:
            sql_pattern = pattern.replace("*", "%")
            results = self.adapter.execute(
                "SELECT key FROM state_warm WHERE key LIKE %s", (sql_pattern,)
            )
            return [row["key"] for row in results]
        except Exception as e:
            logger.debug(f"PostgreSQL key lookup failed for {pattern}: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_counter(
                    "state.repository.postgres.operations.errors_total"
                )
            return []

    async def stats(self) -> dict[str, Any]:
        """
        Get PostgreSQL storage statistics.

        Returns:
            Dict containing 'key_count' and other stats
        """
        try:
            results = self.adapter.execute("SELECT COUNT(*) as count FROM state_warm")
            if results:
                result = results[0]
                return {"key_count": result["count"]}
        except Exception as e:
            logger.debug(f"PostgreSQL stats collection failed: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_counter(
                    "state.repository.postgres.operations.errors_total"
                )

        return {"key_count": 0}

    # Batch operations
    async def store_many(self, items: dict[str, tuple[str, dict[str, Any]]]) -> set[str]:
        """
        Store multiple items in PostgreSQL using batch upsert.

        Args:
            items: Dict mapping keys to (value, metadata) tuples

        Returns:
            Set of keys that were successfully stored
        """
        if not items:
            return set()

        if self.metrics_collector:
            self.metrics_collector.record_counter(
                "state.repository.postgres.operations.store_many_total"
            )
            self.metrics_collector.record_histogram(
                "state.repository.postgres.operations.batch_size", float(len(items))
            )

        try:
            records = []
            for key, (value, metadata) in items.items():
                checksum = metadata.get("checksum", "")
                size_bytes = metadata.get("size_bytes", len(value.encode()))
                records.append(
                    {
                        "key": key,
                        "data": value,
                        "checksum": checksum,
                        "size_bytes": size_bytes,
                    }
                )

            count = self.adapter.batch_upsert("state_warm", "key", records)
            self.adapter.commit()
            # All items succeeded (PostgreSQL batch_upsert is transactional)
            return set(items.keys()) if count > 0 else set()
        except Exception as e:
            logger.error(f"PostgreSQL store_many failed: {e}")
            self.adapter.rollback()
            if self.metrics_collector:
                self.metrics_collector.record_counter(
                    "state.repository.postgres.operations.errors_total"
                )
            return set()

    async def delete_many(self, keys: list[str]) -> int:
        """
        Delete multiple keys from PostgreSQL.

        Args:
            keys: List of keys to delete

        Returns:
            Number of keys successfully deleted
        """
        if not keys:
            return 0

        if self.metrics_collector:
            self.metrics_collector.record_counter(
                "state.repository.postgres.operations.delete_many_total"
            )
            self.metrics_collector.record_histogram(
                "state.repository.postgres.operations.batch_size", float(len(keys))
            )

        try:
            count = self.adapter.batch_delete("state_warm", "key", keys)
            self.adapter.commit()
            return count
        except Exception as e:
            logger.error(f"PostgreSQL delete_many failed: {e}")
            try:
                self.adapter.rollback()
            except Exception:
                logger.debug("PostgreSQL rollback failed after delete_many error", exc_info=True)
            if self.metrics_collector:
                self.metrics_collector.record_counter(
                    "state.repository.postgres.operations.errors_total"
                )
            return 0
