"""Redis failure recovery strategy."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from bot_v2.state.recovery.base import RecoveryStrategy

if TYPE_CHECKING:
    from bot_v2.state.recovery_handler import RecoveryOperation

logger = logging.getLogger(__name__)


class RedisRecoveryStrategy(RecoveryStrategy):
    """Recovers from Redis failure by restoring from PostgreSQL warm storage."""

    @property
    def failure_type_name(self) -> str:
        return "Redis Down"

    async def recover(self, operation: RecoveryOperation) -> bool:
        """Recover from Redis failure."""
        try:
            logger.info("Recovering Redis from warm storage")
            operation.actions_taken.append("Starting Redis recovery from PostgreSQL")

            # Get critical hot state from PostgreSQL backup
            if self.state_manager.pg_conn:
                with self.state_manager.pg_conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT key, data FROM state_warm
                        WHERE last_accessed > %s
                        ORDER BY last_accessed DESC
                        LIMIT 1000
                    """,
                        (datetime.utcnow() - timedelta(minutes=5),),
                    )

                    recovered_count = 0
                    for row in cursor.fetchall():
                        key, data = row["key"], row["data"]
                        try:
                            # Attempt to restore to Redis
                            if self.state_manager.redis_client:
                                self.state_manager.redis_client.set(key, json.dumps(data), ex=3600)
                                recovered_count += 1
                        except Exception as exc:
                            logger.debug(
                                "Failed to restore key %s to Redis: %s",
                                key,
                                exc,
                                exc_info=True,
                            )

                    operation.actions_taken.append(f"Recovered {recovered_count} keys to Redis")
                    logger.info(f"Redis recovery completed with {recovered_count} keys")
                    return recovered_count > 0

            return False

        except Exception as e:
            logger.error(f"Redis recovery failed: {e}")
            operation.actions_taken.append(f"Redis recovery error: {str(e)}")
            return False
