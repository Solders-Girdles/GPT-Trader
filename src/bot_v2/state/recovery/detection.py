"""
Failure Detection and Health Checks

Provides system-wide health monitoring and failure type detection
for the recovery system.
"""

import logging

from bot_v2.state.recovery.models import FailureType

logger = logging.getLogger(__name__)


class FailureDetector:
    """Detects system failures through health checks"""

    def __init__(self, state_manager, checkpoint_handler) -> None:
        self.state_manager = state_manager
        self.checkpoint_handler = checkpoint_handler

    async def detect_failures(self) -> list[FailureType]:
        """
        Detect system failures through health checks.

        Returns:
            List of detected failure types
        """
        failures = []

        # Test Redis connectivity
        if not await self.test_redis_health():
            failures.append(FailureType.REDIS_DOWN)

        # Test PostgreSQL connectivity
        if not await self.test_postgres_health():
            failures.append(FailureType.POSTGRES_DOWN)

        # Test S3 availability
        if not await self.test_s3_health():
            failures.append(FailureType.S3_UNAVAILABLE)

        # Check for data corruption
        if await self.detect_data_corruption():
            failures.append(FailureType.DATA_CORRUPTION)

        # Check system resources
        if await self.check_memory_usage() > 90:
            failures.append(FailureType.MEMORY_OVERFLOW)

        if await self.check_disk_usage() > 95:
            failures.append(FailureType.DISK_FULL)

        # Check trading engine
        if not await self.test_trading_engine_health():
            failures.append(FailureType.TRADING_ENGINE_CRASH)

        return failures

    async def test_redis_health(self) -> bool:
        """Test Redis connectivity"""
        try:
            if self.state_manager.redis_client:
                self.state_manager.redis_client.ping()
                return True
        except Exception as exc:
            logger.debug("Redis health check failed: %s", exc, exc_info=True)
        return False

    async def test_postgres_health(self) -> bool:
        """Test PostgreSQL connectivity"""
        try:
            if self.state_manager.pg_conn:
                with self.state_manager.pg_conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                return True
        except Exception as exc:
            logger.debug("PostgreSQL health check failed: %s", exc, exc_info=True)
        return False

    async def test_s3_health(self) -> bool:
        """Test S3 availability"""
        try:
            if self.state_manager.s3_client:
                self.state_manager.s3_client.head_bucket(Bucket=self.state_manager.config.s3_bucket)
                return True
        except Exception as exc:
            logger.debug("S3 health check failed: %s", exc, exc_info=True)
        return False

    async def test_trading_engine_health(self) -> bool:
        """Test trading engine health"""
        try:
            status = await self.state_manager.get_state("system:trading_engine_status")
            return status not in [None, "crashed", "error"]
        except Exception as exc:
            logger.debug("Trading engine health check failed: %s", exc, exc_info=True)
            return False

    async def detect_data_corruption(self) -> bool:
        """Detect data corruption through checksums"""
        try:
            import hashlib
            import json

            # Sample critical data and verify checksums
            critical_keys = ["portfolio_current", "performance_metrics"]

            for key in critical_keys:
                data = await self.state_manager.get_state(key)
                if data and isinstance(data, dict):
                    stored_checksum = data.get("_checksum")
                    if stored_checksum:
                        # Recalculate checksum
                        data_copy = data.copy()
                        del data_copy["_checksum"]
                        calculated = hashlib.sha256(
                            json.dumps(data_copy, sort_keys=True).encode()
                        ).hexdigest()

                        if calculated != stored_checksum:
                            logger.warning(f"Checksum mismatch for {key}")
                            return True

            return False

        except Exception as e:
            logger.error(f"Corruption detection error: {e}")
            return False

    async def check_memory_usage(self) -> float:
        """Check memory usage percentage"""
        try:
            import psutil

            return psutil.virtual_memory().percent
        except ImportError:
            logger.debug("psutil not available; cannot check memory usage")
        except Exception as exc:
            logger.debug("Memory usage check failed: %s", exc, exc_info=True)
        return 0

    async def check_disk_usage(self) -> float:
        """Check disk usage percentage"""
        try:
            import psutil

            return psutil.disk_usage("/").percent
        except ImportError:
            logger.debug("psutil not available; cannot check disk usage")
        except Exception as exc:
            logger.debug("Disk usage check failed: %s", exc, exc_info=True)
        return 0
