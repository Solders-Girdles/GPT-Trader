"""
Recovery Handler for Bot V2 Trading System

Orchestrates failure recovery with RTO <5 minutes and RPO <1 minute.
Provides automatic failure detection, recovery orchestration, and validation.
"""

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from bot_v2.state.recovery import (
    APIGatewayRecoveryStrategy,
    CorruptionRecoveryStrategy,
    DiskRecoveryStrategy,
    MemoryRecoveryStrategy,
    MLModelsRecoveryStrategy,
    NetworkRecoveryStrategy,
    PostgresRecoveryStrategy,
    RecoveryStrategy,
    RedisRecoveryStrategy,
    S3RecoveryStrategy,
    TradingEngineRecoveryStrategy,
)

logger = logging.getLogger(__name__)


class RecoveryMode(Enum):
    """Recovery operation modes"""

    AUTOMATIC = "automatic"
    MANUAL = "manual"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"


class FailureType(Enum):
    """Types of system failures"""

    REDIS_DOWN = "redis_down"
    POSTGRES_DOWN = "postgres_down"
    S3_UNAVAILABLE = "s3_unavailable"
    NETWORK_PARTITION = "network_partition"
    DATA_CORRUPTION = "data_corruption"
    TRADING_ENGINE_CRASH = "trading_engine_crash"
    ML_MODEL_FAILURE = "ml_model_failure"
    API_GATEWAY_DOWN = "api_gateway_down"
    MEMORY_OVERFLOW = "memory_overflow"
    DISK_FULL = "disk_full"


class RecoveryStatus(Enum):
    """Recovery operation status"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class FailureEvent:
    """Failure event information"""

    failure_type: FailureType
    timestamp: datetime
    severity: str  # critical, high, medium, low
    affected_components: list[str]
    error_message: str
    stack_trace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryOperation:
    """Recovery operation details"""

    operation_id: str
    failure_event: FailureEvent
    recovery_mode: RecoveryMode
    status: RecoveryStatus
    started_at: datetime
    completed_at: datetime | None = None
    recovery_time_seconds: float | None = None
    data_loss_estimate: str | None = None
    actions_taken: list[str] = field(default_factory=list)
    validation_results: dict[str, bool] = field(default_factory=dict)


@dataclass
class RecoveryConfig:
    """Configuration for recovery operations"""

    rto_minutes: int = 5  # Recovery Time Objective
    rpo_minutes: int = 1  # Recovery Point Objective
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 30
    automatic_recovery_enabled: bool = True
    failure_detection_interval_seconds: int = 10
    validation_timeout_seconds: int = 60
    escalation_threshold_minutes: int = 3
    alert_channels: list[str] = field(default_factory=lambda: ["log", "email"])


class RecoveryHandler:
    """
    Orchestrates failure recovery with automatic detection and resolution.
    Ensures RTO <5 minutes and RPO <1 minute for production systems.
    """

    def __init__(
        self,
        state_manager: Any,
        checkpoint_handler: Any,
        backup_manager: Any = None,
        config: RecoveryConfig | None = None,
    ) -> None:
        self.state_manager = state_manager
        self.checkpoint_handler = checkpoint_handler
        self.backup_manager = backup_manager
        self.config = config or RecoveryConfig()

        self._recovery_lock = threading.Lock()
        self._recovery_in_progress = False
        self._current_operation: RecoveryOperation | None = None
        self._recovery_history: list[RecoveryOperation] = []
        self._strategies: dict[FailureType, RecoveryStrategy] = {}
        self._monitoring_task: asyncio.Task | None = None

        # Register recovery strategies
        self._register_strategies()

        logger.info(
            f"RecoveryHandler initialized with RTO={self.config.rto_minutes}min, "
            f"RPO={self.config.rpo_minutes}min"
        )

    def _register_strategies(self) -> None:
        """Register recovery strategies for each failure type."""
        self._strategies = {
            FailureType.REDIS_DOWN: RedisRecoveryStrategy(
                self.state_manager, self.checkpoint_handler, self.backup_manager
            ),
            FailureType.POSTGRES_DOWN: PostgresRecoveryStrategy(
                self.state_manager, self.checkpoint_handler, self.backup_manager
            ),
            FailureType.S3_UNAVAILABLE: S3RecoveryStrategy(
                self.state_manager, self.checkpoint_handler, self.backup_manager
            ),
            FailureType.DATA_CORRUPTION: CorruptionRecoveryStrategy(
                self.state_manager, self.checkpoint_handler, self.backup_manager
            ),
            FailureType.TRADING_ENGINE_CRASH: TradingEngineRecoveryStrategy(
                self.state_manager, self.checkpoint_handler, self.backup_manager
            ),
            FailureType.ML_MODEL_FAILURE: MLModelsRecoveryStrategy(
                self.state_manager, self.checkpoint_handler, self.backup_manager
            ),
            FailureType.MEMORY_OVERFLOW: MemoryRecoveryStrategy(
                self.state_manager, self.checkpoint_handler, self.backup_manager
            ),
            FailureType.DISK_FULL: DiskRecoveryStrategy(
                self.state_manager, self.checkpoint_handler, self.backup_manager
            ),
            FailureType.NETWORK_PARTITION: NetworkRecoveryStrategy(
                self.state_manager, self.checkpoint_handler, self.backup_manager
            ),
            FailureType.API_GATEWAY_DOWN: APIGatewayRecoveryStrategy(
                self.state_manager, self.checkpoint_handler, self.backup_manager
            ),
        }

    async def start_monitoring(self) -> None:
        """Start continuous failure detection monitoring"""
        if self._monitoring_task:
            logger.warning("Monitoring already started")
            return

        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Recovery monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop failure detection monitoring"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            logger.info("Recovery monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Continuous monitoring loop for failure detection"""
        while True:
            try:
                # Detect failures
                failures = await self.detect_failures()

                if failures and self.config.automatic_recovery_enabled:
                    # Prioritize by severity
                    critical_failures = [f for f in failures if self._is_critical(f)]

                    for failure in critical_failures:
                        if not self._recovery_in_progress:
                            # Create failure event
                            event = FailureEvent(
                                failure_type=failure,
                                timestamp=datetime.utcnow(),
                                severity="critical",
                                affected_components=self._get_affected_components(failure),
                                error_message=f"Automatic detection: {failure.value}",
                            )

                            # Initiate recovery
                            await self.initiate_recovery(event, RecoveryMode.AUTOMATIC)

                await asyncio.sleep(self.config.failure_detection_interval_seconds)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.config.failure_detection_interval_seconds * 2)

    async def detect_failures(self) -> list[FailureType]:
        """
        Detect system failures through health checks.

        Returns:
            List of detected failure types
        """
        failures = []

        # Test Redis connectivity
        if not await self._test_redis_health():
            failures.append(FailureType.REDIS_DOWN)

        # Test PostgreSQL connectivity
        if not await self._test_postgres_health():
            failures.append(FailureType.POSTGRES_DOWN)

        # Test S3 availability
        if not await self._test_s3_health():
            failures.append(FailureType.S3_UNAVAILABLE)

        # Check for data corruption
        if await self._detect_data_corruption():
            failures.append(FailureType.DATA_CORRUPTION)

        # Check system resources
        if await self._check_memory_usage() > 90:
            failures.append(FailureType.MEMORY_OVERFLOW)

        if await self._check_disk_usage() > 95:
            failures.append(FailureType.DISK_FULL)

        # Check trading engine
        if not await self._test_trading_engine_health():
            failures.append(FailureType.TRADING_ENGINE_CRASH)

        return failures

    async def initiate_recovery(
        self, failure_event: FailureEvent, mode: RecoveryMode = RecoveryMode.AUTOMATIC
    ) -> RecoveryOperation:
        """
        Initiate recovery operation for detected failure.

        Args:
            failure_event: Failure event details
            mode: Recovery mode

        Returns:
            Recovery operation details
        """
        if self._recovery_in_progress:
            logger.warning("Recovery already in progress")
            return self._current_operation

        with self._recovery_lock:
            self._recovery_in_progress = True
            operation = self._create_recovery_operation(failure_event, mode)
            self._current_operation = operation

            try:
                await self._run_recovery(operation, mode)
            except Exception as exc:
                logger.error(f"Recovery operation failed: {exc}")
                operation.status = RecoveryStatus.FAILED
                operation.actions_taken.append(f"Error: {str(exc)}")
            finally:
                self._finalize_recovery_operation(operation)

        return operation

    def _create_recovery_operation(
        self, failure_event: FailureEvent, mode: RecoveryMode
    ) -> RecoveryOperation:
        return RecoveryOperation(
            operation_id=self._generate_operation_id(),
            failure_event=failure_event,
            recovery_mode=mode,
            status=RecoveryStatus.IN_PROGRESS,
            started_at=datetime.utcnow(),
        )

    async def _run_recovery(self, operation: RecoveryOperation, mode: RecoveryMode) -> None:
        failure_type = operation.failure_event.failure_type.value
        logger.info(
            "Starting recovery operation %s for %s in %s mode",
            operation.operation_id,
            failure_type,
            mode.value,
        )

        await self._send_alert(
            f"Recovery started for {failure_type}",
            operation,
        )

        success = await self._execute_recovery(operation)

        if success:
            await self._complete_recovery(operation)
        else:
            await self._handle_recovery_failure(operation, mode)

        await self._send_completion_alert(operation)

    async def _complete_recovery(self, operation: RecoveryOperation) -> None:
        operation.status = RecoveryStatus.VALIDATING
        if await self._validate_recovery(operation):
            operation.status = RecoveryStatus.COMPLETED
            operation.completed_at = datetime.utcnow()
            operation.recovery_time_seconds = (
                operation.completed_at - operation.started_at
            ).total_seconds()
            logger.info(
                "Recovery %s completed successfully in %.2f seconds",
                operation.operation_id,
                operation.recovery_time_seconds,
            )
            if operation.recovery_time_seconds > self.config.rto_minutes * 60:
                logger.warning(
                    "Recovery exceeded RTO: %.2fs > %.2fs",
                    operation.recovery_time_seconds,
                    self.config.rto_minutes * 60,
                )
        else:
            operation.status = RecoveryStatus.PARTIAL
            logger.warning(f"Recovery {operation.operation_id} validation failed")

    async def _handle_recovery_failure(
        self, operation: RecoveryOperation, mode: RecoveryMode
    ) -> None:
        operation.status = RecoveryStatus.FAILED
        if mode == RecoveryMode.AUTOMATIC:
            logger.info("Automatic recovery failed, escalating to manual mode")
            await self._escalate_recovery(operation)

    async def _send_completion_alert(self, operation: RecoveryOperation) -> None:
        await self._send_alert(
            f"Recovery {operation.status.value} for {operation.failure_event.failure_type.value}",
            operation,
        )

    def _finalize_recovery_operation(self, operation: RecoveryOperation) -> None:
        self._recovery_in_progress = False
        self._current_operation = None
        self._recovery_history.append(operation)
        self._cleanup_recovery_history()

    async def _execute_recovery(self, operation: RecoveryOperation) -> bool:
        """
        Execute recovery based on failure type.

        Args:
            operation: Recovery operation

        Returns:
            Success status
        """
        failure_type = operation.failure_event.failure_type

        # Get appropriate recovery strategy
        strategy = self._strategies.get(failure_type)

        if not strategy:
            logger.error(f"No recovery strategy registered for {failure_type.value}")
            return False

        # Execute recovery with retries
        for attempt in range(self.config.max_retry_attempts):
            try:
                logger.info(f"Recovery attempt {attempt + 1}/{self.config.max_retry_attempts}")

                success = await strategy.recover(operation)

                if success:
                    operation.actions_taken.append(
                        f"Successfully executed {strategy.failure_type_name} recovery on attempt {attempt + 1}"
                    )
                    return True

                if attempt < self.config.max_retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds)

            except Exception as e:
                logger.error(f"Recovery handler error: {e}")
                operation.actions_taken.append(f"Handler error: {str(e)}")

        return False

    async def _validate_recovery(self, operation: RecoveryOperation) -> bool:
        """
        Validate recovery operation success.

        Args:
            operation: Recovery operation

        Returns:
            Validation success status
        """
        try:
            validation_start = time.time()

            # Test all critical systems
            validations = {
                "redis_health": await self._test_redis_health(),
                "postgres_health": await self._test_postgres_health(),
                "data_integrity": not await self._detect_data_corruption(),
                "trading_engine": await self._test_trading_engine_health(),
                "critical_data": await self._validate_critical_data(),
            }

            # Update operation
            operation.validation_results = validations

            # Check if all critical validations passed
            critical_passed = all(
                [validations.get("data_integrity", False), validations.get("critical_data", False)]
            )

            validation_time = time.time() - validation_start
            operation.actions_taken.append(f"Validation completed in {validation_time:.2f}s")

            if not critical_passed:
                logger.warning(f"Recovery validation failed: {validations}")

            return critical_passed

        except Exception as e:
            logger.error(f"Recovery validation error: {e}")
            return False

    async def _test_redis_health(self) -> bool:
        """Test Redis connectivity"""
        try:
            if self.state_manager.redis_client:
                self.state_manager.redis_client.ping()
                return True
        except Exception as exc:
            logger.debug("Redis health check failed: %s", exc, exc_info=True)
        return False

    async def _test_postgres_health(self) -> bool:
        """Test PostgreSQL connectivity"""
        try:
            if self.state_manager.pg_conn:
                with self.state_manager.pg_conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                return True
        except Exception as exc:
            logger.debug("PostgreSQL health check failed: %s", exc, exc_info=True)
        return False

    async def _test_s3_health(self) -> bool:
        """Test S3 availability"""
        try:
            if self.state_manager.s3_client:
                self.state_manager.s3_client.head_bucket(Bucket=self.state_manager.config.s3_bucket)
                return True
        except Exception as exc:
            logger.debug("S3 health check failed: %s", exc, exc_info=True)
        return False

    async def _test_trading_engine_health(self) -> bool:
        """Test trading engine health"""
        try:
            status = await self.state_manager.get_state("system:trading_engine_status")
            return status not in [None, "crashed", "error"]
        except Exception as exc:
            logger.debug("Trading engine health check failed: %s", exc, exc_info=True)
            return False

    async def _detect_data_corruption(self) -> bool:
        """Detect data corruption through checksums"""
        try:
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

    async def _validate_critical_data(self) -> bool:
        """Validate presence of critical data"""
        try:
            # Check for critical data presence
            portfolio = await self.state_manager.get_state("portfolio_current")

            if not portfolio:
                logger.warning("Portfolio data missing")
                return False

            # Validate portfolio structure
            required_fields = ["positions", "cash_balance", "total_value"]
            for field in required_fields:
                if field not in portfolio:
                    logger.warning(f"Portfolio missing field: {field}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Critical data validation error: {e}")
            return False

    def _validate_position(self, position: dict[str, Any]) -> bool:
        """Validate position data structure"""
        required_fields = ["symbol", "quantity", "entry_price"]
        return all(field in position for field in required_fields)

    async def _check_memory_usage(self) -> float:
        """Check memory usage percentage"""
        try:
            import psutil

            return psutil.virtual_memory().percent
        except ImportError:
            logger.debug("psutil not available; cannot check memory usage")
        except Exception as exc:
            logger.debug("Memory usage check failed: %s", exc, exc_info=True)
        return 0

    async def _check_disk_usage(self) -> float:
        """Check disk usage percentage"""
        try:
            import psutil

            return psutil.disk_usage("/").percent
        except ImportError:
            logger.debug("psutil not available; cannot check disk usage")
        except Exception as exc:
            logger.debug("Disk usage check failed: %s", exc, exc_info=True)
        return 0

    async def _replay_transactions_from(self, timestamp: datetime) -> bool:
        """Replay transactions from given timestamp"""
        try:
            # This would replay transaction log if available
            # Placeholder for transaction replay logic
            logger.info(f"Would replay transactions from {timestamp}")
            return True
        except Exception as e:
            logger.error(f"Transaction replay failed: {e}")
            return False

    async def _synchronize_state(self) -> None:
        """Synchronize distributed state after network recovery"""
        try:
            # Re-sync critical state across tiers
            hot_keys = await self.state_manager.get_keys_by_pattern("position:*")

            for key in hot_keys:
                value = await self.state_manager.get_state(key)
                if value:
                    # Force write to ensure consistency
                    from bot_v2.state.state_manager import StateCategory

                    await self.state_manager.set_state(key, value, StateCategory.HOT)

            logger.info("State synchronization completed")

        except Exception as e:
            logger.error(f"State synchronization failed: {e}")

    async def _escalate_recovery(self, operation: RecoveryOperation) -> None:
        """Escalate failed recovery to manual intervention"""
        try:
            # Send high-priority alert
            await self._send_alert(
                f"URGENT: Manual intervention required for {operation.failure_event.failure_type.value}",
                operation,
                priority="high",
            )

            # Create manual recovery checklist
            checklist = self._generate_manual_recovery_checklist(operation)

            logger.critical(f"Manual recovery required. Checklist:\n{checklist}")

            # Store for operator access
            await self.state_manager.set_state(
                "system:manual_recovery_required",
                {
                    "operation_id": operation.operation_id,
                    "failure_type": operation.failure_event.failure_type.value,
                    "checklist": checklist,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"Recovery escalation failed: {e}")

    def _generate_manual_recovery_checklist(self, operation: RecoveryOperation) -> str:
        """Generate manual recovery checklist"""
        failure_type = operation.failure_event.failure_type

        checklists = {
            FailureType.DATA_CORRUPTION: """
            1. Stop all trading operations immediately
            2. Create emergency checkpoint
            3. Verify backup integrity
            4. Restore from last known good backup
            5. Validate all position data
            6. Reconcile with broker records
            7. Resume operations gradually
            """,
            FailureType.TRADING_ENGINE_CRASH: """
            1. Cancel all pending orders
            2. Verify position consistency
            3. Check for partially filled orders
            4. Reconcile with exchange
            5. Restart trading engine
            6. Run validation tests
            7. Resume trading with reduced limits
            """,
        }

        return checklists.get(
            failure_type, "1. Assess situation\n2. Contact support\n3. Follow runbook"
        )

    async def _send_alert(
        self, message: str, operation: RecoveryOperation, priority: str = "normal"
    ) -> None:
        """Send recovery alert"""
        try:
            alert_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "message": message,
                "priority": priority,
                "operation_id": operation.operation_id,
                "failure_type": operation.failure_event.failure_type.value,
                "recovery_mode": operation.recovery_mode.value,
                "status": operation.status.value,
            }

            # Log alert
            if priority == "high":
                logger.critical(f"ALERT: {message}")
            else:
                logger.warning(f"Alert: {message}")

            # Store alert
            await self.state_manager.set_state(
                f"alert:{operation.operation_id}:{datetime.utcnow().timestamp()}", alert_data
            )

            # Send to external systems (placeholder)
            # await self._send_to_slack(alert_data)
            # await self._send_email(alert_data)

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def _is_critical(self, failure_type: FailureType) -> bool:
        """Check if failure is critical"""
        critical_failures = {
            FailureType.DATA_CORRUPTION,
            FailureType.TRADING_ENGINE_CRASH,
            FailureType.POSTGRES_DOWN,
            FailureType.MEMORY_OVERFLOW,
        }
        return failure_type in critical_failures

    def _get_affected_components(self, failure_type: FailureType) -> list[str]:
        """Get components affected by failure"""
        affected_map = {
            FailureType.REDIS_DOWN: ["cache", "hot_storage", "real_time_data"],
            FailureType.POSTGRES_DOWN: ["warm_storage", "persistence", "history"],
            FailureType.S3_UNAVAILABLE: ["cold_storage", "backups", "archives"],
            FailureType.DATA_CORRUPTION: ["all_storage", "data_integrity"],
            FailureType.TRADING_ENGINE_CRASH: ["order_execution", "position_management"],
            FailureType.ML_MODEL_FAILURE: ["predictions", "strategy_selection"],
            FailureType.API_GATEWAY_DOWN: ["external_api", "user_interface"],
            FailureType.MEMORY_OVERFLOW: ["performance", "cache"],
            FailureType.DISK_FULL: ["storage", "logging", "checkpoints"],
        }
        return affected_map.get(failure_type, ["unknown"])

    def _generate_operation_id(self) -> str:
        """Generate unique operation ID"""
        import uuid

        return f"REC_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def _cleanup_recovery_history(self) -> None:
        """Clean up old recovery history"""
        # Keep only last 100 operations
        if len(self._recovery_history) > 100:
            self._recovery_history = self._recovery_history[-100:]

    def get_recovery_stats(self) -> dict[str, Any]:
        """Get recovery statistics"""
        if not self._recovery_history:
            return {"total_recoveries": 0, "success_rate": 0, "average_recovery_time": 0}

        successful = [op for op in self._recovery_history if op.status == RecoveryStatus.COMPLETED]

        total_time = sum(op.recovery_time_seconds for op in successful if op.recovery_time_seconds)

        return {
            "total_recoveries": len(self._recovery_history),
            "successful_recoveries": len(successful),
            "success_rate": len(successful) / len(self._recovery_history) * 100,
            "average_recovery_time": total_time / len(successful) if successful else 0,
            "rto_compliance": all(
                op.recovery_time_seconds <= self.config.rto_minutes * 60
                for op in successful
                if op.recovery_time_seconds
            ),
            "last_recovery": (
                self._recovery_history[-1].started_at if self._recovery_history else None
            ),
        }


# Convenience functions
async def detect_and_recover(
    state_manager: Any, checkpoint_handler: Any, backup_manager: Any = None
) -> RecoveryOperation | None:
    """Detect failures and initiate recovery if needed"""
    handler = RecoveryHandler(state_manager, checkpoint_handler, backup_manager)
    failures = await handler.detect_failures()

    if failures:
        # Handle most critical failure first
        critical = [f for f in failures if handler._is_critical(f)]

        if critical:
            event = FailureEvent(
                failure_type=critical[0],
                timestamp=datetime.utcnow(),
                severity="critical",
                affected_components=handler._get_affected_components(critical[0]),
                error_message=f"Detected: {critical[0].value}",
            )

            return await handler.initiate_recovery(event)

    return None
