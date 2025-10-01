"""
Recovery orchestrator - main facade for failure recovery operations

Coordinates failure detection, handler execution, validation, and alerting.
"""

import asyncio
import inspect
import logging
import threading
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from bot_v2.state.recovery.alerting import RecoveryAlerter
from bot_v2.state.recovery.detection import FailureDetector
from bot_v2.state.recovery.handlers import (
    StorageRecoveryHandlers,
    SystemRecoveryHandlers,
    TradingRecoveryHandlers,
)
from bot_v2.state.recovery.models import (
    FailureEvent,
    FailureType,
    RecoveryConfig,
    RecoveryMode,
    RecoveryOperation,
    RecoveryStatus,
)
from bot_v2.state.recovery.validation import RecoveryValidator

logger = logging.getLogger(__name__)


class RecoveryOrchestrator:
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

        # Initialize components
        self.detector = FailureDetector(state_manager, checkpoint_handler)
        self.validator = RecoveryValidator(state_manager, checkpoint_handler)
        self.alerter = RecoveryAlerter(state_manager)

        # Initialize handlers
        self.storage_handlers = StorageRecoveryHandlers(
            state_manager, checkpoint_handler, backup_manager
        )
        self.trading_handlers = TradingRecoveryHandlers(state_manager)
        self.system_handlers = SystemRecoveryHandlers(state_manager, checkpoint_handler)

        # State tracking
        self._recovery_lock = threading.Lock()
        self._recovery_in_progress = False
        self._current_operation: RecoveryOperation | None = None
        self._recovery_history: list[RecoveryOperation] = []
        self._failure_handlers: dict[FailureType, Callable] = {}
        self._monitoring_task: asyncio.Task | None = None

        # Register handlers
        self._register_handlers()

        logger.info(
            f"RecoveryOrchestrator initialized with RTO={self.config.rto_minutes}min, "
            f"RPO={self.config.rpo_minutes}min"
        )

    def _register_handlers(self) -> None:
        """Register failure recovery handlers"""
        self._failure_handlers = {
            FailureType.REDIS_DOWN: self.storage_handlers.recover_redis,
            FailureType.POSTGRES_DOWN: self.storage_handlers.recover_postgres,
            FailureType.S3_UNAVAILABLE: self.storage_handlers.recover_s3,
            FailureType.DATA_CORRUPTION: self.storage_handlers.recover_from_corruption,
            FailureType.TRADING_ENGINE_CRASH: self.trading_handlers.recover_trading_engine,
            FailureType.ML_MODEL_FAILURE: self.trading_handlers.recover_ml_models,
            FailureType.MEMORY_OVERFLOW: self.system_handlers.recover_from_memory_overflow,
            FailureType.DISK_FULL: self.system_handlers.recover_from_disk_full,
            FailureType.NETWORK_PARTITION: self.system_handlers.recover_from_network_partition,
            FailureType.API_GATEWAY_DOWN: self.system_handlers.recover_api_gateway,
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
            task = self._monitoring_task

            cancel = getattr(task, "cancel", None)
            if callable(cancel):
                cancel()

            try:
                if isinstance(task, asyncio.Task) or inspect.isawaitable(task):
                    await task
            except asyncio.CancelledError:
                pass
            finally:
                self._monitoring_task = None
                logger.info("Recovery monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Continuous monitoring loop for failure detection"""
        while True:
            try:
                # Detect failures
                failures = await self.detector.detect_failures()

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
        """Detect system failures - delegates to FailureDetector"""
        return await self.detector.detect_failures()

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

        await self.alerter.send_alert(
            f"Recovery started for {failure_type}",
            operation,
        )

        success = await self._execute_recovery(operation)

        if success:
            await self._complete_recovery(operation)
        else:
            await self._handle_recovery_failure(operation, mode)

        await self.alerter.send_alert(
            f"Recovery {operation.status.value} for {operation.failure_event.failure_type.value}",
            operation,
        )

    async def _complete_recovery(self, operation: RecoveryOperation) -> None:
        operation.status = RecoveryStatus.VALIDATING
        if await self.validator.validate_recovery(operation):
            operation.status = RecoveryStatus.COMPLETED
            if not operation.completed_at:
                operation.completed_at = datetime.utcnow()
            if operation.recovery_time_seconds is None:
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
            await self.alerter.escalate_recovery(operation)

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

        # Get appropriate handler
        handler = self._failure_handlers.get(failure_type)

        if not handler:
            logger.error(f"No handler registered for {failure_type.value}")
            return False

        # Execute handler with retries
        for attempt in range(self.config.max_retry_attempts):
            try:
                logger.info(f"Recovery attempt {attempt + 1}/{self.config.max_retry_attempts}")

                success = await handler(operation)

                if success:
                    operation.actions_taken.append(
                        f"Successfully executed {handler.__name__} on attempt {attempt + 1}"
                    )
                    return True

                if attempt < self.config.max_retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds)

            except Exception as e:
                logger.error(f"Recovery handler error: {e}")
                operation.actions_taken.append(f"Handler error: {str(e)}")

        return False

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


# Convenience alias for backward compatibility
RecoveryHandler = RecoveryOrchestrator


# Convenience functions
async def detect_and_recover(
    state_manager: Any, checkpoint_handler: Any, backup_manager: Any = None
) -> RecoveryOperation | None:
    """Detect failures and initiate recovery if needed"""
    orchestrator = RecoveryOrchestrator(state_manager, checkpoint_handler, backup_manager)
    failures = await orchestrator.detect_failures()

    if failures:
        # Handle most critical failure first
        critical = [f for f in failures if orchestrator._is_critical(f)]

        if critical:
            event = FailureEvent(
                failure_type=critical[0],
                timestamp=datetime.utcnow(),
                severity="critical",
                affected_components=orchestrator._get_affected_components(critical[0]),
                error_message=f"Detected: {critical[0].value}",
            )

            return await orchestrator.initiate_recovery(event)

    return None
