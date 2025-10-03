"""
Recovery Workflow - High-level recovery execution flow

Encapsulates the detection-to-completion workflow:
validation → handler execution → post-validation → alerting → escalation.

Extracted from RecoveryOrchestrator to improve separation of concerns.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime

from bot_v2.state.recovery.alerting import RecoveryAlerter
from bot_v2.state.recovery.handler_registry import RecoveryHandlerRegistry
from bot_v2.state.recovery.models import (
    RecoveryConfig,
    RecoveryMode,
    RecoveryOperation,
    RecoveryStatus,
)
from bot_v2.state.recovery.validation import RecoveryValidator

logger = logging.getLogger(__name__)


@dataclass
class RecoveryOutcome:
    """
    Structured result of recovery workflow execution.

    Captures all outcomes, metrics, and escalation requirements
    for clean integration with orchestrator.
    """

    success: bool
    status: RecoveryStatus
    actions_taken: list[str] = field(default_factory=list)
    validation_passed: bool = False
    recovery_time_seconds: float | None = None
    escalation_required: bool = False
    error_message: str | None = None
    rto_exceeded: bool = False


class RecoveryWorkflow:
    """
    Executes high-level recovery workflow.

    Responsibilities:
    - Execute recovery handlers with retry logic
    - Validate recovery success
    - Manage status transitions
    - Coordinate alerting
    - Determine escalation requirements

    Does NOT manage:
    - Operation creation (orchestrator)
    - Recovery history (orchestrator)
    - Concurrency locks (orchestrator)
    - Background monitoring (monitor)
    """

    def __init__(
        self,
        handler_registry: RecoveryHandlerRegistry,
        validator: RecoveryValidator,
        alerter: RecoveryAlerter,
        config: RecoveryConfig | None = None,
    ) -> None:
        """
        Initialize recovery workflow.

        Args:
            handler_registry: Registry of recovery handlers
            validator: Recovery validator
            alerter: Alert dispatcher
            config: Recovery configuration
        """
        self.handler_registry = handler_registry
        self.validator = validator
        self.alerter = alerter
        self.config = config or RecoveryConfig()

    async def execute(
        self,
        operation: RecoveryOperation,
        mode: RecoveryMode = RecoveryMode.AUTOMATIC,
    ) -> RecoveryOutcome:
        """
        Execute complete recovery workflow for an operation.

        Args:
            operation: Recovery operation to execute
            mode: Recovery mode (automatic or manual)

        Returns:
            RecoveryOutcome with execution results
        """
        failure_type = operation.failure_event.failure_type.value

        logger.info(
            "Starting recovery workflow for %s (operation %s) in %s mode",
            failure_type,
            operation.operation_id,
            mode.value,
        )

        # Send start alert
        await self.alerter.send_alert(
            f"Recovery started for {failure_type}",
            operation,
        )

        # Execute handler with retries
        execution_success = await self._execute_handler(operation)

        if execution_success:
            # Validate and complete
            outcome = await self._complete_recovery(operation)
        else:
            # Handle failure
            outcome = self._handle_failure(operation, mode)

        # Send completion alert
        await self.alerter.send_alert(
            f"Recovery {outcome.status.value} for {failure_type}",
            operation,
        )

        # Escalate if needed
        if outcome.escalation_required:
            await self.alerter.escalate_recovery(operation)

        return outcome

    async def _execute_handler(self, operation: RecoveryOperation) -> bool:
        """
        Execute recovery handler with retry logic.

        Args:
            operation: Recovery operation

        Returns:
            True if handler succeeded, False otherwise
        """
        failure_type = operation.failure_event.failure_type

        # Get handler from registry
        handler = self.handler_registry.get_handler(failure_type)

        if not handler:
            logger.error(f"No handler registered for {failure_type.value}")
            operation.actions_taken.append(f"No handler found for {failure_type.value}")
            return False

        # Execute with retries
        for attempt in range(self.config.max_retry_attempts):
            try:
                logger.info(
                    "Recovery attempt %d/%d for %s",
                    attempt + 1,
                    self.config.max_retry_attempts,
                    failure_type.value,
                )

                success = await handler(operation)

                if success:
                    operation.actions_taken.append(
                        f"Successfully executed {getattr(handler, '__name__', 'handler')} "
                        f"on attempt {attempt + 1}"
                    )
                    return True

                # Wait before retry (unless last attempt)
                if attempt < self.config.max_retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds)

            except Exception as e:
                logger.error(f"Recovery handler error on attempt {attempt + 1}: {e}")
                operation.actions_taken.append(f"Handler error: {str(e)}")

                # Wait before retry (unless last attempt)
                if attempt < self.config.max_retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds)

        return False

    async def _complete_recovery(self, operation: RecoveryOperation) -> RecoveryOutcome:
        """
        Complete recovery with validation and metrics.

        Args:
            operation: Recovery operation

        Returns:
            RecoveryOutcome with completion details
        """
        # Set validating status
        operation.status = RecoveryStatus.VALIDATING

        # Run validation
        validation_passed = await self.validator.validate_recovery(operation)

        if validation_passed:
            # Mark completed
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

            # Check RTO compliance
            rto_exceeded = operation.recovery_time_seconds > self.config.rto_minutes * 60
            if rto_exceeded:
                logger.warning(
                    "Recovery exceeded RTO: %.2fs > %.2fs",
                    operation.recovery_time_seconds,
                    self.config.rto_minutes * 60,
                )

            return RecoveryOutcome(
                success=True,
                status=RecoveryStatus.COMPLETED,
                actions_taken=operation.actions_taken.copy(),
                validation_passed=True,
                recovery_time_seconds=operation.recovery_time_seconds,
                escalation_required=False,
                rto_exceeded=rto_exceeded,
            )
        else:
            # Validation failed - partial recovery
            operation.status = RecoveryStatus.PARTIAL
            logger.warning(f"Recovery {operation.operation_id} validation failed")

            return RecoveryOutcome(
                success=False,
                status=RecoveryStatus.PARTIAL,
                actions_taken=operation.actions_taken.copy(),
                validation_passed=False,
                escalation_required=False,
                error_message="Validation failed after handler execution",
            )

    def _handle_failure(
        self,
        operation: RecoveryOperation,
        mode: RecoveryMode,
    ) -> RecoveryOutcome:
        """
        Handle recovery failure and determine escalation.

        Args:
            operation: Recovery operation
            mode: Recovery mode

        Returns:
            RecoveryOutcome with failure details
        """
        operation.status = RecoveryStatus.FAILED

        # Escalate if automatic mode
        escalation_required = mode == RecoveryMode.AUTOMATIC
        if escalation_required:
            logger.info("Automatic recovery failed, escalating to manual mode")

        return RecoveryOutcome(
            success=False,
            status=RecoveryStatus.FAILED,
            actions_taken=operation.actions_taken.copy(),
            validation_passed=False,
            escalation_required=escalation_required,
            error_message="Handler execution failed after all retry attempts",
        )
