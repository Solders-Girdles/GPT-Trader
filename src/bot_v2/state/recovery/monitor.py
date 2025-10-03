"""
Recovery Monitor - Background failure detection and recovery initiation

Manages periodic health checks and automatic recovery initiation.
Extracted from RecoveryOrchestrator for separation of concerns.
"""

import asyncio
import inspect
import logging
from collections.abc import Callable
from datetime import datetime

from bot_v2.state.recovery.detection import FailureDetector
from bot_v2.state.recovery.models import (
    FailureEvent,
    FailureType,
    RecoveryConfig,
    RecoveryMode,
)

logger = logging.getLogger(__name__)


class RecoveryMonitor:
    """
    Background monitor for continuous failure detection.

    Responsibilities:
    - Periodic health checks via FailureDetector
    - Automatic recovery initiation for critical failures
    - Monitoring lifecycle management (start/stop)
    - Error handling with backoff

    Does NOT manage:
    - Recovery execution (delegated to workflow/orchestrator)
    - Recovery history (orchestrator)
    - Handler registration (registry)
    """

    def __init__(
        self,
        detector: FailureDetector,
        recovery_initiator: Callable,  # async callable(FailureEvent, RecoveryMode)
        is_critical_checker: Callable[[FailureType], bool],
        affected_components_getter: Callable[[FailureType], list[str]],
        recovery_in_progress_checker: Callable[[], bool],
        config: RecoveryConfig | None = None,
    ) -> None:
        """
        Initialize recovery monitor.

        Args:
            detector: Failure detection service
            recovery_initiator: Async callable to initiate recovery
            is_critical_checker: Function to check if failure is critical
            affected_components_getter: Function to get affected components
            recovery_in_progress_checker: Function to check if recovery in progress
            config: Recovery configuration
        """
        self.detector = detector
        self.recovery_initiator = recovery_initiator
        self.is_critical_checker = is_critical_checker
        self.affected_components_getter = affected_components_getter
        self.recovery_in_progress_checker = recovery_in_progress_checker
        self.config = config or RecoveryConfig()

        self._monitoring_task: asyncio.Task | None = None
        self._stop_requested = False

    async def start(self) -> None:
        """Start continuous failure detection monitoring."""
        if self._monitoring_task:
            logger.warning("Monitoring already started")
            return

        self._stop_requested = False
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Recovery monitoring started")

    async def stop(self) -> None:
        """Stop failure detection monitoring."""
        if not self._monitoring_task:
            return

        self._stop_requested = True
        task = self._monitoring_task

        # Cancel task
        cancel = getattr(task, "cancel", None)
        if callable(cancel):
            cancel()

        # Wait for cancellation
        try:
            if isinstance(task, asyncio.Task) or inspect.isawaitable(task):
                await task
        except asyncio.CancelledError:
            pass
        finally:
            self._monitoring_task = None
            logger.info("Recovery monitoring stopped")

    def is_running(self) -> bool:
        """Check if monitoring is currently running."""
        return self._monitoring_task is not None and not self._monitoring_task.done()

    async def tick(self) -> int:
        """
        Execute single monitoring cycle (for testing).

        Returns:
            Number of recoveries initiated
        """
        recoveries_initiated = 0

        try:
            # Detect failures
            failures = await self.detector.detect_failures()

            if failures and self.config.automatic_recovery_enabled:
                # Prioritize by severity
                critical_failures = [f for f in failures if self.is_critical_checker(f)]

                for failure in critical_failures:
                    if not self.recovery_in_progress_checker():
                        # Create failure event
                        event = FailureEvent(
                            failure_type=failure,
                            timestamp=datetime.utcnow(),
                            severity="critical",
                            affected_components=self.affected_components_getter(failure),
                            error_message=f"Automatic detection: {failure.value}",
                        )

                        # Initiate recovery
                        await self.recovery_initiator(event, RecoveryMode.AUTOMATIC)
                        recoveries_initiated += 1

        except Exception as e:
            logger.error(f"Monitoring tick error: {e}")

        return recoveries_initiated

    async def _monitoring_loop(self) -> None:
        """Continuous monitoring loop for failure detection."""
        while not self._stop_requested:
            try:
                # Detect failures
                failures = await self.detector.detect_failures()

                if failures and self.config.automatic_recovery_enabled:
                    # Prioritize by severity
                    critical_failures = [f for f in failures if self.is_critical_checker(f)]

                    for failure in critical_failures:
                        if not self.recovery_in_progress_checker():
                            # Create failure event
                            event = FailureEvent(
                                failure_type=failure,
                                timestamp=datetime.utcnow(),
                                severity="critical",
                                affected_components=self.affected_components_getter(failure),
                                error_message=f"Automatic detection: {failure.value}",
                            )

                            # Initiate recovery
                            await self.recovery_initiator(event, RecoveryMode.AUTOMATIC)

                await asyncio.sleep(self.config.failure_detection_interval_seconds)

            except asyncio.CancelledError:
                # Clean shutdown
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                # Back off on error
                await asyncio.sleep(self.config.failure_detection_interval_seconds * 2)
