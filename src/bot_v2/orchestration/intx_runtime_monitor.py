"""
INTX runtime permission monitoring.

Periodically re-verifies INTX eligibility during bot operation to detect
permission changes, revocations, or API key issues.
"""

from datetime import datetime, timedelta
from typing import Any

from bot_v2.orchestration.intx_eligibility import IntxEligibilityChecker, IntxEligibilityStatus
from bot_v2.persistence.event_store import EventStore
from bot_v2.utilities.logging_patterns import get_logger
from bot_v2.utilities.telemetry import emit_metric

logger = get_logger(__name__, component="intx_monitor")


class IntxRuntimeMonitor:
    """
    Monitors INTX eligibility during runtime.

    Periodically re-checks eligibility to detect:
    - Permission revocations
    - API key changes
    - Portfolio UUID changes
    - API mode changes

    Actions on eligibility loss:
    - Emit warning metrics
    - Log alerts
    - Optionally trigger reduce-only mode
    """

    def __init__(
        self,
        *,
        eligibility_checker: IntxEligibilityChecker,
        event_store: EventStore,
        check_interval_minutes: int = 60,  # Check every hour
        enable_derivatives: bool = False,
    ):
        """
        Initialize runtime monitor.

        Args:
            eligibility_checker: INTX eligibility checker
            event_store: Event store for metrics
            check_interval_minutes: Minutes between eligibility checks
            enable_derivatives: Whether derivatives are enabled
        """
        self.eligibility_checker = eligibility_checker
        self.event_store = event_store
        self.check_interval = timedelta(minutes=check_interval_minutes)
        self.enable_derivatives = enable_derivatives

        self.last_check: datetime | None = None
        self.last_status: IntxEligibilityStatus | None = None
        self.permission_loss_detected: bool = False

    def check_if_due(self) -> bool:
        """
        Check if eligibility check is due.

        Returns:
            True if check should be performed
        """
        if not self.enable_derivatives:
            return False

        if self.last_check is None:
            return True

        time_since_check = datetime.now() - self.last_check
        return time_since_check >= self.check_interval

    def run_periodic_check(self) -> None:
        """
        Run periodic eligibility check if due.

        Call this method in your main trading loop.
        """
        if not self.check_if_due():
            return

        logger.debug("Running periodic INTX eligibility check...")

        result = self.eligibility_checker.check_eligibility(force_refresh=False)

        self.last_check = datetime.now()

        # Detect status changes
        if self.last_status and self.last_status != result.status:
            self._handle_status_change(old_status=self.last_status, new_status=result.status)

        # Detect permission loss
        if result.status != IntxEligibilityStatus.ELIGIBLE and not self.permission_loss_detected:
            self._handle_permission_loss(result)
            self.permission_loss_detected = True

        # Detect permission restoration
        if result.status == IntxEligibilityStatus.ELIGIBLE and self.permission_loss_detected:
            self._handle_permission_restored(result)
            self.permission_loss_detected = False

        self.last_status = result.status

        logger.debug(
            "Periodic INTX check complete | status=%s | portfolio_uuid=%s",
            result.status.value,
            result.portfolio_uuid,
        )

    def _handle_status_change(
        self,
        *,
        old_status: IntxEligibilityStatus,
        new_status: IntxEligibilityStatus,
    ) -> None:
        """Handle eligibility status change."""
        logger.warning(
            "INTX eligibility status changed | old=%s | new=%s",
            old_status.value,
            new_status.value,
        )

        emit_metric(
            self.event_store,
            "intx_monitor",
            {
                "event_type": "eligibility_status_change",
                "old_status": old_status.value,
                "new_status": new_status.value,
                "component": "intx_runtime_monitor",
            },
            logger=logger,
        )

    def _handle_permission_loss(self, result: any) -> None:
        """Handle INTX permission loss."""
        logger.error(
            "⚠️  INTX PERMISSION LOSS DETECTED | status=%s | error=%s",
            result.status.value,
            result.error_message,
        )

        emit_metric(
            self.event_store,
            "intx_monitor",
            {
                "event_type": "permission_loss",
                "status": result.status.value,
                "error_message": result.error_message,
                "portfolio_uuid": result.portfolio_uuid,
                "component": "intx_runtime_monitor",
                "severity": "critical",
            },
            logger=logger,
        )

        # Log detailed alert
        logger.error(
            """
╔════════════════════════════════════════════════════════════════╗
║                  INTX PERMISSION LOSS DETECTED                  ║
╚════════════════════════════════════════════════════════════════╝

Status: %s
Error: %s

Derivatives trading will be blocked until permissions are restored.

Possible Causes:
- API key revoked or changed
- INTX entitlements removed
- Portfolio UUID changed
- API mode changed from 'advanced'

Actions Required:
1. Check Coinbase account status
2. Verify API key permissions
3. Check INTX enrollment status
4. Review account notifications

Current orders will continue, but new derivatives orders will be rejected.
        """,
            result.status.value,
            result.error_message or "Unknown",
        )

    def _handle_permission_restored(self, result: any) -> None:
        """Handle INTX permission restoration."""
        logger.info(
            "✅ INTX permissions restored | portfolio_uuid=%s",
            result.portfolio_uuid,
        )

        emit_metric(
            self.event_store,
            "intx_monitor",
            {
                "event_type": "permission_restored",
                "portfolio_uuid": result.portfolio_uuid,
                "component": "intx_runtime_monitor",
            },
            logger=logger,
        )

    def get_status_summary(self) -> dict[str, Any]:
        """
        Get current monitoring status.

        Returns:
            Dict with monitoring status
        """
        return {
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_status": self.last_status.value if self.last_status else None,
            "permission_loss_detected": self.permission_loss_detected,
            "check_interval_minutes": self.check_interval.total_seconds() / 60,
            "enabled": self.enable_derivatives,
        }


def create_runtime_monitor(
    *,
    eligibility_checker: IntxEligibilityChecker,
    event_store: EventStore,
    enable_derivatives: bool = False,
    check_interval_minutes: int = 60,
) -> IntxRuntimeMonitor:
    """
    Create INTX runtime monitor.

    Args:
        eligibility_checker: Eligibility checker instance
        event_store: Event store for metrics
        enable_derivatives: Whether derivatives are enabled
        check_interval_minutes: Minutes between checks

    Returns:
        IntxRuntimeMonitor configured for periodic checks
    """
    return IntxRuntimeMonitor(
        eligibility_checker=eligibility_checker,
        event_store=event_store,
        check_interval_minutes=check_interval_minutes,
        enable_derivatives=enable_derivatives,
    )


__all__ = ["IntxRuntimeMonitor", "create_runtime_monitor"]
