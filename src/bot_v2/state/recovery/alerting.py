"""Alert dispatch and recovery escalation"""

import logging
from datetime import datetime

from bot_v2.state.recovery.models import FailureType, RecoveryOperation

logger = logging.getLogger(__name__)


class RecoveryAlerter:
    """Handles alert dispatch and recovery escalation"""

    def __init__(self, state_manager) -> None:
        self.state_manager = state_manager

    async def send_alert(
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

    async def escalate_recovery(self, operation: RecoveryOperation) -> None:
        """Escalate failed recovery to manual intervention"""
        try:
            # Send high-priority alert
            await self.send_alert(
                f"URGENT: Manual intervention required for {operation.failure_event.failure_type.value}",
                operation,
                priority="high",
            )

            # Create manual recovery checklist
            checklist = self.generate_manual_recovery_checklist(operation)

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

    def generate_manual_recovery_checklist(self, operation: RecoveryOperation) -> str:
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
