"""Post-recovery validation and health verification"""

import logging
import time
from typing import Any

from bot_v2.state.recovery.detection import FailureDetector
from bot_v2.state.recovery.models import RecoveryOperation

logger = logging.getLogger(__name__)


class RecoveryValidator:
    """Validates recovery operations and system health"""

    def __init__(self, state_manager, checkpoint_handler) -> None:
        self.state_manager = state_manager
        self.detector = FailureDetector(state_manager, checkpoint_handler)

    async def validate_recovery(self, operation: RecoveryOperation) -> bool:
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
                "redis_health": await self.detector.test_redis_health(),
                "postgres_health": await self.detector.test_postgres_health(),
                "data_integrity": not await self.detector.detect_data_corruption(),
                "trading_engine": await self.detector.test_trading_engine_health(),
                "critical_data": await self.validate_critical_data(),
            }

            # Update operation
            operation.validation_results = validations

            # Check if all critical validations passed
            critical_passed = all(
                [
                    validations.get("data_integrity", False),
                    validations.get("critical_data", False),
                ]
            )

            validation_time = time.time() - validation_start
            operation.actions_taken.append(f"Validation completed in {validation_time:.2f}s")

            if not critical_passed:
                logger.warning(f"Recovery validation failed: {validations}")

            return critical_passed

        except Exception as e:
            logger.error(f"Recovery validation error: {e}")
            return False

    async def validate_critical_data(self) -> bool:
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

    def validate_position(self, position: dict[str, Any]) -> bool:
        """Validate position data structure"""
        required_fields = ["symbol", "quantity", "entry_price"]
        return all(field in position for field in required_fields)
