"""
INTX eligibility pre-trade validation.

Validates INTX eligibility before allowing derivatives orders.
Implements fail-closed logic: orders rejected if eligibility cannot be verified.
"""

from decimal import Decimal

from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.orchestration.intx_eligibility import IntxEligibilityChecker, IntxEligibilityStatus
from bot_v2.persistence.event_store import EventStore
from bot_v2.utilities.logging_patterns import get_logger
from bot_v2.utilities.telemetry import emit_metric

logger = get_logger(__name__, component="live_trade_risk")


class IntxEligibilityViolation(Exception):
    """INTX eligibility violation - order rejected."""


class IntxPreTradeValidator:
    """
    Pre-trade validator for INTX eligibility.

    Ensures that derivatives orders are only allowed when:
    1. INTX eligibility has been verified
    2. Portfolio UUID is valid
    3. API mode is correct
    4. Permissions are current

    Fail-closed behavior:
    - If eligibility check fails → order rejected
    - If eligibility unknown → order rejected
    - Only ELIGIBLE status allows orders
    """

    def __init__(
        self,
        *,
        eligibility_checker: IntxEligibilityChecker,
        event_store: EventStore,
        enable_derivatives: bool = False,
    ):
        """
        Initialize INTX pre-trade validator.

        Args:
            eligibility_checker: INTX eligibility checker instance
            event_store: Event store for metrics
            enable_derivatives: Whether derivatives trading is enabled
        """
        self.eligibility_checker = eligibility_checker
        self.event_store = event_store
        self.enable_derivatives = enable_derivatives

        # Stats
        self.total_checks = 0
        self.rejections = 0
        self.approvals = 0

    def validate_intx_eligibility(
        self,
        *,
        symbol: str,
        side: str,
        quantity: Decimal,
        product: Product | None = None,
    ) -> None:
        """
        Validate INTX eligibility before placing order.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            product: Product metadata (optional)

        Raises:
            IntxEligibilityViolation: If not eligible for INTX trading
        """
        self.total_checks += 1

        # Skip check if derivatives not enabled
        if not self.enable_derivatives:
            return

        # Skip check for spot products
        if product and product.market_type == MarketType.SPOT:
            return

        # Check if symbol looks like a derivatives product
        if not self._is_derivatives_symbol(symbol):
            return

        logger.debug(
            "Checking INTX eligibility | symbol=%s | side=%s | qty=%s",
            symbol,
            side,
            quantity,
        )

        # Perform eligibility check
        result = self.eligibility_checker.check_eligibility()

        if not result.should_allow_trading():
            self.rejections += 1

            # Emit rejection metric
            emit_metric(
                self.event_store,
                "risk_engine",
                {
                    "event_type": "intx_eligibility_rejection",
                    "symbol": symbol,
                    "status": result.status.value,
                    "error_message": result.error_message,
                    "component": "intx_validator",
                },
                logger=logger,
            )

            # Build error message
            error_msg = self._build_rejection_message(result)

            logger.warning(
                "INTX eligibility check FAILED | symbol=%s | status=%s | error=%s",
                symbol,
                result.status.value,
                result.error_message,
            )

            raise IntxEligibilityViolation(error_msg)

        # Eligible - allow order
        self.approvals += 1

        logger.debug(
            "INTX eligibility check PASSED | symbol=%s | portfolio_uuid=%s",
            symbol,
            result.portfolio_uuid,
        )

    def _is_derivatives_symbol(self, symbol: str) -> bool:
        """
        Check if symbol appears to be a derivatives product.

        Args:
            symbol: Trading symbol

        Returns:
            True if likely a derivatives symbol
        """
        # Common derivatives suffixes
        derivatives_patterns = [
            "-PERP",  # Perpetuals
            "-FUTURE",  # Futures
            "PERP",  # Alternative format
        ]

        symbol_upper = symbol.upper()

        for pattern in derivatives_patterns:
            if pattern in symbol_upper:
                return True

        return False

    def _build_rejection_message(self, result: any) -> str:
        """Build user-friendly rejection message."""
        if result.status == IntxEligibilityStatus.INELIGIBLE:
            if result.error_message:
                return f"INTX eligibility check failed: {result.error_message}"
            return "INTX trading not available - check API permissions and enrollment"

        elif result.status == IntxEligibilityStatus.UNKNOWN:
            return "INTX eligibility could not be verified - trading blocked (fail-closed)"

        else:
            return f"INTX trading blocked - status: {result.status.value}"

    def get_stats(self) -> dict[str, any]:
        """Get validator statistics."""
        rejection_rate = (
            self.rejections / self.total_checks if self.total_checks > 0 else 0.0
        )

        return {
            "total_checks": self.total_checks,
            "approvals": self.approvals,
            "rejections": self.rejections,
            "rejection_rate": rejection_rate,
        }


def create_intx_validator(
    *,
    eligibility_checker: IntxEligibilityChecker,
    event_store: EventStore,
    enable_derivatives: bool = False,
) -> IntxPreTradeValidator:
    """
    Create INTX pre-trade validator.

    Args:
        eligibility_checker: Eligibility checker instance
        event_store: Event store for metrics
        enable_derivatives: Whether derivatives are enabled

    Returns:
        IntxPreTradeValidator configured for fail-closed behavior
    """
    return IntxPreTradeValidator(
        eligibility_checker=eligibility_checker,
        event_store=event_store,
        enable_derivatives=enable_derivatives,
    )


__all__ = [
    "IntxPreTradeValidator",
    "IntxEligibilityViolation",
    "create_intx_validator",
]
