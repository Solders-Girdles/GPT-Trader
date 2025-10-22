"""
INTX startup validation.

Validates INTX eligibility at bot startup when derivatives are enabled.
Fails startup if derivatives enabled but INTX not available (fail-closed).
"""

from bot_v2.orchestration.intx_eligibility import IntxEligibilityChecker, IntxEligibilityStatus
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="startup")


class IntxStartupValidationError(Exception):
    """INTX startup validation failed - cannot start bot."""


class IntxStartupValidator:
    """
    Validates INTX eligibility at bot startup.

    Ensures that if derivatives trading is enabled, INTX is properly
    configured and accessible. Prevents bot from starting if derivatives
    are enabled but INTX is not available.

    Fail-closed behavior:
    - Derivatives enabled + INTX ineligible = startup failure
    - Derivatives disabled = skip validation
    - Spot-only mode = skip validation
    """

    def __init__(
        self,
        *,
        eligibility_checker: IntxEligibilityChecker,
        enable_derivatives: bool = False,
        fail_closed: bool = True,
    ):
        """
        Initialize startup validator.

        Args:
            eligibility_checker: INTX eligibility checker
            enable_derivatives: Whether derivatives trading is enabled
            fail_closed: Whether to fail startup on eligibility issues
        """
        self.eligibility_checker = eligibility_checker
        self.enable_derivatives = enable_derivatives
        self.fail_closed = fail_closed

    def validate_on_startup(self) -> None:
        """
        Validate INTX eligibility at startup.

        Raises:
            IntxStartupValidationError: If validation fails and fail_closed=True
        """
        if not self.enable_derivatives:
            logger.info("Derivatives trading disabled, skipping INTX validation")
            return

        logger.info("Validating INTX eligibility at startup...")

        # Force fresh check (don't use cache)
        result = self.eligibility_checker.check_eligibility(force_refresh=True)

        if result.is_eligible():
            logger.info(
                "✅ INTX eligibility verified | portfolio_uuid=%s | api_mode=%s",
                result.portfolio_uuid,
                result.api_mode,
            )

            if result.warnings:
                for warning in result.warnings:
                    logger.warning("INTX warning: %s", warning)

            return

        # Not eligible
        error_msg = self._build_startup_error_message(result)

        logger.error(
            "❌ INTX eligibility check FAILED | status=%s | error=%s",
            result.status.value,
            result.error_message,
        )

        if self.fail_closed:
            raise IntxStartupValidationError(error_msg)
        else:
            logger.warning(
                "INTX not eligible but fail_closed=False, allowing startup anyway"
            )

    def _build_startup_error_message(self, result: any) -> str:
        """Build detailed startup error message."""
        lines = [
            "=" * 80,
            "INTX ELIGIBILITY CHECK FAILED",
            "=" * 80,
            "",
            "Derivatives trading is enabled but INTX is not available.",
            "",
            f"Status: {result.status.value}",
            f"API Mode: {result.api_mode or 'unknown'}",
            f"Portfolio UUID: {result.portfolio_uuid or 'not found'}",
            "",
        ]

        if result.error_message:
            lines.extend([
                "Error:",
                f"  {result.error_message}",
                "",
            ])

        lines.extend([
            "Possible Solutions:",
            "",
            "1. Verify INTX Enrollment:",
            "   - Ensure your Coinbase account is approved for INTX",
            "   - Check that you have completed INTX onboarding",
            "   - Verify institutional entitlements on your API key",
            "",
            "2. Check API Configuration:",
            "   - Ensure COINBASE_API_MODE=advanced (not 'exchange')",
            "   - Verify API key has derivatives permissions",
            "   - Check that COINBASE_INTX_PORTFOLIO_UUID is set (if using override)",
            "",
            "3. Disable Derivatives (if not ready):",
            "   - Set COINBASE_ENABLE_DERIVATIVES=0",
            "   - Or remove derivatives_enabled from bot config",
            "   - Trade spot-only until INTX is ready",
            "",
            "4. Contact Coinbase:",
            "   - If you believe you should have INTX access",
            "   - Check institutional@coinbase.com for enrollment",
            "",
            "=" * 80,
            "",
            "Bot startup FAILED to prevent invalid derivatives trading.",
            "Fix INTX eligibility or disable derivatives to continue.",
            "",
            "=" * 80,
        ])

        return "\n".join(lines)


def validate_intx_on_startup(
    *,
    eligibility_checker: IntxEligibilityChecker,
    enable_derivatives: bool = False,
    fail_closed: bool = True,
) -> None:
    """
    Convenience function to validate INTX on startup.

    Args:
        eligibility_checker: INTX eligibility checker
        enable_derivatives: Whether derivatives are enabled
        fail_closed: Whether to fail on validation errors

    Raises:
        IntxStartupValidationError: If validation fails and fail_closed=True
    """
    validator = IntxStartupValidator(
        eligibility_checker=eligibility_checker,
        enable_derivatives=enable_derivatives,
        fail_closed=fail_closed,
    )

    validator.validate_on_startup()


__all__ = [
    "IntxStartupValidator",
    "IntxStartupValidationError",
    "validate_intx_on_startup",
]
