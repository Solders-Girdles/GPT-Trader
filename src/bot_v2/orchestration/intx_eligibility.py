"""
INTX eligibility verification with fail-closed logic.

Programmatically verifies Coinbase International Exchange (INTX) eligibility
before allowing derivatives trading. Implements fail-closed behavior to prevent
orders when permissions are missing or invalid.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="intx_eligibility")


class IntxEligibilityStatus(Enum):
    """INTX eligibility status."""

    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    UNKNOWN = "unknown"
    CHECKING = "checking"


@dataclass
class IntxEligibilityResult:
    """Result of INTX eligibility verification."""

    status: IntxEligibilityStatus
    portfolio_uuid: str | None
    api_mode: str | None
    supports_intx: bool
    verification_time: datetime
    error_message: str | None = None
    warnings: list[str] | None = None

    def is_eligible(self) -> bool:
        """Check if eligible for INTX trading."""
        return self.status == IntxEligibilityStatus.ELIGIBLE

    def should_allow_trading(self) -> bool:
        """
        Determine if trading should be allowed.

        Fail-closed: Only allow if explicitly eligible.
        """
        return self.is_eligible()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for logging/telemetry."""
        return {
            "status": self.status.value,
            "portfolio_uuid": self.portfolio_uuid,
            "api_mode": self.api_mode,
            "supports_intx": self.supports_intx,
            "verification_time": self.verification_time.isoformat(),
            "error_message": self.error_message,
            "warnings": self.warnings or [],
        }


class IntxEligibilityChecker:
    """
    Verifies INTX eligibility with fail-closed logic.

    Checks:
    1. API mode is "advanced" (not "exchange")
    2. Can resolve INTX portfolio UUID
    3. Portfolio UUID is valid

    Caching:
    - Eligible status cached for 1 hour
    - Ineligible status cached for 5 minutes
    - Unknown status not cached (retry immediately)

    Fail-closed behavior:
    - If verification fails, status = INELIGIBLE
    - Only ELIGIBLE status allows trading
    - Missing permissions = trading blocked
    """

    def __init__(
        self,
        *,
        intx_portfolio_service: Any,  # IntxPortfolioService
        cache_ttl_seconds: int = 3600,  # 1 hour for eligible
        cache_ttl_ineligible_seconds: int = 300,  # 5 minutes for ineligible
        require_portfolio_uuid: bool = True,
    ):
        """
        Initialize eligibility checker.

        Args:
            intx_portfolio_service: INTX portfolio service instance
            cache_ttl_seconds: Cache TTL for eligible status
            cache_ttl_ineligible_seconds: Cache TTL for ineligible status
            require_portfolio_uuid: Require valid portfolio UUID for eligibility
        """
        self.intx_service = intx_portfolio_service
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_ttl_ineligible_seconds = cache_ttl_ineligible_seconds
        self.require_portfolio_uuid = require_portfolio_uuid

        # Cache
        self._cached_result: IntxEligibilityResult | None = None
        self._last_verification: datetime | None = None

        # Stats
        self.total_checks = 0
        self.cache_hits = 0
        self.eligibility_changes: list[dict[str, Any]] = []

    def check_eligibility(self, *, force_refresh: bool = False) -> IntxEligibilityResult:
        """
        Check INTX eligibility.

        Args:
            force_refresh: Bypass cache and re-verify

        Returns:
            IntxEligibilityResult
        """
        self.total_checks += 1

        # Check cache first
        if not force_refresh and self._cached_result and self._last_verification:
            ttl = (
                self.cache_ttl_seconds
                if self._cached_result.is_eligible()
                else self.cache_ttl_ineligible_seconds
            )
            cache_age = (datetime.now() - self._last_verification).total_seconds()

            if cache_age < ttl:
                self.cache_hits += 1
                logger.debug(
                    "INTX eligibility cache hit | status=%s | age=%.0fs",
                    self._cached_result.status.value,
                    cache_age,
                )
                return self._cached_result

        # Perform fresh verification
        logger.info("Verifying INTX eligibility | force_refresh=%s", force_refresh)
        result = self._verify_eligibility()

        # Detect status changes
        if self._cached_result and self._cached_result.status != result.status:
            self._log_eligibility_change(old=self._cached_result, new=result)

        # Update cache
        self._cached_result = result
        self._last_verification = datetime.now()

        logger.info(
            "INTX eligibility verified | status=%s | portfolio_uuid=%s | api_mode=%s",
            result.status.value,
            result.portfolio_uuid,
            result.api_mode,
        )

        return result

    def _verify_eligibility(self) -> IntxEligibilityResult:
        """
        Perform actual eligibility verification.

        Returns:
            IntxEligibilityResult
        """
        warnings: list[str] = []
        error_message: str | None = None

        try:
            # Step 1: Check if broker supports INTX
            supports_intx = self.intx_service.supports_intx()

            if not supports_intx:
                return IntxEligibilityResult(
                    status=IntxEligibilityStatus.INELIGIBLE,
                    portfolio_uuid=None,
                    api_mode="unknown",
                    supports_intx=False,
                    verification_time=datetime.now(),
                    error_message="Broker does not support INTX (api_mode must be 'advanced')",
                )

            # Step 2: Try to get API mode
            api_mode = self._get_api_mode()

            if api_mode != "advanced":
                return IntxEligibilityResult(
                    status=IntxEligibilityStatus.INELIGIBLE,
                    portfolio_uuid=None,
                    api_mode=api_mode,
                    supports_intx=False,
                    verification_time=datetime.now(),
                    error_message=f"API mode is '{api_mode}', must be 'advanced' for INTX",
                )

            # Step 3: Try to resolve portfolio UUID
            try:
                portfolio_uuid = self.intx_service.get_portfolio_uuid(refresh=True)
            except Exception as e:
                logger.warning("Failed to resolve INTX portfolio UUID | error=%s", str(e))
                return IntxEligibilityResult(
                    status=IntxEligibilityStatus.INELIGIBLE,
                    portfolio_uuid=None,
                    api_mode=api_mode,
                    supports_intx=True,
                    verification_time=datetime.now(),
                    error_message=f"Failed to resolve portfolio UUID: {str(e)}",
                )

            # Step 4: Validate portfolio UUID
            if self.require_portfolio_uuid and not portfolio_uuid:
                return IntxEligibilityResult(
                    status=IntxEligibilityStatus.INELIGIBLE,
                    portfolio_uuid=None,
                    api_mode=api_mode,
                    supports_intx=True,
                    verification_time=datetime.now(),
                    error_message="No INTX portfolio UUID found (missing permissions or not enrolled)",
                )

            # Step 5: Warn if using override
            snapshot = self.intx_service.snapshot()
            if snapshot.get("override_uuid"):
                warnings.append(
                    f"Using override portfolio UUID: {snapshot['override_uuid']}"
                )

            # Success - eligible
            return IntxEligibilityResult(
                status=IntxEligibilityStatus.ELIGIBLE,
                portfolio_uuid=portfolio_uuid,
                api_mode=api_mode,
                supports_intx=True,
                verification_time=datetime.now(),
                warnings=warnings if warnings else None,
            )

        except Exception as e:
            # Fail closed on any unexpected error
            logger.exception("INTX eligibility check failed unexpectedly")
            return IntxEligibilityResult(
                status=IntxEligibilityStatus.INELIGIBLE,
                portfolio_uuid=None,
                api_mode=None,
                supports_intx=False,
                verification_time=datetime.now(),
                error_message=f"Eligibility check failed: {str(e)}",
            )

    def _get_api_mode(self) -> str:
        """Get API mode from service."""
        try:
            # Try to get from account manager via intx service
            account_manager = getattr(self.intx_service, "_account_manager", None)
            if account_manager:
                broker = getattr(account_manager, "broker", None)
                if broker:
                    config = getattr(broker, "config", None)
                    if config:
                        return getattr(config, "api_mode", "unknown")
            return "unknown"
        except Exception:
            logger.warning("Failed to get API mode, defaulting to 'unknown'")
            return "unknown"

    def _log_eligibility_change(
        self, *, old: IntxEligibilityResult, new: IntxEligibilityResult
    ) -> None:
        """Log eligibility status change."""
        change = {
            "timestamp": datetime.now().isoformat(),
            "old_status": old.status.value,
            "new_status": new.status.value,
            "old_portfolio_uuid": old.portfolio_uuid,
            "new_portfolio_uuid": new.portfolio_uuid,
        }

        self.eligibility_changes.append(change)

        logger.warning(
            "INTX eligibility status changed | old=%s | new=%s | portfolio_uuid=%s",
            old.status.value,
            new.status.value,
            new.portfolio_uuid,
        )

    def invalidate_cache(self) -> None:
        """Invalidate cached eligibility result."""
        self._cached_result = None
        self._last_verification = None
        logger.info("INTX eligibility cache invalidated")

    def get_stats(self) -> dict[str, Any]:
        """Get checker statistics."""
        cache_hit_rate = (
            self.cache_hits / self.total_checks if self.total_checks > 0 else 0.0
        )

        return {
            "total_checks": self.total_checks,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "eligibility_changes": len(self.eligibility_changes),
            "current_status": (
                self._cached_result.status.value if self._cached_result else "unknown"
            ),
            "last_verification": (
                self._last_verification.isoformat() if self._last_verification else None
            ),
        }


def create_fail_closed_checker(
    intx_portfolio_service: Any,
) -> IntxEligibilityChecker:
    """
    Create an eligibility checker with fail-closed defaults.

    Fail-closed means:
    - Strict verification required
    - Portfolio UUID must be valid
    - Short cache TTL for ineligible status
    - Any error = ineligible

    Args:
        intx_portfolio_service: INTX portfolio service

    Returns:
        IntxEligibilityChecker configured for fail-closed behavior
    """
    return IntxEligibilityChecker(
        intx_portfolio_service=intx_portfolio_service,
        cache_ttl_seconds=3600,  # 1 hour for eligible
        cache_ttl_ineligible_seconds=300,  # 5 minutes for ineligible
        require_portfolio_uuid=True,  # Strict requirement
    )


__all__ = [
    "IntxEligibilityChecker",
    "IntxEligibilityResult",
    "IntxEligibilityStatus",
    "create_fail_closed_checker",
]
