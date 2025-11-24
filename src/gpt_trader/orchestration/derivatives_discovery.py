"""Derivatives eligibility discovery and safety gating for US and INTX derivatives.

This module implements startup checks to discover futures/perps portfolio and eligibility
by calling Coinbase Advanced Trade derivatives endpoints. If derivatives are enabled but
not accessible, the system will hard-fail to reduce-only mode or disable trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.features.brokerages.core.interfaces import IBrokerage

logger = get_logger(__name__, component="derivatives_discovery")


@dataclass(frozen=True)
class DerivativesEligibility:
    """Result of derivatives eligibility discovery."""

    us_derivatives_enabled: bool
    intx_derivatives_enabled: bool
    cfm_portfolio_accessible: bool
    intx_portfolio_accessible: bool
    error_message: str | None
    reduce_only_required: bool
    discovery_data: dict[str, Any]


DerivativesMarket = Literal["US", "INTX", "BOTH", "NONE"]


def discover_derivatives_eligibility(
    broker: IBrokerage,
    *,
    requested_market: DerivativesMarket = "BOTH",
    fail_on_inaccessible: bool = True,
) -> DerivativesEligibility:
    """Discover derivatives eligibility by probing portfolio endpoints.

    Args:
        broker: The brokerage instance to query
        requested_market: Which derivatives market is requested (US, INTX, BOTH, NONE)
        fail_on_inaccessible: If True, require reduce-only mode when derivatives
            are requested but not accessible

    Returns:
        DerivativesEligibility with discovery results and safety recommendations
    """
    logger.info(
        "Discovering derivatives eligibility",
        operation="derivatives_discovery",
        stage="start",
        requested_market=requested_market,
    )

    discovery_data: dict[str, Any] = {
        "requested_market": requested_market,
        "timestamp": None,
    }

    us_enabled = False
    intx_enabled = False
    cfm_accessible = False
    intx_accessible = False
    error_message: str | None = None

    # Skip discovery if derivatives not requested
    if requested_market == "NONE":
        logger.info(
            "Derivatives not requested, skipping discovery",
            operation="derivatives_discovery",
            stage="skip",
        )
        return DerivativesEligibility(
            us_derivatives_enabled=False,
            intx_derivatives_enabled=False,
            cfm_portfolio_accessible=False,
            intx_portfolio_accessible=False,
            error_message=None,
            reduce_only_required=False,
            discovery_data=discovery_data,
        )

    # Attempt US derivatives (CFM) discovery
    if requested_market in ("US", "BOTH"):
        us_enabled, cfm_accessible, us_error = _discover_us_derivatives(broker)
        if us_error:
            error_message = us_error if error_message is None else f"{error_message}; {us_error}"
        discovery_data["us_derivatives"] = {
            "enabled": us_enabled,
            "accessible": cfm_accessible,
            "error": us_error,
        }

    # Attempt INTX derivatives discovery
    if requested_market in ("INTX", "BOTH"):
        intx_enabled, intx_accessible, intx_error = _discover_intx_derivatives(broker)
        if intx_error:
            error_message = (
                intx_error if error_message is None else f"{error_message}; {intx_error}"
            )
        discovery_data["intx_derivatives"] = {
            "enabled": intx_enabled,
            "accessible": intx_accessible,
            "error": intx_error,
        }

    # Determine if reduce-only mode should be required
    reduce_only_required = False
    if fail_on_inaccessible:
        requested_but_inaccessible = False
        if requested_market == "US" and not cfm_accessible:
            requested_but_inaccessible = True
        elif requested_market == "INTX" and not intx_accessible:
            requested_but_inaccessible = True
        elif requested_market == "BOTH" and not (cfm_accessible or intx_accessible):
            requested_but_inaccessible = True

        if requested_but_inaccessible:
            reduce_only_required = True
            logger.warning(
                "Derivatives requested but not accessible - reducing to reduce-only mode",
                operation="derivatives_discovery",
                stage="safety_gate",
                requested_market=requested_market,
                cfm_accessible=cfm_accessible,
                intx_accessible=intx_accessible,
            )

    logger.info(
        "Derivatives eligibility discovery complete",
        operation="derivatives_discovery",
        stage="complete",
        us_enabled=us_enabled,
        intx_enabled=intx_enabled,
        cfm_accessible=cfm_accessible,
        intx_accessible=intx_accessible,
        reduce_only_required=reduce_only_required,
    )

    return DerivativesEligibility(
        us_derivatives_enabled=us_enabled,
        intx_derivatives_enabled=intx_enabled,
        cfm_portfolio_accessible=cfm_accessible,
        intx_portfolio_accessible=intx_accessible,
        error_message=error_message,
        reduce_only_required=reduce_only_required,
        discovery_data=discovery_data,
    )


def _discover_us_derivatives(broker: IBrokerage) -> tuple[bool, bool, str | None]:
    """Discover US derivatives (CFM) eligibility.

    Returns:
        Tuple of (enabled, accessible, error_message)
    """
    try:
        # Check if broker has CFM endpoints
        if not hasattr(broker, "get_cfm_balance_summary"):
            logger.debug(
                "Broker does not support CFM endpoints",
                operation="derivatives_discovery",
                stage="us_check",
            )
            return False, False, "Broker does not support CFM endpoints"

        # Try to fetch CFM balance summary
        balance_summary = broker.get_cfm_balance_summary()  # type: ignore[attr-defined]

        if not balance_summary or not isinstance(balance_summary, dict):
            logger.debug(
                "CFM balance summary returned empty or invalid response",
                operation="derivatives_discovery",
                stage="us_check",
            )
            return True, False, "CFM balance summary unavailable"

        # CFM is accessible
        logger.info(
            "US derivatives (CFM) portfolio accessible",
            operation="derivatives_discovery",
            stage="us_check",
            balance_summary_keys=list(balance_summary.keys()) if balance_summary else [],
        )
        return True, True, None

    except AttributeError as exc:
        logger.debug(
            "Broker missing CFM methods",
            operation="derivatives_discovery",
            stage="us_check",
            error=str(exc),
        )
        return False, False, f"CFM methods not available: {exc}"
    except Exception as exc:
        logger.warning(
            "Failed to discover US derivatives eligibility",
            operation="derivatives_discovery",
            stage="us_check",
            error=str(exc),
            exc_info=True,
        )
        return True, False, f"CFM discovery error: {exc}"


def _discover_intx_derivatives(broker: IBrokerage) -> tuple[bool, bool, str | None]:
    """Discover INTX (International) derivatives eligibility.

    Returns:
        Tuple of (enabled, accessible, error_message)
    """
    try:
        # Check if broker has INTX endpoints
        if not hasattr(broker, "list_portfolios"):
            logger.debug(
                "Broker does not support portfolio listing",
                operation="derivatives_discovery",
                stage="intx_check",
            )
            return False, False, "Broker does not support portfolio listing"

        # Try to list portfolios to find INTX portfolio
        portfolios = broker.list_portfolios()  # type: ignore[attr-defined]

        if not portfolios or not isinstance(portfolios, list):
            logger.debug(
                "No portfolios found or invalid response",
                operation="derivatives_discovery",
                stage="intx_check",
            )
            return True, False, "No portfolios accessible"

        # Look for INTX/perpetuals portfolio
        intx_portfolio = None
        for portfolio in portfolios:
            if not isinstance(portfolio, dict):
                continue
            portfolio_type = str(portfolio.get("type", "")).lower()
            portfolio_name = str(portfolio.get("name", "")).lower()

            if "perpetuals" in portfolio_type or "intx" in portfolio_type:
                intx_portfolio = portfolio
                break
            if "perpetuals" in portfolio_name or "intx" in portfolio_name:
                intx_portfolio = portfolio
                break

        if not intx_portfolio:
            logger.debug(
                "No INTX/perpetuals portfolio found",
                operation="derivatives_discovery",
                stage="intx_check",
                portfolios_count=len(portfolios),
            )
            return True, False, "No INTX portfolio found"

        # Try to access INTX portfolio details
        portfolio_uuid = intx_portfolio.get("uuid")
        if not portfolio_uuid:
            logger.debug(
                "INTX portfolio missing UUID",
                operation="derivatives_discovery",
                stage="intx_check",
            )
            return True, False, "INTX portfolio missing UUID"

        if not hasattr(broker, "get_intx_portfolio"):
            logger.debug(
                "Broker does not support INTX portfolio endpoints",
                operation="derivatives_discovery",
                stage="intx_check",
            )
            return True, False, "INTX portfolio endpoints not available"

        portfolio_details = broker.get_intx_portfolio(portfolio_uuid)  # type: ignore[attr-defined]

        if not portfolio_details or not isinstance(portfolio_details, dict):
            logger.debug(
                "INTX portfolio details unavailable",
                operation="derivatives_discovery",
                stage="intx_check",
            )
            return True, False, "INTX portfolio details unavailable"

        # INTX is accessible
        logger.info(
            "INTX derivatives portfolio accessible",
            operation="derivatives_discovery",
            stage="intx_check",
            portfolio_uuid=portfolio_uuid,
            portfolio_details_keys=list(portfolio_details.keys()) if portfolio_details else [],
        )
        return True, True, None

    except AttributeError as exc:
        logger.debug(
            "Broker missing INTX methods",
            operation="derivatives_discovery",
            stage="intx_check",
            error=str(exc),
        )
        return False, False, f"INTX methods not available: {exc}"
    except Exception as exc:
        logger.warning(
            "Failed to discover INTX derivatives eligibility",
            operation="derivatives_discovery",
            stage="intx_check",
            error=str(exc),
            exc_info=True,
        )
        return True, False, f"INTX discovery error: {exc}"


__all__ = [
    "DerivativesEligibility",
    "DerivativesMarket",
    "discover_derivatives_eligibility",
]
