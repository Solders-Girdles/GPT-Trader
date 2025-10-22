"""Margin health monitoring for US derivatives (CFM).

This module implements margin health queries for US futures including:
- Intraday vs overnight margin requirements
- Margin ratio = available_margin / liquidation_threshold
- Cash sweep tracking
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bot_v2.utilities.logging_patterns import get_logger
from bot_v2.utilities.telemetry import emit_metric

if TYPE_CHECKING:
    from bot_v2.features.brokerages.core.interfaces import IBrokerage
    from bot_v2.persistence.event_store import EventStore

logger = get_logger(__name__, component="margin_monitor")


@dataclass(frozen=True)
class MarginHealthSnapshot:
    """Snapshot of margin health for US derivatives."""

    total_cash_available: Decimal
    total_margin_requirement: Decimal
    available_margin: Decimal
    liquidation_threshold: Decimal
    margin_ratio: Decimal  # available_margin / liquidation_threshold
    margin_window: str  # "INTRADAY" or "OVERNIGHT"
    collateral_value: Decimal
    position_notional: Decimal
    timestamp: str
    raw_data: dict[str, Any]


def get_margin_health(
    broker: IBrokerage,
    *,
    event_store: EventStore | None = None,
    bot_id: str = "unknown",
) -> MarginHealthSnapshot | None:
    """Query margin health and log margin ratio.

    Args:
        broker: The brokerage instance to query
        event_store: Optional event store to record metrics
        bot_id: Bot identifier for logging

    Returns:
        MarginHealthSnapshot or None if query fails
    """
    logger.debug(
        "Querying margin health",
        operation="margin_health_query",
        stage="start",
    )

    try:
        # Query CFM balance summary
        if not hasattr(broker, "get_cfm_balance_summary"):
            logger.debug(
                "Broker does not support CFM balance summary",
                operation="margin_health_query",
                stage="skip",
            )
            return None

        balance_summary = broker.get_cfm_balance_summary()  # type: ignore[attr-defined]

        if not balance_summary or not isinstance(balance_summary, dict):
            logger.warning(
                "CFM balance summary returned empty or invalid response",
                operation="margin_health_query",
                stage="empty_response",
            )
            return None

        # Extract margin metrics
        snapshot = _extract_margin_snapshot(balance_summary)

        if snapshot:
            # Log margin ratio
            logger.info(
                "Margin health snapshot",
                operation="margin_health_query",
                stage="complete",
                margin_ratio=float(snapshot.margin_ratio),
                available_margin=float(snapshot.available_margin),
                liquidation_threshold=float(snapshot.liquidation_threshold),
                margin_window=snapshot.margin_window,
            )

            # Emit metric to event store
            if event_store is not None:
                try:
                    emit_metric(
                        event_store,
                        bot_id,
                        {
                            "event_type": "margin_health_snapshot",
                            "margin_ratio": str(snapshot.margin_ratio),
                            "available_margin": str(snapshot.available_margin),
                            "liquidation_threshold": str(snapshot.liquidation_threshold),
                            "margin_window": snapshot.margin_window,
                            "collateral_value": str(snapshot.collateral_value),
                            "position_notional": str(snapshot.position_notional),
                            "timestamp": snapshot.timestamp,
                        },
                        logger=logger,
                    )
                except Exception as exc:
                    logger.debug(
                        "Failed to emit margin health metric",
                        error=str(exc),
                        operation="margin_health_query",
                        stage="emit_metric",
                    )

            # Warn if margin ratio is concerning
            if snapshot.margin_ratio < Decimal("0.2"):  # Less than 20%
                logger.warning(
                    "Low margin ratio detected",
                    operation="margin_health_query",
                    stage="low_margin_warning",
                    margin_ratio=float(snapshot.margin_ratio),
                    available_margin=float(snapshot.available_margin),
                    liquidation_threshold=float(snapshot.liquidation_threshold),
                )

        return snapshot

    except Exception as exc:
        logger.error(
            "Failed to query margin health",
            operation="margin_health_query",
            stage="error",
            error=str(exc),
            exc_info=True,
        )
        return None


def _extract_margin_snapshot(balance_summary: dict[str, Any]) -> MarginHealthSnapshot | None:
    """Extract margin health snapshot from CFM balance summary.

    Args:
        balance_summary: Raw balance summary from CFM endpoint

    Returns:
        MarginHealthSnapshot or None if extraction fails
    """
    try:
        from datetime import UTC, datetime

        # Extract cash and margin metrics
        # CFM balance summary typically includes:
        # - total_cash_available: Available cash for margin
        # - total_margin_requirement: Current margin requirement
        # - liquidation_threshold: Threshold for liquidation
        # - margin_buffer: Buffer before liquidation
        # - collateral_value: Total collateral value

        total_cash = Decimal(str(balance_summary.get("total_cash_available", "0")))
        margin_req = Decimal(str(balance_summary.get("total_margin_requirement", "0")))
        liq_threshold = Decimal(str(balance_summary.get("liquidation_threshold", "0")))
        collateral = Decimal(str(balance_summary.get("collateral_value", "0")))

        # Calculate available margin
        # available_margin = total_cash - margin_requirement
        available_margin = total_cash - margin_req

        # If liquidation threshold is not provided, estimate it
        # Typically liquidation occurs when available margin drops below maintenance requirement
        if liq_threshold == Decimal("0"):
            # Estimate as maintenance margin (typically 50-80% of initial margin requirement)
            # Use margin_buffer if available
            margin_buffer = balance_summary.get("margin_buffer")
            if margin_buffer:
                liq_threshold = Decimal(str(margin_buffer))
            else:
                # Conservative estimate: liquidation at 20% of current margin requirement
                liq_threshold = margin_req * Decimal("0.2")

        # Calculate margin ratio = available_margin / liquidation_threshold
        if liq_threshold > Decimal("0"):
            margin_ratio = available_margin / liq_threshold
        else:
            # If no liquidation threshold, set ratio very high (safe)
            margin_ratio = Decimal("999.0")

        # Extract margin window
        margin_window = str(balance_summary.get("margin_window", "UNKNOWN")).upper()
        if margin_window not in ("INTRADAY", "OVERNIGHT"):
            margin_window = "UNKNOWN"

        # Extract position notional
        position_notional = Decimal(str(balance_summary.get("position_notional", "0")))

        # Timestamp
        timestamp = balance_summary.get("timestamp") or balance_summary.get("as_of")
        if not timestamp:
            timestamp = datetime.now(UTC).isoformat()

        return MarginHealthSnapshot(
            total_cash_available=total_cash,
            total_margin_requirement=margin_req,
            available_margin=available_margin,
            liquidation_threshold=liq_threshold,
            margin_ratio=margin_ratio,
            margin_window=margin_window,
            collateral_value=collateral,
            position_notional=position_notional,
            timestamp=str(timestamp),
            raw_data=balance_summary,
        )

    except Exception as exc:
        logger.warning(
            "Failed to extract margin snapshot",
            operation="margin_health_query",
            stage="extract_snapshot",
            error=str(exc),
        )
        return None


def get_cfm_sweeps(
    broker: IBrokerage,
    *,
    event_store: EventStore | None = None,
    bot_id: str = "unknown",
) -> list[dict[str, Any]]:
    """Query CFM cash sweeps schedule.

    Args:
        broker: The brokerage instance to query
        event_store: Optional event store to record metrics
        bot_id: Bot identifier for logging

    Returns:
        List of sweep records
    """
    try:
        if not hasattr(broker, "list_cfm_sweeps"):
            return []

        sweeps = broker.list_cfm_sweeps()  # type: ignore[attr-defined]

        if sweeps and event_store:
            try:
                emit_metric(
                    event_store,
                    bot_id,
                    {
                        "event_type": "cfm_sweeps_queried",
                        "sweep_count": len(sweeps),
                        "sweeps": sweeps,
                    },
                    logger=logger,
                )
            except Exception as exc:
                logger.debug(
                    "Failed to emit CFM sweeps metric",
                    error=str(exc),
                    operation="cfm_sweeps_query",
                    stage="emit_metric",
                )

        return sweeps

    except Exception as exc:
        logger.error(
            "Failed to query CFM sweeps",
            operation="cfm_sweeps_query",
            stage="error",
            error=str(exc),
            exc_info=True,
        )
        return []


__all__ = [
    "MarginHealthSnapshot",
    "get_cfm_sweeps",
    "get_margin_health",
]
