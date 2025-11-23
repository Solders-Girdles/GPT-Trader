"""Utility helpers and response builders for position sizing."""

from __future__ import annotations

from typing import List

from bot_v2.features.position_sizing.types import (
    PositionSizeRequest,
    PositionSizeResponse,
)
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="position_sizing")


def optional_float(value: float | None, default: float) -> float:
    return float(value) if value is not None else default


def estimate_position_risk(
    request: PositionSizeRequest,
    position_size_pct: float,
) -> float:
    if request.avg_loss is not None:
        return float(position_size_pct * abs(request.avg_loss))
    if request.volatility is not None:
        return float(position_size_pct * request.volatility * 1.5)
    return float(position_size_pct * 0.05)


def estimate_portfolio_risk(
    position_size_pct: float,
    avg_loss: float | None,
    volatility: float | None,
) -> float:
    try:
        if avg_loss is not None:
            risk = position_size_pct * abs(avg_loss)
        elif volatility is not None:
            risk = position_size_pct * volatility * 1.5
        else:
            risk = position_size_pct * 0.05
        return float(max(0.0, min(1.0, risk)))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Risk estimation failed: %s, using conservative default", exc)
        return float(position_size_pct * 0.05)


def create_error_response(
    request: PositionSizeRequest,
    errors: List[str],
) -> PositionSizeResponse:
    return PositionSizeResponse(
        symbol=request.symbol or "UNKNOWN",
        recommended_shares=0,
        recommended_value=0.0,
        position_size_pct=0.0,
        risk_pct=0.0,
        method_used=request.method,
        warnings=errors,
    )


__all__ = [
    "optional_float",
    "estimate_position_risk",
    "estimate_portfolio_risk",
    "create_error_response",
]
