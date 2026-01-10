"""
Canonical rejection reason normalization for order execution telemetry.
"""

from __future__ import annotations

from typing import Final

STABLE_REJECTION_CODES: Final[set[str]] = {
    "broker_rejected",
    "broker_status",
    "exchange_rules",
    "guard_error",
    "guard_failure",
    "insufficient_funds",
    "invalid_request",
    "invalid_size",
    "invalid_price",
    "mark_staleness",
    "order_preview",
    "order_validation",
    "paused",
    "pre_trade_validation",
    "quantity_zero",
    "rate_limit",
    "reduce_only",
    "security_validation",
    "slippage_guard",
    "timeout",
    "network",
    "market_closed",
    "unknown",
}


def normalize_rejection_reason(reason: str) -> tuple[str, str | None]:
    """Normalize raw rejection reasons into stable codes.

    Returns:
        (code, detail) where code is stable for metrics/logs and detail
        preserves the raw reason for diagnostics.
    """
    raw = reason.strip() if reason else ""
    if not raw:
        return "unknown", None
    lowered = raw.lower()

    if lowered in STABLE_REJECTION_CODES:
        return lowered, None

    if lowered.startswith("paused:"):
        return "paused", raw.partition(":")[2] or None
    if lowered.startswith("guard_error:"):
        return "guard_error", raw.partition(":")[2] or None
    if lowered.startswith("guard_failure:"):
        return "guard_failure", raw.partition(":")[2] or None
    if lowered.startswith("broker_status_"):
        return "broker_status", raw[len("broker_status_") :] or None
    if "order rejected by broker" in lowered:
        detail = raw.partition(":")[2].strip()
        return ("broker_status", detail) if detail else ("broker_rejected", None)

    if lowered in {"quantity_zero"}:
        return "quantity_zero", None
    if lowered in {
        "reduce_only_not_reducing",
        "reduce_only_empty_position",
        "reduce_only_mode_blocked",
    }:
        return "reduce_only", lowered
    if "reduce_only" in lowered or "reduce-only" in lowered:
        return "reduce_only", raw

    if lowered == "security_validation_failed" or "security validation failed" in lowered:
        return "security_validation", raw if lowered != "security_validation_failed" else None
    if lowered == "mark_staleness" or ("mark" in lowered and "stale" in lowered):
        return "mark_staleness", raw if lowered != "mark_staleness" else None
    if "slippage" in lowered:
        return "slippage_guard", raw
    if "preview" in lowered:
        return "order_preview", raw

    size_terms = ["size", "quantity", "amount", "notional"]
    size_qualifiers = [
        "min",
        "max",
        "minimum",
        "maximum",
        "below",
        "above",
        "outside",
        "invalid",
        "too small",
        "too large",
        "less",
        "greater",
        "step",
        "increment",
    ]
    if any(term in lowered for term in size_terms) and any(
        qualifier in lowered for qualifier in size_qualifiers
    ):
        return "invalid_size", raw

    price_terms = ["price", "tick", "increment", "step"]
    price_qualifiers = [
        "tick",
        "increment",
        "step",
        "invalid",
        "price",
        "below",
        "above",
        "outside",
    ]
    if any(term in lowered for term in price_terms) and any(
        qualifier in lowered for qualifier in price_qualifiers
    ):
        return "invalid_price", raw

    if "rejected" in lowered:
        return "broker_rejected", raw
    if any(
        term in lowered
        for term in ["exchange", "rule", "min_", "minimum", "step", "tick", "increment"]
    ):
        return "exchange_rules", raw
    if any(term in lowered for term in ["leverage", "exposure", "liquidation", "mmr", "risk"]):
        return "pre_trade_validation", raw

    if any(term in lowered for term in ["insufficient", "margin", "balance", "funds"]):
        return "insufficient_funds", raw
    if any(term in lowered for term in ["rate_limit", "rate limit", "429", "too many"]):
        return "rate_limit", raw
    if any(term in lowered for term in ["timeout", "timed out", "deadline"]):
        return "timeout", raw
    if any(term in lowered for term in ["network", "connection", "socket", "dns", "ssl"]):
        return "network", raw
    if any(term in lowered for term in ["market closed", "trading halt", "suspended"]):
        return "market_closed", raw

    if "invalid" in lowered or "spec" in lowered:
        return "invalid_request", raw
    return "unknown", raw


__all__ = ["normalize_rejection_reason", "STABLE_REJECTION_CODES"]
