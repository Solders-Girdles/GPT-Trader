"""Exposure and correlation validation helpers."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping

from ..exceptions import ValidationError
from ..utils import to_decimal


def validate_position_size_limit(config: Any, symbol: str, quantity: Decimal) -> None:
    """Ensure order quantity does not exceed configured position size limits."""
    max_size = getattr(config, "max_position_size", None)
    if max_size is None:
        return
    try:
        limit_decimal = Decimal(str(max_size))
    except Exception:
        return
    if limit_decimal <= 0:
        return
    if abs(quantity) > limit_decimal:
        raise ValidationError(
            f"Position size {abs(quantity)} exceeds limit {limit_decimal} for {symbol}"
        )


def validate_symbol_and_total_exposure(
    *,
    config: Any,
    symbol: str,
    notional: Decimal,
    equity: Decimal,
    current_positions: Mapping[str, Any],
) -> None:
    """Validate per-symbol and total exposure limits."""
    existing_symbol_notional = Decimal("0")
    symbol_payload = current_positions.get(symbol)
    if isinstance(symbol_payload, Mapping):
        existing_symbol_notional = _extract_notional(symbol_payload)

    combined_symbol_notional = abs(notional) + existing_symbol_notional

    max_notional_cap = getattr(config, "max_notional_per_symbol", {}).get(symbol)
    if max_notional_cap is not None and combined_symbol_notional > max_notional_cap:
        raise ValidationError(
            f"Symbol notional {combined_symbol_notional} exceeds cap {max_notional_cap} for {symbol}"
        )

    symbol_exposure_pct = (
        combined_symbol_notional / equity if equity and equity > 0 else Decimal("Infinity")
    )
    symbol_exposure_cap = Decimal(str(config.max_position_pct_per_symbol))

    if symbol_exposure_pct > symbol_exposure_cap:
        raise ValidationError(
            f"Symbol exposure {float(symbol_exposure_pct):.1%} exceeds cap of "
            f"{float(symbol_exposure_cap):.1%} for {symbol}"
        )

    total_exposure = combined_symbol_notional
    for pos_symbol, pos_data in current_positions.items():
        if pos_symbol == symbol or not isinstance(pos_data, Mapping):
            continue
        total_exposure += _extract_notional(pos_data)

    total_exposure_pct = total_exposure / equity if equity and equity > 0 else Decimal("Infinity")
    total_exposure_cap = Decimal(str(config.max_exposure_pct))

    if total_exposure_pct > total_exposure_cap:
        raise ValidationError(
            f"Total exposure {float(total_exposure_pct):.1%} would exceed cap of "
            f"{float(total_exposure_cap):.1%} (new notional: {notional})"
        )


def validate_correlation_risk(
    *,
    config: Any,
    symbol: str,
    notional: Decimal,
    current_positions: Mapping[str, Any],
) -> None:
    """Evaluate concentration of correlated exposures for the portfolio."""
    if not current_positions:
        return

    correlated_notional = Decimal("0")
    for payload in current_positions.values():
        if not isinstance(payload, Mapping):
            continue
        side = str(payload.get("side", "")).lower()
        if side == "short":
            continue
        correlated_notional += abs(_extract_notional(payload))

    correlated_notional += abs(notional)

    cap_value = getattr(config, "max_correlation_notional", None)
    if cap_value is None:
        cap_value = getattr(config, "max_correlated_notional", Decimal("50000"))
    try:
        cap_decimal = Decimal(str(cap_value))
    except Exception:
        cap_decimal = Decimal("50000")

    if cap_decimal <= 0:
        return

    if correlated_notional > cap_decimal:
        raise ValidationError(
            f"correlation risk breach: combined notional {correlated_notional} exceeds "
            f"limit {cap_decimal} for {symbol}"
        )


def _extract_notional(position_payload: Mapping[str, Any]) -> Decimal:
    if "notional" in position_payload:
        return abs(to_decimal(position_payload.get("notional")))

    qty_value = to_decimal(position_payload.get("quantity", position_payload.get("qty")))
    price_value = to_decimal(
        position_payload.get(
            "mark_price", position_payload.get("mark", position_payload.get("price"))
        )
    )
    return abs(qty_value * price_value)


__all__ = [
    "validate_position_size_limit",
    "validate_symbol_and_total_exposure",
    "validate_correlation_risk",
]
