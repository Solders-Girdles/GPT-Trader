"""Compatibility shim re-exporting pre-trade validation package."""

from bot_v2.features.live_trade.risk_calculations import (
    effective_mmr,
    effective_symbol_leverage_cap,
)

from .pre_trade import PreTradeValidator, ValidationError, coalesce_quantity, logger, to_decimal

_coalesce_quantity = coalesce_quantity
_to_decimal = to_decimal

__all__ = [
    "PreTradeValidator",
    "ValidationError",
    "coalesce_quantity",
    "_coalesce_quantity",
    "to_decimal",
    "_to_decimal",
    "effective_mmr",
    "effective_symbol_leverage_cap",
    "logger",
]
