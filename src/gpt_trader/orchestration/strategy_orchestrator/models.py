"""Datamodels for strategy orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from gpt_trader.features.brokerages.core.interfaces import Balance, Position, Product


@dataclass(slots=True)
class SymbolProcessingContext:
    symbol: str
    balances: Sequence[Balance]
    equity: Decimal
    positions: dict[str, Position]
    position_state: dict[str, Any] | None
    position_quantity: Decimal
    marks: list[Decimal]
    product: Product | None


__all__ = ["SymbolProcessingContext"]
