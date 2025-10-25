"""Position access helpers for the live risk manager."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any


class LiveRiskManagerPositionMixin:
    """Expose normalized access to current positions."""

    def get_current_positions(self, *, as_dict: bool = False) -> list[Any] | dict[str, Any]:
        """Return current positions snapshot in requested format."""
        positions = getattr(self.runtime_monitor, "positions", {})
        if isinstance(positions, dict):
            normalized_map: dict[str, Any] = {}
            normalized_list: list[Any] = []
            for symbol, payload in positions.items():
                if hasattr(payload, "symbol"):
                    normalized_map[symbol] = payload
                    normalized_entry = self._coerce_decimal(getattr(payload, "entry_price", None))
                    normalized_quantity = getattr(payload, "quantity", None)
                    size_value = (
                        float(normalized_quantity) if normalized_quantity is not None else 0.0
                    )
                    normalized_list.append(
                        SimpleNamespace(
                            symbol=symbol,
                            size=size_value,
                            entry_price=normalized_entry,
                        )
                    )
                    continue
                if isinstance(payload, dict):
                    normalized_map[symbol] = dict(payload)
                    size_decimal = self._coerce_decimal(payload.get("quantity", payload.get("qty")))
                    entry_decimal = self._coerce_decimal(
                        payload.get("entry_price", payload.get("mark"))
                    )
                    normalized_list.append(
                        SimpleNamespace(
                            symbol=symbol,
                            size=float(size_decimal),
                            entry_price=entry_decimal,
                        )
                    )
                    continue
                normalized_map[symbol] = payload
                normalized_list.append(
                    SimpleNamespace(symbol=symbol, size=None, entry_price=Decimal("0"))
                )
            return normalized_map if as_dict else normalized_list
        if isinstance(positions, list):
            if as_dict:
                normalized_map = {}
                for payload in positions:
                    symbol = getattr(payload, "symbol", None)
                    if symbol is None:
                        continue
                    normalized_map[symbol] = payload
                return normalized_map
            return list(positions)
        return {} if as_dict else []

    @staticmethod
    def _coerce_decimal(value: Any) -> Decimal:
        """Safe conversion helper for numeric integration fields."""
        if value in (None, "", "null"):
            return Decimal("0")
        try:
            return Decimal(str(value))
        except Exception:
            return Decimal("0")

    @staticmethod
    def _normalize_quantity_key(value: Any) -> str:
        """Normalize quantity identifiers for integration bookkeeping."""
        try:
            decimal_value = Decimal(str(value))
        except Exception:
            return "0"
        normalized = format(decimal_value, "f").rstrip("0").rstrip(".")
        return normalized or "0"


__all__ = ["LiveRiskManagerPositionMixin"]
