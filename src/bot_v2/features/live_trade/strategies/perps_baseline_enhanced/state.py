"""State container for the enhanced baseline strategy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from bot_v2.features.live_trade.strategies.shared import (
    update_mark_window as _update_mark_window,
)
from bot_v2.features.live_trade.strategies.shared import (
    clear_trailing_stop_state,
    update_trailing_stop as _update_trailing_stop,
)


def _default_rejection_counts() -> dict[str, int]:
    return {
        "filter_spread": 0,
        "filter_depth": 0,
        "filter_volume": 0,
        "filter_rsi": 0,
        "guard_liquidation": 0,
        "guard_slippage": 0,
        "stale_data": 0,
        "entries_accepted": 0,
    }


@dataclass
class StrategyState:
    """Tracks rolling marks, trailing stops, and guard counters."""

    mark_windows: dict[str, list[Decimal]] = field(default_factory=dict)
    position_adds: dict[str, int] = field(default_factory=dict)
    trailing_stops: dict[str, tuple[Decimal, Decimal]] = field(default_factory=dict)
    rejection_counts: dict[str, int] = field(default_factory=_default_rejection_counts)

    def reset(self, symbol: str | None = None) -> None:
        """Reset state for a specific symbol or the entire strategy."""
        if symbol is not None:
            self.mark_windows.pop(symbol, None)
            self.position_adds.pop(symbol, None)
            if self.trailing_stops.pop(symbol, None) is not None:
                clear_trailing_stop_state(symbol)
            return

        self.mark_windows.clear()
        self.position_adds.clear()
        self.trailing_stops.clear()
        clear_trailing_stop_state()
        self.rejection_counts.clear()
        self.rejection_counts.update(_default_rejection_counts())

    def update_mark_window(
        self,
        *,
        symbol: str,
        current_mark: Decimal,
        short_period: int,
        long_period: int,
        recent_marks: Sequence[Decimal] | None = None,
        buffer: int = 5,
    ) -> list[Decimal]:
        """Maintain the rolling mark window for MA/RSI calculations."""
        return _update_mark_window(
            self.mark_windows,
            symbol=symbol,
            current_mark=current_mark,
            short_period=short_period,
            long_period=long_period,
            recent_marks=recent_marks,
            buffer=buffer,
        )

    def update_trailing_stop(
        self,
        *,
        symbol: str,
        side: str,
        current_price: Decimal,
        trailing_pct: Decimal,
    ) -> bool:
        """Update trailing stop tracking and return True if it fired."""
        return _update_trailing_stop(
            self.trailing_stops,
            symbol=symbol,
            side=side,
            current_price=current_price,
            trailing_pct=trailing_pct,
        )

    def record_rejection(self, rejection_type: str) -> None:
        """Increment rejection counters by type."""
        self.rejection_counts[rejection_type] = self.rejection_counts.get(rejection_type, 0) + 1

    def record_acceptance(self) -> None:
        """Increment acceptance counter."""
        self.rejection_counts["entries_accepted"] = (
            self.rejection_counts.get("entries_accepted", 0) + 1
        )

    def get_metrics(self) -> dict[str, int | float | dict[str, int]]:
        """Return aggregate metrics for telemetry."""
        total_rejections = sum(
            count for key, count in self.rejection_counts.items() if key != "entries_accepted"
        )
        entries = self.rejection_counts.get("entries_accepted", 0)
        acceptance_rate = (
            entries / (total_rejections + entries) if (total_rejections + entries) else 0
        )

        return {
            "rejection_counts": dict(self.rejection_counts),
            "total_rejections": total_rejections,
            "entries_accepted": entries,
            "acceptance_rate": acceptance_rate,
        }


__all__ = ["StrategyState"]
