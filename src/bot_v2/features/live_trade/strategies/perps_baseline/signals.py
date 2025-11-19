"""Signal helpers for the baseline perpetuals strategy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from bot_v2.features.live_trade.strategies.shared import MASnapshot, calculate_ma_snapshot

from .config import StrategyConfig


@dataclass(frozen=True)
class StrategySignal:
    """Computed signal and MA snapshot for the latest price."""

    label: str
    snapshot: MASnapshot

    @property
    def is_bullish(self) -> bool:
        return self.label == "bullish"

    @property
    def is_bearish(self) -> bool:
        return self.label == "bearish"


def build_signal(
    *,
    current_mark: Decimal,
    recent_marks: Sequence[Decimal] | None,
    config: StrategyConfig,
) -> StrategySignal:
    """Construct the MA snapshot and derive the canonical signal label."""
    history = [Decimal(str(value)) for value in (recent_marks or [])]
    marks = history + [Decimal(str(current_mark))]

    snapshot = calculate_ma_snapshot(
        marks,
        short_period=config.short_ma_period,
        long_period=config.long_ma_period,
        epsilon_bps=config.ma_cross_epsilon_bps,
        confirm_bars=config.ma_cross_confirm_bars,
    )

    if snapshot.bullish_cross or (config.force_entry_on_trend and snapshot.short_ma > snapshot.long_ma):
        label = "bullish"
    elif snapshot.bearish_cross or (config.force_entry_on_trend and snapshot.short_ma < snapshot.long_ma):
        label = "bearish"
    else:
        label = "neutral"

    return StrategySignal(label=label, snapshot=snapshot)


__all__ = ["StrategySignal", "build_signal"]
