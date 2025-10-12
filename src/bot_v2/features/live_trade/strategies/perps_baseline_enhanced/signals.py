"""Signal helpers for the enhanced baseline strategy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from bot_v2.features.live_trade.strategies.shared import MASnapshot, calculate_ma_snapshot

from .config import StrategyConfig


@dataclass(frozen=True)
class StrategySignal:
    """Computed crossover signal with optional RSI confirmation."""

    label: str
    snapshot: MASnapshot
    rsi: Decimal | None

    @property
    def is_bullish(self) -> bool:
        return self.label == "bullish"

    @property
    def is_bearish(self) -> bool:
        return self.label == "bearish"


def build_signal(
    *,
    marks: Sequence[Decimal],
    config: StrategyConfig,
) -> StrategySignal:
    """Construct the MA snapshot and derive the canonical signal label."""
    decimal_marks = [Decimal(str(value)) for value in marks]

    snapshot = calculate_ma_snapshot(
        decimal_marks,
        short_period=config.short_ma_period,
        long_period=config.long_ma_period,
        epsilon_bps=config.ma_cross_epsilon_bps,
        confirm_bars=config.ma_cross_confirm_bars,
    )

    if snapshot.bullish_cross:
        label = "bullish"
    elif snapshot.bearish_cross:
        label = "bearish"
    else:
        label = "neutral"

    rsi: Decimal | None = None
    filters = config.filters_config
    if filters and filters.require_rsi_confirmation:
        rsi = _calculate_rsi(decimal_marks, filters.rsi_period)

    return StrategySignal(label=label, snapshot=snapshot, rsi=rsi)


def _calculate_rsi(marks: Sequence[Decimal], period: int) -> Decimal | None:
    """Return RSI for the provided marks, or None if insufficient history."""
    try:
        period = int(period)
    except Exception:
        period = 14

    if period <= 1 or len(marks) < period + 1:
        return None

    changes = [marks[i] - marks[i - 1] for i in range(1, len(marks))]
    gains = [max(change, Decimal("0")) for change in changes]
    losses = [abs(min(change, Decimal("0"))) for change in changes]

    period_decimal = Decimal(period)
    avg_gain = sum(gains[:period], Decimal("0")) / period_decimal
    avg_loss = sum(losses[:period], Decimal("0")) / period_decimal

    for idx in range(period, len(changes)):
        avg_gain = (avg_gain * (period_decimal - 1) + gains[idx]) / period_decimal
        avg_loss = (avg_loss * (period_decimal - 1) + losses[idx]) / period_decimal

    if avg_loss == 0:
        return Decimal("100")

    rs = avg_gain / avg_loss
    return Decimal("100") - (Decimal("100") / (Decimal("1") + rs))


__all__ = ["StrategySignal", "build_signal"]
