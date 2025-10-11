"""Moving-average signal helpers shared across strategies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class MASnapshot:
    """Moving-average snapshot for the latest bar."""

    short_ma: Decimal
    long_ma: Decimal
    epsilon: Decimal
    bullish_cross: bool
    bearish_cross: bool

    @property
    def diff(self) -> Decimal:
        return self.short_ma - self.long_ma


def calculate_ma_snapshot(
    marks: Sequence[Decimal],
    *,
    short_period: int,
    long_period: int,
    epsilon_bps: Decimal = Decimal("0"),
    confirm_bars: int = 0,
) -> MASnapshot:
    """Return moving-average snapshot and crossover flags for the newest bar."""
    if not marks:
        zero = Decimal("0")
        return MASnapshot(
            short_ma=zero, long_ma=zero, epsilon=zero, bullish_cross=False, bearish_cross=False
        )

    marks = [Decimal(str(value)) for value in marks]
    short_period = max(1, short_period)
    long_period = max(short_period, long_period)

    if len(marks) < short_period:
        ref = marks[-1]
        epsilon = _compute_epsilon(ref, epsilon_bps)
        return MASnapshot(
            short_ma=ref, long_ma=ref, epsilon=epsilon, bullish_cross=False, bearish_cross=False
        )

    short_ma = sum(marks[-short_period:], Decimal("0")) / Decimal(short_period)

    if len(marks) >= long_period:
        long_ma = sum(marks[-long_period:], Decimal("0")) / Decimal(long_period)
    else:
        long_ma = short_ma

    epsilon = _compute_epsilon(marks[-1], epsilon_bps)

    bullish_cross = False
    bearish_cross = False

    if len(marks) >= long_period + 1:
        prev_marks = marks[:-1]
        prev_short = sum(prev_marks[-short_period:], Decimal("0")) / Decimal(short_period)
        prev_long = sum(prev_marks[-long_period:], Decimal("0")) / Decimal(long_period)

        prev_diff = prev_short - prev_long
        cur_diff = short_ma - long_ma

        bullish_cross = (prev_diff <= epsilon) and (cur_diff > epsilon)
        bearish_cross = (prev_diff >= -epsilon) and (cur_diff < -epsilon)

        if confirm_bars > 0 and (bullish_cross or bearish_cross):
            if len(marks) >= long_period + confirm_bars + 1:
                confirmed = True
                for i in range(1, confirm_bars + 1):
                    check_marks = marks[:-i]
                    check_short = sum(check_marks[-short_period:], Decimal("0")) / Decimal(
                        short_period
                    )
                    check_long = sum(check_marks[-long_period:], Decimal("0")) / Decimal(
                        long_period
                    )
                    check_diff = check_short - check_long
                    if bullish_cross and check_diff <= epsilon:
                        confirmed = False
                        break
                    if bearish_cross and check_diff >= -epsilon:
                        confirmed = False
                        break
                bullish_cross = bullish_cross and confirmed
                bearish_cross = bearish_cross and confirmed
            else:
                bullish_cross = False
                bearish_cross = False

    return MASnapshot(
        short_ma=short_ma,
        long_ma=long_ma,
        epsilon=epsilon,
        bullish_cross=bullish_cross,
        bearish_cross=bearish_cross,
    )


def _compute_epsilon(reference_price: Decimal, epsilon_bps: Decimal) -> Decimal:
    try:
        return Decimal(str(reference_price)) * (Decimal(str(epsilon_bps)) / Decimal("10000"))
    except Exception:
        return Decimal("0")


__all__ = ["MASnapshot", "calculate_ma_snapshot"]
