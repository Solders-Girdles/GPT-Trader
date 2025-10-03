"""
Signal calculation for trading strategies.

Extracts MA crossover detection, RSI calculation, and technical signal
computation from strategy classes for focused testing and reusability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class SignalSnapshot:
    """Summary of technical signals used for trading decisions."""

    short_ma: Decimal
    long_ma: Decimal
    epsilon: Decimal
    bullish_cross: bool
    bearish_cross: bool
    rsi: Decimal | None

    @property
    def ma_diff(self) -> Decimal:
        """Difference between short and long MAs."""
        return self.short_ma - self.long_ma


class StrategySignalCalculator:
    """
    Calculates technical signals for strategy decisions.

    Responsibilities:
    - Moving average calculation (short/long periods)
    - MA crossover detection (bullish/bearish)
    - Epsilon tolerance for crossover robustness
    - Crossover confirmation over multiple bars
    - RSI calculation via enhancements module
    """

    def __init__(
        self,
        short_ma_period: int,
        long_ma_period: int,
        ma_cross_epsilon_bps: Decimal,
        ma_cross_confirm_bars: int = 0,
        rsi_calculator: any = None,  # Optional StrategyEnhancements for RSI
    ) -> None:
        """
        Initialize signal calculator.

        Args:
            short_ma_period: Period for short moving average
            long_ma_period: Period for long moving average
            ma_cross_epsilon_bps: Tolerance in basis points for crossover detection
            ma_cross_confirm_bars: Bars to confirm crossover persistence (0 = no confirmation)
            rsi_calculator: Optional RSI calculator (e.g., StrategyEnhancements instance)
        """
        self.short_ma_period = short_ma_period
        self.long_ma_period = long_ma_period
        self.ma_cross_epsilon_bps = ma_cross_epsilon_bps
        self.ma_cross_confirm_bars = ma_cross_confirm_bars
        self.rsi_calculator = rsi_calculator

    def calculate_signals(
        self, marks: list[Decimal], current_mark: Decimal, require_rsi: bool = False
    ) -> SignalSnapshot:
        """
        Compute moving averages, crossovers, and supporting indicators.

        Args:
            marks: Historical price data (must include current_mark appended)
            current_mark: Current price
            require_rsi: Whether to calculate RSI

        Returns:
            SignalSnapshot with MA values, crossover flags, and optional RSI
        """
        # Calculate MAs
        short_ma = self._calculate_ma(marks, self.short_ma_period)
        long_ma = self._calculate_ma(marks, self.long_ma_period)

        # Calculate epsilon tolerance
        epsilon = self._calculate_epsilon(current_mark)

        # Detect crossovers
        bullish_cross, bearish_cross = self._detect_crossovers(marks, epsilon)

        # Apply confirmation if configured
        if self.ma_cross_confirm_bars > 0 and (bullish_cross or bearish_cross):
            bullish_cross, bearish_cross = self._confirm_crossover(
                marks, epsilon, bullish_cross, bearish_cross
            )

        # Calculate RSI if required
        rsi = None
        if require_rsi and self.rsi_calculator:
            rsi_raw = self.rsi_calculator.calculate_rsi(marks)
            rsi = Decimal(str(rsi_raw)) if rsi_raw is not None else None

        return SignalSnapshot(
            short_ma=short_ma,
            long_ma=long_ma,
            epsilon=epsilon,
            bullish_cross=bullish_cross,
            bearish_cross=bearish_cross,
            rsi=rsi,
        )

    def _calculate_ma(self, marks: list[Decimal], period: int) -> Decimal:
        """Calculate simple moving average for given period."""
        if len(marks) < period:
            # Not enough data - return simple average
            return sum(marks, Decimal("0")) / Decimal(len(marks)) if marks else Decimal("0")

        ma_sum = sum(marks[-period:], Decimal("0"))
        return ma_sum / Decimal(period)

    def _calculate_epsilon(self, current_mark: Decimal) -> Decimal:
        """Calculate epsilon tolerance for crossover detection."""
        epsilon_rate = self.ma_cross_epsilon_bps / Decimal("10000")
        return current_mark * epsilon_rate

    def _detect_crossovers(self, marks: list[Decimal], epsilon: Decimal) -> tuple[bool, bool]:
        """
        Detect bullish and bearish MA crossovers.

        Returns:
            (bullish_cross, bearish_cross)
        """
        bullish_cross = False
        bearish_cross = False

        # Need at least long_period + 1 for previous MA calculation
        if len(marks) < self.long_ma_period + 1:
            return bullish_cross, bearish_cross

        # Calculate previous MAs (excluding current mark)
        prev_marks = marks[:-1]
        prev_short = self._calculate_ma(prev_marks, self.short_ma_period)
        prev_long = self._calculate_ma(prev_marks, self.long_ma_period)
        prev_diff = prev_short - prev_long

        # Calculate current MAs
        cur_short = self._calculate_ma(marks, self.short_ma_period)
        cur_long = self._calculate_ma(marks, self.long_ma_period)
        cur_diff = cur_short - cur_long

        # Detect crossovers with epsilon tolerance
        bullish_cross = (prev_diff <= epsilon) and (cur_diff > epsilon)
        bearish_cross = (prev_diff >= -epsilon) and (cur_diff < -epsilon)

        if bullish_cross or bearish_cross:
            cross_type = "Bullish" if bullish_cross else "Bearish"
            logger.debug(
                f"{cross_type} cross detected: prev_diff={prev_diff:.4f}, "
                f"cur_diff={cur_diff:.4f}, eps={epsilon:.4f}, "
                f"confirm_bars={self.ma_cross_confirm_bars}"
            )

        return bullish_cross, bearish_cross

    def _confirm_crossover(
        self,
        marks: list[Decimal],
        epsilon: Decimal,
        bullish_cross: bool,
        bearish_cross: bool,
    ) -> tuple[bool, bool]:
        """
        Confirm crossover persists over multiple bars.

        Returns:
            (confirmed_bullish, confirmed_bearish)
        """
        # Need enough data for confirmation lookback
        min_required = self.long_ma_period + self.ma_cross_confirm_bars + 1
        if len(marks) < min_required:
            return False, False

        confirmed = True
        for i in range(1, self.ma_cross_confirm_bars + 1):
            check_marks = marks[:-i]
            check_short = self._calculate_ma(check_marks, self.short_ma_period)
            check_long = self._calculate_ma(check_marks, self.long_ma_period)
            check_diff = check_short - check_long

            # Check if crossover persists
            if bullish_cross and check_diff <= epsilon:
                confirmed = False
                break
            if bearish_cross and check_diff >= -epsilon:
                confirmed = False
                break

        return (bullish_cross and confirmed, bearish_cross and confirmed)
