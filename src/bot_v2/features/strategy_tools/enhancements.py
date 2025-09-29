"""Strategy enhancement helpers (RSI confirmation, volatility adjustments)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from collections.abc import Iterable


@dataclass
class StrategyEnhancements:
    """Enhanced strategy logic with adaptive parameters."""

    rsi_period: int = 14
    rsi_confirmation_enabled: bool = True
    volatility_lookback: int = 20
    volatility_scaling_enabled: bool = False
    min_volatility_percentile: Decimal = Decimal("25")

    def calculate_rsi(self, prices: list[Decimal], period: int | None = None) -> Decimal | None:
        period = period or self.rsi_period
        if len(prices) < period + 1:
            return None

        changes: list[Decimal] = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains: list[Decimal] = [max(change, Decimal("0")) for change in changes]
        losses: list[Decimal] = [abs(min(change, Decimal("0"))) for change in changes]

        avg_gain = sum(gains[:period], start=Decimal("0")) / Decimal(period)
        avg_loss = sum(losses[:period], start=Decimal("0")) / Decimal(period)

        for i in range(period, len(changes)):
            avg_gain = (avg_gain * Decimal(period - 1) + gains[i]) / Decimal(period)
            avg_loss = (avg_loss * Decimal(period - 1) + losses[i]) / Decimal(period)

        if avg_loss == 0:
            return Decimal("100")

        rs = avg_gain / avg_loss
        return Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

    def should_confirm_ma_crossover(
        self,
        ma_signal: str,
        prices: list[Decimal],
        rsi: Decimal | None = None,
    ) -> tuple[bool, str]:
        if not self.rsi_confirmation_enabled:
            return True, "RSI confirmation disabled"

        if rsi is None:
            rsi = self.calculate_rsi(prices)

        if rsi is None:
            return False, "Insufficient price data for RSI calculation"

        if ma_signal == "buy":
            if rsi > Decimal("70"):
                return False, f"RSI too high for buy: {rsi} > 70"
            return True, f"RSI confirms buy signal: {rsi}"

        if ma_signal == "sell":
            if rsi < Decimal("30"):
                return False, f"RSI too low for sell: {rsi} < 30"
            return True, f"RSI confirms sell signal: {rsi}"

        return False, f"Unknown MA signal: {ma_signal}"
