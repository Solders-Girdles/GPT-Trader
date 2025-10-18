"""Simple, deterministic strategy implementations for early backtests."""

from __future__ import annotations

from collections.abc import Callable
from statistics import fmean, pstdev

from gpt_trader.domain import Bar, Signal, Strategy

StrategyFactory = Callable[[], Strategy]


class MovingAverageCrossStrategy(Strategy):
    """Naive SMA crossover strategy suitable for scaffolding tests."""

    def __init__(self, short_window: int = 5, long_window: int = 20) -> None:
        if short_window <= 0 or long_window <= 0:
            raise ValueError("Window sizes must be positive.")
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window.")
        self.short_window = short_window
        self.long_window = long_window

    def decide(self, bars: list[Bar]) -> Signal:
        closes = [float(b.close) for b in bars]
        symbol = bars[-1].symbol

        if len(closes) < self.long_window:
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                metadata={
                    "reason": "insufficient_history",
                    "observations": len(closes),
                    "required": self.long_window,
                },
            )

        short_ma = fmean(closes[-self.short_window :])
        long_ma = fmean(closes[-self.long_window :])

        if short_ma > long_ma:
            delta = short_ma - long_ma
            return Signal(
                symbol=symbol,
                action="BUY",
                confidence=float(delta / long_ma) if long_ma else 0.0,
                metadata={
                    "reason": "short_ma_cross_above_long_ma",
                    "short_ma": short_ma,
                    "long_ma": long_ma,
                    "delta": delta,
                },
            )
        if short_ma < long_ma:
            delta = short_ma - long_ma
            return Signal(
                symbol=symbol,
                action="SELL",
                confidence=float(delta / long_ma) if long_ma else 0.0,
                metadata={
                    "reason": "short_ma_cross_below_long_ma",
                    "short_ma": short_ma,
                    "long_ma": long_ma,
                    "delta": delta,
                },
            )
        return Signal(
            symbol=symbol,
            action="HOLD",
            confidence=0.0,
            metadata={
                "reason": "averages_equal",
                "short_ma": short_ma,
                "long_ma": long_ma,
                "delta": 0.0,
            },
        )


def get_strategy(name: str) -> Strategy:
    """Factory for built-in strategies."""
    normalized = name.strip().lower()
    if normalized in {"ma", "ma-cross", "ma-crossover", "moving-average"}:
        return MovingAverageCrossStrategy()
    if normalized in {"donchian", "donchian-breakout"}:
        return DonchianBreakoutStrategy()
    if normalized in {"buy-and-hold", "buyhold"}:
        return _BuyHoldStrategy()
    if normalized.startswith("volatility-scaled"):
        base_name = "ma-crossover"
        if ":" in normalized:
            _, base_name = normalized.split(":", 1)
        elif "(" in normalized and normalized.endswith(")"):
            base_name = normalized[normalized.find("(") + 1 : -1]
        base_strategy = get_strategy(base_name)
        return VolatilityScaledStrategy(base_strategy)
    raise ValueError(f"Unknown strategy: {name}")


class _BuyHoldStrategy(Strategy):
    """Trivial baseline that buys once and never sells."""

    def __init__(self) -> None:
        self._bought = False

    def decide(self, bars: list[Bar]) -> Signal:
        symbol = bars[-1].symbol
        if not self._bought:
            self._bought = True
            return Signal(
                symbol=symbol,
                action="BUY",
                confidence=1.0,
                metadata={"reason": "initial_position"},
            )
        return Signal(
            symbol=symbol,
            action="HOLD",
            confidence=0.0,
            metadata={"reason": "position_held"},
        )


__all__ = ["MovingAverageCrossStrategy", "get_strategy"]


class DonchianBreakoutStrategy(Strategy):
    """Breakout strategy tracking channel highs/lows."""

    def __init__(self, lookback: int = 20) -> None:
        if lookback <= 1:
            raise ValueError("lookback must be > 1")
        self.lookback = lookback

    def decide(self, bars: list[Bar]) -> Signal:
        symbol = bars[-1].symbol
        closes = [float(bar.close) for bar in bars]
        if len(bars) < self.lookback:
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                metadata={
                    "reason": "insufficient_history",
                    "observations": len(bars),
                    "required": self.lookback,
                },
            )

        window = bars[-self.lookback :]
        highs = [float(bar.high) for bar in window]
        lows = [float(bar.low) for bar in window]
        channel_high = max(highs[:-1])  # exclude current bar for signal
        channel_low = min(lows[:-1])
        last_close = closes[-1]

        metadata = {
            "reason": "channel_hold",
            "channel_high": channel_high,
            "channel_low": channel_low,
            "close": last_close,
            "lookback": self.lookback,
        }
        confidence = 0.0
        action: str = "HOLD"

        if last_close > channel_high:
            action = "BUY"
            spread = last_close - channel_high
            confidence = float(spread / channel_high)
            metadata["reason"] = "breakout_above"
        elif last_close < channel_low:
            action = "SELL"
            spread = channel_low - last_close
            confidence = float(spread / channel_low)
            metadata["reason"] = "breakout_below"

        return Signal(
            symbol=symbol, action=action, confidence=min(confidence, 1.0), metadata=metadata
        )


class VolatilityScaledStrategy(Strategy):
    """Wrap another strategy and modulate confidence by realized volatility."""

    def __init__(self, base: Strategy, vol_window: int = 20, target_vol: float = 0.02) -> None:
        self.base = base
        self.vol_window = vol_window
        self.target_vol = target_vol

    def decide(self, bars: list[Bar]) -> Signal:
        base_signal = self.base.decide(bars)
        metadata = {
            "reason": "vol_scaled",
            "base_action": base_signal.action,
            "base_confidence": base_signal.confidence,
            "vol_window": self.vol_window,
            "target_vol": self.target_vol,
        }

        if len(bars) <= 1:
            return Signal(
                symbol=base_signal.symbol,
                action=base_signal.action,
                confidence=0.0,
                metadata={"reason": "insufficient_history"},
            )

        closes = [float(bar.close) for bar in bars[-(self.vol_window + 1) :]]
        returns = []
        for prev, current in zip(closes, closes[1:]):
            if prev == 0:
                continue
            returns.append((current - prev) / prev)

        if not returns:
            realized_vol = 0.0
        else:
            realized_vol = float(pstdev(returns))

        metadata["realized_vol"] = realized_vol

        if realized_vol <= 0:
            scaling = 1.0
        else:
            scaling = min(1.0, self.target_vol / realized_vol)

        metadata["scaling"] = scaling

        return Signal(
            symbol=base_signal.symbol,
            action=base_signal.action,
            confidence=min(1.0, base_signal.confidence * scaling),
            metadata={"volatility_scaled": metadata, "base": base_signal.metadata},
        )
