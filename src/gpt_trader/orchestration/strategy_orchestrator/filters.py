"""
Spot market filters for strategy decisions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Protocol

import pandas as pd

from gpt_trader.features.analyze.indicators import calculate_adx
from gpt_trader.features.live_trade.indicators import mean_decimal as _mean_decimal
from gpt_trader.features.live_trade.indicators import (
    relative_strength_index as _rsi_from_closes,
)
from gpt_trader.features.live_trade.indicators import to_decimal as _to_decimal
from gpt_trader.features.live_trade.indicators import true_range as _true_range
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision
from gpt_trader.orchestration.strategy_orchestrator.logging_utils import logger
from gpt_trader.orchestration.strategy_orchestrator.models import SymbolProcessingContext


class Filter(ABC):
    """Abstract base class for spot filters."""

    @abstractmethod
    def check(
        self,
        context: SymbolProcessingContext,
        config: dict[str, Any],
        candles: list[Any],
    ) -> Decision | None:
        """
        Check if the filter blocks the decision.

        Args:
            context: Symbol context
            config: Filter configuration
            candles: List of candle data

        Returns:
            Decision(HOLD) if blocked, None if passed.
        """
        pass

    @property
    @abstractmethod
    def config_key(self) -> str:
        """Key in the rules dictionary for this filter's config."""
        pass

    def get_window(self, config: dict[str, Any]) -> int:
        """Get the required window size from config."""
        return int(config.get("window", 0))


class VolumeFilter(Filter):
    @property
    def config_key(self) -> str:
        return "volume_filter"

    def check(
        self,
        context: SymbolProcessingContext,
        config: dict[str, Any],
        candles: list[Any],
    ) -> Decision | None:
        window = self.get_window(config)
        min_volume = _to_decimal(config.get("min_volume", 0))

        volumes = [
            _to_decimal(getattr(c, "volume", getattr(c, "size", 0))) for c in candles
        ]

        if len(volumes) < window:
            return Decision(action=Action.HOLD, reason="volume_filter_wait")

        avg_volume = _mean_decimal(volumes[-window:])
        if avg_volume < min_volume:
            logger.info(
                "%s entry blocked by volume filter (avg=%.6f < %.6f)",
                context.symbol,
                float(avg_volume),
                float(min_volume),
            )
            return Decision(action=Action.HOLD, reason="volume_filter_blocked")
        return None


class MomentumFilter(Filter):
    @property
    def config_key(self) -> str:
        return "momentum_filter"

    def check(
        self,
        context: SymbolProcessingContext,
        config: dict[str, Any],
        candles: list[Any],
    ) -> Decision | None:
        window = self.get_window(config)
        threshold = _to_decimal(config.get("threshold", 60))
        
        closes = [
            _to_decimal(getattr(c, "close", getattr(c, "price", 0))) for c in candles
        ]

        if len(closes) < window + 1:
            return Decision(action=Action.HOLD, reason="momentum_filter_wait")

        rsi = _rsi_from_closes(closes[-(window + 1) :], window)
        latest_rsi = rsi[-1]
        if latest_rsi < threshold:
            logger.info(
                "%s entry blocked by momentum filter (RSI=%.2f < %.2f)",
                context.symbol,
                float(latest_rsi),
                float(threshold),
            )
            return Decision(action=Action.HOLD, reason="momentum_filter_blocked")
        return None


class TrendFilter(Filter):
    @property
    def config_key(self) -> str:
        return "trend_filter"

    def check(
        self,
        context: SymbolProcessingContext,
        config: dict[str, Any],
        candles: list[Any],
    ) -> Decision | None:
        window = self.get_window(config)
        min_slope = _to_decimal(config.get("min_slope", 0))
        
        closes = [
            _to_decimal(getattr(c, "close", getattr(c, "price", 0))) for c in candles
        ]

        if len(closes) < window + 1:
            return Decision(action=Action.HOLD, reason="trend_filter_wait")

        current_ma = _mean_decimal(closes[-window:])
        prev_ma = _mean_decimal(closes[-(window + 1) : -1])
        slope = (current_ma - prev_ma) / Decimal(window)

        if slope < min_slope:
            logger.info(
                "%s entry blocked by trend filter (slope=%.6f)",
                context.symbol,
                float(slope),
            )
            return Decision(action=Action.HOLD, reason="trend_filter_blocked")
        return None


class VolatilityFilter(Filter):
    @property
    def config_key(self) -> str:
        return "volatility_filter"

    def check(
        self,
        context: SymbolProcessingContext,
        config: dict[str, Any],
        candles: list[Any],
    ) -> Decision | None:
        window = self.get_window(config)
        min_vol = _to_decimal(config.get("min_vol", 0))
        max_vol = _to_decimal(config.get("max_vol", 1))
        
        closes = [
            _to_decimal(getattr(c, "close", getattr(c, "price", 0))) for c in candles
        ]
        highs = [
            _to_decimal(getattr(c, "high", getattr(c, "price", 0))) for c in candles
        ]
        lows = [
            _to_decimal(getattr(c, "low", getattr(c, "price", 0))) for c in candles
        ]

        if len(closes) < window + 1:
            return Decision(action=Action.HOLD, reason="volatility_filter_wait")

        atr_values: list[Decimal] = []
        prev_close: Decimal | None = None
        start_idx = max(len(closes) - window - 1, 0)
        
        # Reconstruct ATR calculation logic from original
        # Note: Original logic iterated from start_idx to end
        for idx in range(start_idx, len(closes)):
            if prev_close is None and idx > 0:
                prev_close = closes[idx - 1]
            tr = _true_range(highs[idx], lows[idx], prev_close)
            atr_values.append(tr)
            prev_close = closes[idx]
            
        atr = _mean_decimal(atr_values[-window:])
        if atr <= Decimal("0"):
            return Decision(action=Action.HOLD, reason="volatility_filter_blocked")

        vol_pct = atr / closes[-1]
        if vol_pct < min_vol or vol_pct > max_vol:
            logger.info(
                "%s entry blocked by volatility filter (%.6f)",
                context.symbol,
                float(vol_pct),
            )
            return Decision(action=Action.HOLD, reason="volatility_filter_blocked")
        return None


class RegimeFilter(Filter):
    @property
    def config_key(self) -> str:
        return "regime_filter"

    def get_window(self, config: dict[str, Any]) -> int:
        return int(config.get("window", 14))

    def check(
        self,
        context: SymbolProcessingContext,
        config: dict[str, Any],
        candles: list[Any],
    ) -> Decision | None:
        window = self.get_window(config)
        adx_threshold = _to_decimal(config.get("adx_threshold", 25))
        
        closes = [
            _to_decimal(getattr(c, "close", getattr(c, "price", 0))) for c in candles
        ]
        highs = [
            _to_decimal(getattr(c, "high", getattr(c, "price", 0))) for c in candles
        ]
        lows = [
            _to_decimal(getattr(c, "low", getattr(c, "price", 0))) for c in candles
        ]

        if len(closes) < window + 15:
            return Decision(action=Action.HOLD, reason="regime_filter_wait")

        high_series = pd.Series([float(h) for h in highs])
        low_series = pd.Series([float(low_value) for low_value in lows])
        close_series = pd.Series([float(c) for c in closes])

        adx, plus_di, minus_di = calculate_adx(
            high_series, low_series, close_series, window
        )
        current_adx = Decimal(str(adx.iloc[-1]))

        regime = "trending" if current_adx >= adx_threshold else "choppy"
        logger.info(
            "%s regime=%s ADX=%.2f threshold=%.2f +DI=%.2f -DI=%.2f",
            context.symbol,
            regime,
            float(current_adx),
            float(adx_threshold),
            float(plus_di.iloc[-1]),
            float(minus_di.iloc[-1]),
            operation="regime_filter",
            symbol=context.symbol,
        )
        if current_adx < adx_threshold:
            logger.info(
                "%s entry blocked by regime filter (ADX=%.2f < %.2f)",
                context.symbol,
                float(current_adx),
                float(adx_threshold),
            )
            return Decision(
                action=Action.HOLD,
                reason=f"regime_filter_blocked_choppy_adx_{float(current_adx):.1f}",
            )
        return None
