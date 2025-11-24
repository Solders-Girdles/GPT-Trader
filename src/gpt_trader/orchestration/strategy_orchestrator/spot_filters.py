"""Spot-market specific filters for strategy decisions."""

from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from gpt_trader.features.analyze.indicators import calculate_adx
from gpt_trader.features.live_trade.indicators import mean_decimal as _mean_decimal
from gpt_trader.features.live_trade.indicators import (
    relative_strength_index as _rsi_from_closes,
)
from gpt_trader.features.live_trade.indicators import to_decimal as _to_decimal
from gpt_trader.features.live_trade.indicators import true_range as _true_range
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision
from gpt_trader.utilities.quantities import quantity_from

from .logging_utils import logger
from .models import SymbolProcessingContext


class SpotFiltersMixin:
    """Apply spot-specific filters to strategy decisions."""

    async def _apply_spot_filters(
        self, context: SymbolProcessingContext, decision: Decision
    ) -> Decision:
        rules = self._spot_profiles.get(context.symbol)
        if not rules or decision.action != Action.BUY:
            return decision
        if context.position_state:
            position_qty_raw = quantity_from(context.position_state, default=Decimal("0"))
            position_qty = (
                position_qty_raw if isinstance(position_qty_raw, Decimal) else Decimal("0")
            )
            if position_qty != Decimal("0"):
                return decision

        needs_data = False
        max_window = 0

        vol_config = rules.get("volatility_filter") if isinstance(rules, dict) else None
        if isinstance(vol_config, dict):
            window = int(vol_config.get("window", 0))
            if window > 0:
                needs_data = True
                max_window = max(max_window, window)

        volma_config = rules.get("volume_filter") if isinstance(rules, dict) else None
        if isinstance(volma_config, dict):
            window = int(volma_config.get("window", 0))
            if window > 0:
                needs_data = True
                max_window = max(max_window, window)

        rsi_config = rules.get("momentum_filter") if isinstance(rules, dict) else None
        if isinstance(rsi_config, dict):
            window = int(rsi_config.get("window", 0))
            if window > 0:
                needs_data = True
                max_window = max(max_window, window)

        trend_config = rules.get("trend_filter") if isinstance(rules, dict) else None
        if isinstance(trend_config, dict):
            window = int(trend_config.get("window", 0))
            if window > 0:
                needs_data = True
                max_window = max(max_window, window)

        regime_config = rules.get("regime_filter") if isinstance(rules, dict) else None
        if isinstance(regime_config, dict):
            window = int(regime_config.get("window", 14))
            needs_data = True
            max_window = max(max_window, window + 15)

        closes: list[Decimal] = []
        highs: list[Decimal] = []
        lows: list[Decimal] = []
        volumes: list[Decimal] = []

        if needs_data:
            candles = await self._fetch_spot_candles(context.symbol, max_window)
            if not candles:
                return Decision(action=Action.HOLD, reason="spot_filters_wait")
            for candle in candles:
                closes.append(_to_decimal(getattr(candle, "close", getattr(candle, "price", 0))))
                highs.append(_to_decimal(getattr(candle, "high", getattr(candle, "price", 0))))
                lows.append(_to_decimal(getattr(candle, "low", getattr(candle, "price", 0))))
                volumes.append(_to_decimal(getattr(candle, "volume", getattr(candle, "size", 0))))

        if isinstance(volma_config, dict):
            window = int(volma_config.get("window", 0))
            min_volume = _to_decimal(volma_config.get("min_volume", 0))
            if window > 0:
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

        if isinstance(rsi_config, dict):
            window = int(rsi_config.get("window", 0))
            threshold = _to_decimal(rsi_config.get("threshold", 60))
            if window > 0:
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

        if isinstance(trend_config, dict):
            window = int(trend_config.get("window", 0))
            min_slope = _to_decimal(trend_config.get("min_slope", 0))
            if window > 0:
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

        if isinstance(vol_config := rules.get("volatility_filter"), dict):
            window = int(vol_config.get("window", 0))
            min_vol = _to_decimal(vol_config.get("min_vol", 0))
            max_vol = _to_decimal(vol_config.get("max_vol", 1))
            if window > 0:
                if len(closes) < window + 1:
                    return Decision(action=Action.HOLD, reason="volatility_filter_wait")
                atr_values: list[Decimal] = []
                prev_close: Decimal | None = None
                start_idx = max(len(closes) - window - 1, 0)
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

        if isinstance(regime_config, dict):
            window = int(regime_config.get("window", 14))
            adx_threshold = _to_decimal(regime_config.get("adx_threshold", 25))
            if window > 0:
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

        return decision

    async def _fetch_spot_candles(self, symbol: str, window: int) -> list[Any]:
        bot = self._bot
        limit = max(window + 2, 10)
        try:
            candles = await asyncio.to_thread(
                bot.broker.get_candles,
                symbol,
                "ONE_HOUR",
                limit,
            )
        except Exception as exc:
            logger.debug("Failed to fetch candles for %s: %s", symbol, exc, exc_info=True)
            return []
        if not candles:
            return []
        return sorted(
            candles,
            key=lambda c: getattr(c, "ts", getattr(c, "timestamp", datetime.utcnow())),
        )


__all__ = ["SpotFiltersMixin"]
