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
        from .filters import (
            MomentumFilter,
            RegimeFilter,
            TrendFilter,
            VolatilityFilter,
            VolumeFilter,
        )

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

        filters = [
            VolumeFilter(),
            MomentumFilter(),
            TrendFilter(),
            VolatilityFilter(),
            RegimeFilter(),
        ]

        # 1. Calculate required window
        max_window = 0
        active_filters = []
        
        for flt in filters:
            config = rules.get(flt.config_key)
            if isinstance(config, dict):
                window = flt.get_window(config)
                if window > 0 or flt.config_key == "regime_filter": # Regime always needs data if present
                     # Regime filter logic in original code had slightly different window logic (window + 15)
                     # But get_window returns base window.
                     # Let's trust the filter's check method to handle specific data needs if we pass enough.
                     # Actually, we need to know how much to fetch.
                     # The original code calculated max_window based on specific logic per filter.
                     # Let's keep it simple: fetch enough for the max requirement.
                     req_window = window
                     if flt.config_key == "regime_filter":
                         req_window = window + 15
                     elif flt.config_key == "volatility_filter":
                         req_window = window + 1
                     elif flt.config_key == "trend_filter":
                         req_window = window + 1
                     elif flt.config_key == "momentum_filter":
                         req_window = window + 1
                     
                     if req_window > 0:
                        max_window = max(max_window, req_window)
                        active_filters.append((flt, config))

        if not active_filters:
            return decision

        # 2. Fetch Data
        candles = await self._fetch_spot_candles(context.symbol, max_window)
        if not candles:
            return Decision(action=Action.HOLD, reason="spot_filters_wait")
            
        # 3. Apply Filters
        for flt, config in active_filters:
            result = flt.check(context, config, candles)
            if result:
                return result

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
