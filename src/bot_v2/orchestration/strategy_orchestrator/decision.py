"""Decision evaluation helpers for strategy orchestration."""

from __future__ import annotations

import time as _time
from collections.abc import Sequence
from decimal import Decimal
from typing import Any, cast

from bot_v2.features.brokerages.core.interfaces import Product
from bot_v2.features.live_trade.strategies.perps_baseline import (
    BaselinePerpsStrategy,
    Decision,
)
from bot_v2.logging import log_strategy_decision
from bot_v2.monitoring.system import get_logger as _get_plog
from bot_v2.orchestration.configuration import Profile

from .logging_utils import logger
from .models import SymbolProcessingContext


class DecisionEngineMixin:
    """Evaluate strategy decisions and record results."""

    async def _resolve_decision(self, symbol_context: SymbolProcessingContext) -> Decision:
        strategy = self.get_strategy(symbol_context.symbol)
        decision = self._evaluate_strategy(
            strategy,
            symbol_context.symbol,
            symbol_context.marks,
            symbol_context.position_state,
            symbol_context.equity,
            symbol_context.product,
        )

        if self._bot.config.profile == Profile.SPOT:
            decision = await self._apply_spot_filters(symbol_context, decision)
        return decision

    def _evaluate_strategy(
        self,
        strategy: BaselinePerpsStrategy,
        symbol: str,
        marks: Sequence[Decimal],
        position_state: dict[str, Any] | None,
        equity: Decimal,
        product: Product | None,
    ) -> Decision:
        _t0 = _time.perf_counter()
        product_meta = product
        if product_meta is None and hasattr(strategy, "_build_default_product"):
            try:
                product_meta = cast(Product, strategy._build_default_product(symbol))  # type: ignore[attr-defined]
            except Exception:
                product_meta = None
        if product_meta is None:
            raise ValueError(f"Missing product metadata for {symbol}")

        decision = strategy.decide(
            symbol=symbol,
            current_mark=marks[-1],
            position_state=position_state,
            recent_marks=list(marks[:-1]) if len(marks) > 1 else [],
            equity=equity,
            product=product_meta,
        )
        _dt_ms = (_time.perf_counter() - _t0) * 1000.0
        try:
            _get_plog().log_strategy_duration(strategy=type(strategy).__name__, duration_ms=_dt_ms)
        except Exception as exc:
            logger.debug("Failed to log strategy duration: %s", exc, exc_info=True)
        return decision

    def _record_decision(self, symbol: str, decision: Decision) -> None:
        self._bot.last_decisions[symbol] = decision
        logger.info(f"{symbol} Decision: {decision.action.value} - {decision.reason}")
        log_strategy_decision(
            symbol=symbol,
            decision=decision.action.value,
            reason=getattr(decision, "reason", None),
            confidence=getattr(decision, "confidence", None),
        )


__all__ = ["DecisionEngineMixin"]
