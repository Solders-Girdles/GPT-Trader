"""Strategy orchestrator implementation composed from focused mixins."""

from __future__ import annotations

from collections.abc import Sequence

from gpt_trader.core import Balance, Position
from gpt_trader.features.live_trade.strategies.perps_baseline import Action
from gpt_trader.logging import correlation_context, log_execution_error, symbol_context

from .context import ContextBuilderMixin
from .decision import DecisionEngineMixin
from .initialization import StrategyInitializationMixin
from .logging_utils import json_logger, logger  # naming: allow
from .models import SymbolProcessingContext
from .spot_filters import SpotFiltersMixin


class StrategyOrchestrator(
    SpotFiltersMixin,
    DecisionEngineMixin,
    ContextBuilderMixin,
    StrategyInitializationMixin,
):
    """Encapsulates strategy initialization and decision execution per symbol."""

    requires_context = True

    async def process_symbol(
        self,
        symbol: str,
        balances: Sequence[Balance] | None = None,
        position_map: dict[str, Position] | None = None,
    ) -> None:
        with correlation_context(operation="process_symbol"), symbol_context(symbol):
            try:
                context = await self._prepare_context(symbol, balances, position_map)
                if context is None:
                    return

                decision = await self._resolve_decision(context)
                self._record_decision(symbol, decision)

                if decision.action in {Action.BUY, Action.SELL, Action.CLOSE}:
                    product = context.product
                    if product is None:
                        logger.warning(
                            "Skipping execution for %s: missing product metadata",
                            symbol,
                            operation="strategy_execute",
                            stage="missing_product",
                            symbol=symbol,
                        )
                        json_logger.warning(
                            "Skipping execution: missing product metadata",
                            extra={
                                "symbol": symbol,
                                "operation": "strategy_execute",
                                "stage": "missing_product",
                            },
                        )
                        return
                    await self._bot.execute_decision(
                        symbol,
                        decision,
                        context.marks[-1],
                        product,
                        context.position_state,
                    )
            except Exception as exc:
                logger.error(
                    "Error processing %s: %s",
                    symbol,
                    exc,
                    exc_info=True,
                    operation="strategy_execute",
                    stage="process_symbol",
                    symbol=symbol,
                )
                log_execution_error(error=exc, operation="process_symbol", symbol=symbol)


__all__ = ["StrategyOrchestrator", "SymbolProcessingContext"]
