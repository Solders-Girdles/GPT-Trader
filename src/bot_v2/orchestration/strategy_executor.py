"""Strategy execution and decision recording with telemetry.

This module provides utilities for executing trading strategies, measuring performance,
and recording decisions with proper telemetry logging.
"""

from __future__ import annotations

import logging
import time as _time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bot_v2.monitoring.system import get_logger as _get_plog

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bot_v2.features.live_trade.strategies.perps_baseline import (
        BaselinePerpsStrategy,
        Decision,
    )
    from bot_v2.orchestration.perps_bot import PerpsBot

logger = logging.getLogger(__name__)


class StrategyExecutor:
    """Executes strategies and records decisions with telemetry.

    This executor is responsible for:
    1. Calling strategy.decide() with proper parameters
    2. Measuring execution time
    3. Logging telemetry (strategy duration)
    4. Recording decision to bot.last_decisions
    5. Logging decision at INFO level

    Example:
        >>> executor = StrategyExecutor(bot)
        >>> decision = executor.evaluate_and_record(
        ...     strategy, "BTC-USD", marks, position_state, equity
        ... )
    """

    def __init__(self, bot: PerpsBot) -> None:
        """Initialize strategy executor.

        Args:
            bot: Bot instance for product retrieval and decision storage
        """
        self._bot = bot

    def evaluate_and_record(
        self,
        strategy: BaselinePerpsStrategy,
        symbol: str,
        marks: Sequence[Decimal],
        position_state: dict[str, Any] | None,
        equity: Decimal,
    ) -> Decision:
        """Evaluate strategy and record decision with telemetry.

        Use this when no intermediate processing is needed between evaluation
        and recording. For cases where decision needs modification (e.g., SPOT filters),
        use evaluate() then record_decision() separately.

        Args:
            strategy: Strategy instance to execute
            symbol: Trading symbol
            marks: Recent mark prices (current mark is last element)
            position_state: Current position state (quantity, side, entry)
            equity: Total portfolio equity

        Returns:
            Decision from strategy evaluation

        Example:
            >>> marks = [Decimal("50000"), Decimal("50100"), Decimal("50200")]
            >>> decision = executor.evaluate_and_record(
            ...     strategy, "BTC-USD", marks, None, Decimal("10000")
            ... )
        """
        # Evaluate strategy with timing
        decision = self.evaluate(strategy, symbol, marks, position_state, equity)

        # Record decision
        self.record_decision(symbol, decision)

        return decision

    def evaluate(
        self,
        strategy: BaselinePerpsStrategy,
        symbol: str,
        marks: Sequence[Decimal],
        position_state: dict[str, Any] | None,
        equity: Decimal,
    ) -> Decision:
        """Evaluate strategy with performance timing.

        Use this when you need to modify the decision before recording
        (e.g., applying SPOT filters). Follow up with record_decision().

        Args:
            strategy: Strategy instance
            symbol: Trading symbol
            marks: Recent mark prices
            position_state: Current position state
            equity: Portfolio equity

        Returns:
            Decision from strategy

        Example:
            >>> decision = executor.evaluate(strategy, "BTC-USD", marks, None, Decimal("10000"))
            >>> # Apply filters or modifications
            >>> executor.record_decision("BTC-USD", decision)
        """
        _t0 = _time.perf_counter()

        decision = strategy.decide(
            symbol=symbol,
            current_mark=marks[-1],
            position_state=position_state,
            recent_marks=list(marks[:-1]) if len(marks) > 1 else [],
            equity=equity,
            product=self._bot.get_product(symbol),
        )

        _dt_ms = (_time.perf_counter() - _t0) * 1000.0

        # Log telemetry
        try:
            _get_plog().log_strategy_duration(strategy=type(strategy).__name__, duration_ms=_dt_ms)
        except Exception as exc:
            logger.debug("Failed to log strategy duration: %s", exc, exc_info=True)

        return decision

    def record_decision(self, symbol: str, decision: Decision) -> None:
        """Record decision to bot and log.

        Args:
            symbol: Trading symbol
            decision: Strategy decision

        Example:
            >>> executor.record_decision("BTC-USD", decision)
        """
        self._bot.last_decisions[symbol] = decision
        logger.info(f"{symbol} Decision: {decision.action.value} - {decision.reason}")
