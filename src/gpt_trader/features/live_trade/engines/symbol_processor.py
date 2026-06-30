"""Per-symbol processing for the live TradingEngine trading cycle.

Fetches candles for a symbol, runs the strategy to produce a Decision, and hands
it to the decision handler. Extracted from strategy.py following the engine's
collaborator-function pattern.
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from gpt_trader.core import Position
from gpt_trader.monitoring.profiling import profile_span
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.engines.strategy import TradingEngine

logger = get_logger(__name__, component="trading_engine")


async def process_symbol(
    engine: TradingEngine,
    *,
    symbol: str,
    broker: Any,
    ticker: dict[str, Any] | None,
    positions: dict[str, Position],
    equity: Decimal,
) -> None:
    candles: list[Any] = []
    start_time = time.time()

    if ticker is None:
        try:
            ticker = await engine._broker_calls(broker.get_ticker, symbol)
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            engine._connection_status = "DISCONNECTED"
            return

    if ticker is None or not ticker.get("price"):
        logger.error(f"No ticker data for {symbol}")
        engine._connection_status = "DISCONNECTED"
        return

    try:
        candles_result = await engine._broker_calls(
            broker.get_candles,
            symbol,
            granularity="ONE_MINUTE",
        )
        if isinstance(candles_result, Exception):
            logger.warning(f"Failed to fetch candles for {symbol}: {candles_result}")
        else:
            candles = candles_result or []
    except Exception as e:
        logger.warning(f"Failed to fetch candles for {symbol}: {e}")

    engine._last_latency = time.time() - start_time
    engine._connection_status = "CONNECTED"

    price = Decimal(str(ticker.get("price", 0)))
    logger.info(f"{symbol} price: {price}")

    if engine.context.risk_manager is not None:
        engine.context.risk_manager.last_mark_update[symbol] = time.time()

    engine._status_reporter.update_price(symbol, price)
    await engine._price_tick_store.record_price_tick_async(symbol, price)

    position_state = engine._build_position_state(symbol, positions)
    with profile_span("strategy_decision", {"symbol": symbol}) as _strat_span:
        decision = engine.strategy.decide(
            symbol=symbol,
            current_mark=price,
            position_state=position_state,
            recent_marks=engine.price_history[symbol],
            equity=equity,
            product=None,
            candles=candles,
        )

    logger.info(f"Strategy Decision for {symbol}: {decision.action} ({decision.reason})")

    active_strats = getattr(
        engine.strategy, "active_strategies", [engine.strategy.__class__.__name__]
    )
    decision_record = {
        "symbol": symbol,
        "action": decision.action.value,
        "reason": decision.reason,
        "confidence": str(decision.confidence),
        "timestamp": time.time(),
    }
    engine._status_reporter.update_strategy(active_strats, [decision_record])

    await engine._handle_decision(
        symbol=symbol,
        decision=decision,
        price=price,
        equity=equity,
        position_state=position_state,
    )
