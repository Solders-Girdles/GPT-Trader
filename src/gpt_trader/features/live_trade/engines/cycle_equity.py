"""Equity and ticker acquisition for the live TradingEngine trading cycle.

Computes account equity from positions, updates risk-manager state, and fetches
batched tickers from the broker. Extracted from strategy.py following the
engine's collaborator-function pattern.
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


async def compute_equity(engine: TradingEngine, positions: dict[str, Position]) -> Decimal | None:
    logger.info("Step 2: Calculating total equity...")
    with profile_span("equity_computation") as _eq_span:
        equity = await engine._fetch_total_equity(positions)
    if equity is None:
        logger.error(
            "Failed to fetch equity - cannot continue cycle. "
            "Check logs above for balance fetch errors."
        )
        engine._status_reporter.record_error("Failed to fetch equity")
        return None
    return equity


def update_equity_and_risk(engine: TradingEngine, equity: Decimal) -> None:
    logger.info(f"Successfully calculated equity: ${equity}")
    engine._status_reporter.update_equity(equity)
    logger.info("Equity updated in status reporter")

    if engine.context.risk_manager:
        triggered = engine.context.risk_manager.track_daily_pnl(equity, {})
        if triggered:
            logger.warning("Daily loss limit triggered! Reduce-only mode activated.")

        rm = engine.context.risk_manager
        daily_loss_pct = 0.0
        start_equity = getattr(rm, "_start_of_day_equity", 0)
        if start_equity and start_equity > 0:
            daily_pnl = equity - start_equity
            daily_loss_pct = float(-daily_pnl / start_equity)

        engine._status_reporter.update_risk(
            max_leverage=float(getattr(rm.config, "max_leverage", 0.0) if rm.config else 0.0),
            daily_loss_limit=float(
                getattr(rm.config, "daily_loss_limit_pct", 0.0) if rm.config else 0.0
            ),
            current_daily_loss=daily_loss_pct,
            reduce_only=getattr(rm, "_reduce_only_mode", False),
            reduce_reason=getattr(rm, "_reduce_only_reason", ""),
        )


async def fetch_batch_tickers(
    engine: TradingEngine, broker: Any, symbols: list[str]
) -> dict[str, dict[str, Any]]:
    tickers: dict[str, dict[str, Any]] = {}
    batch_start = time.time()

    get_tickers_method = getattr(broker, "get_tickers", None)
    if get_tickers_method is not None and callable(get_tickers_method):
        try:
            result = await engine._broker_calls(get_tickers_method, symbols)
            if isinstance(result, dict):
                tickers = result
                logger.debug(
                    f"Batch ticker fetch: {len(tickers)}/{len(symbols)} symbols "
                    f"in {time.time() - batch_start:.3f}s"
                )
        except Exception as e:
            logger.warning(f"Batch ticker fetch failed, falling back to individual: {e}")

    return tickers
