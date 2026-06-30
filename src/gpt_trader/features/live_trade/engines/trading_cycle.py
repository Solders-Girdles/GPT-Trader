"""Trading cycle loop for the live TradingEngine.

The main run loop and per-cycle orchestration: fetch positions, compute equity,
run each symbol through the strategy, and report status. Extracted from
strategy.py following the engine's collaborator-function pattern.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from gpt_trader.logging.correlation import correlation_context
from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.monitoring.metrics_collector import record_histogram
from gpt_trader.monitoring.tracing import trace_span
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.engines.strategy import TradingEngine

logger = get_logger(__name__, component="trading_engine")


async def run_loop(engine: TradingEngine) -> None:
    logger.info("Starting strategy loop...")
    while engine.running:
        try:
            await run_cycle(engine)
            # Record successful cycle
            engine._status_reporter.record_cycle()
        except Exception as e:
            logger.error(f"Error in strategy cycle: {e}", exc_info=True)
            # Record error in status reporter
            engine._status_reporter.record_error(str(e))
            await engine._notify(
                title="Strategy Cycle Error",
                message=f"Error during trading cycle: {e}",
                severity=AlertSeverity.ERROR,
                context={"error": str(e)},
            )

        await asyncio.sleep(engine.context.config.interval)


def report_system_status(engine: TradingEngine) -> None:
    """Collect and report system health metrics.

    Delegates to SystemMaintenanceService for the actual reporting.
    """
    engine._system_maintenance.report_system_status(
        latency_seconds=engine._last_latency,
        connection_status=engine._connection_status,
    )


async def run_cycle(engine: TradingEngine) -> None:
    """One trading cycle."""
    assert engine.context.broker is not None, "Broker not initialized"
    engine._cycle_count += 1

    # Wrap entire cycle in correlation context and trace span
    start_time = time.perf_counter()
    result = "ok"
    with correlation_context(cycle=engine._cycle_count):
        with trace_span("cycle", {"cycle": engine._cycle_count}) as span:
            try:
                await run_cycle_inner(engine)
            except Exception:
                result = "error"
                if span:
                    span.set_attribute("error", True)
                raise
            finally:
                duration = time.perf_counter() - start_time
                if span:
                    span.set_attribute("duration_seconds", duration)
                    span.set_attribute("result", result)
                record_histogram(
                    "gpt_trader_cycle_duration_seconds",
                    duration,
                    labels={"result": result},
                )


async def run_cycle_inner(engine: TradingEngine) -> None:
    """Inner cycle logic wrapped in correlation context."""
    logger.info(f"=== CYCLE {engine._cycle_count} START ===")

    # Report system status at start of cycle
    report_system_status(engine)
    broker = engine.context.broker
    if broker is None:
        logger.error("Broker not initialized", operation="cycle")
        engine._connection_status = "DISCONNECTED"
        return

    positions, audit_task = await engine._fetch_positions_and_audit()
    equity = await engine._compute_equity(positions)
    if equity is None:
        await engine._await_audit_task(audit_task, context="during equity error path")
        return

    await engine._await_audit_task(audit_task, context="post equity")
    engine._update_equity_and_risk(equity)

    # Ensure symbols is a list to avoid iterator exhaustion during multiple iterations
    symbols = list(engine.context.config.symbols)
    tickers = await engine._fetch_batch_tickers(broker, symbols)

    tasks = [
        engine._process_symbol(
            symbol=symbol,
            broker=broker,
            ticker=tickers.get(symbol),
            positions=positions,
            equity=equity,
        )
        for symbol in symbols
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    failures: list[Exception] = []

    for symbol, res in zip(symbols, results):
        if isinstance(res, Exception):
            logger.error(
                f"Failed to process symbol {symbol}: {res}",
                exc_info=res,
                symbol=symbol,
            )
            failures.append(res)

    if failures:
        raise ExceptionGroup("Cycle completed with symbol processing errors", failures)
