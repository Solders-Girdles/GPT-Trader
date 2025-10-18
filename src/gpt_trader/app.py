"""Application composition root for the next-generation GPT-Trader stack."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from bot_v2.utilities.logging_patterns import get_logger
from gpt_trader.data import MarketData, YahooMarketData
from gpt_trader.domain import Bar

from .settings import Settings, get_settings

logger = get_logger("gpt_trader.app", component="gpt_trader", enable_console=True)


def run(
    symbols: Sequence[str] | None = None,
    cfg: Settings | None = None,
    *,
    lookback: int = 120,
    interval: str = "1d",
    market_data: MarketData | None = None,
    log_dir: Path | None = None,
) -> None:
    """Kick off a trading cycle for the provided ``symbols``.

    The implementation is intentionally lean: Stage A establishes the seam that future
    refactors can attach to without disturbing the existing ``bot_v2`` runtime.
    """

    active_settings = cfg or get_settings()
    if log_dir is not None:
        active_settings = active_settings.model_copy(update={"log_dir": log_dir})
    target_symbols: Sequence[str] = symbols or ("BTC-USD",)

    logger.info(
        f"Bootstrapping with model={active_settings.model} "
        f"symbols={list(target_symbols)} log_dir={active_settings.log_dir}",
        operation="bootstrap",
        status="starting",
    )

    data_source = market_data or YahooMarketData()

    for symbol in target_symbols:
        bars = list(data_source.bars(symbol, lookback=lookback, interval=interval))
        if not bars:
            logger.warning(
                f"No market data returned for symbol={symbol} interval={interval}",
                symbol=symbol,
            )
            continue

        latest: Bar = bars[-1]
        logger.info(
            f"Fetched market data bars={len(bars)} interval={interval} "
            f"lookback={lookback} last_close={float(latest.close)}",
            symbol=symbol,
            operation="market_data_update",
        )

    # TODO: wire in data providers, strategies, and broker adapters as they migrate
    # into the ``gpt_trader`` package. For now we simply log the configuration seam.


__all__ = ["run"]
