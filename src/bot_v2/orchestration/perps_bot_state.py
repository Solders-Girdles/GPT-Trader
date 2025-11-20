"""Runtime state container for :mod:`bot_v2.orchestration.perps_bot`."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from bot_v2.features.brokerages.coinbase.account_manager import CoinbaseAccountManager
    from bot_v2.orchestration.account_telemetry import AccountTelemetryService
    from bot_v2.orchestration.intx_portfolio_service import IntxPortfolioService
    from bot_v2.orchestration.market_monitor import MarketActivityMonitor


class PerpsBotRuntimeState:
    """Mutable state tracked by ``PerpsBot`` during execution."""

    def __init__(self, symbols: Sequence[str]) -> None:
        self.symbols = list(symbols)
        self.reset(symbols)

    def reset(self, symbols: Sequence[str]) -> None:
        self.symbols = list(symbols)
        self.product_map: dict[str, Any] = {}
        self.mark_windows: dict[str, list[Decimal]] = {symbol: [] for symbol in symbols}
        self.last_decisions: dict[str, Any] = {}
        self.last_positions: dict[str, dict[str, Any]] = {}
        self.order_stats: dict[str, int] = {"attempted": 0, "successful": 0, "failed": 0}
        self.order_lock: asyncio.Lock | None = None
        self.mark_lock = threading.RLock()
        self.symbol_strategies: dict[str, Any] = {}
        self.strategy: Any | None = None
        self.exec_engine: Any | None = None
        self.process_symbol_dispatch: Any | None = None
        self.process_symbol_needs_context: bool | None = None
        self.account_manager: CoinbaseAccountManager | None = None
        self.account_telemetry: AccountTelemetryService | None = None
        self.market_monitor: MarketActivityMonitor | None = None
        self.intx_portfolio_service: IntxPortfolioService | None = None
