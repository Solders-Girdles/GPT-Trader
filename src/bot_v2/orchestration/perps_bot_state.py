"""Runtime state container for :mod:`bot_v2.orchestration.perps_bot`."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from decimal import Decimal
from typing import Any


class PerpsBotRuntimeState:
    """Mutable state tracked by ``PerpsBot`` during execution."""

    def __init__(self, symbols: Sequence[str]) -> None:
        self.reset(symbols)

    def reset(self, symbols: Sequence[str]) -> None:
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
