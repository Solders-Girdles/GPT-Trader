"""Symbol processing extension interfaces for ``PerpsBot`` orchestration."""

from __future__ import annotations

from collections.abc import Awaitable, Sequence
from typing import Protocol

from bot_v2.features.brokerages.core.interfaces import Balance, Position


class SymbolProcessor(Protocol):
    """Extension point for symbol-level strategy execution."""

    requires_context: bool

    def process_symbol(
        self,
        symbol: str,
        balances: Sequence[Balance] | None = None,
        position_map: dict[str, Position] | None = None,
    ) -> Awaitable[None] | None:
        """Trigger processing for ``symbol`` and optionally use shared context."""
        ...
