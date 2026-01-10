"""Runtime UI adapter protocol and null implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.bot import TradingBot


class RuntimeUIAdapter(Protocol):
    """Interface for attaching UI behavior to the runtime."""

    def attach(self, bot: TradingBot) -> None:
        """Attach to a TradingBot instance."""

    def detach(self) -> None:
        """Detach from the current bot."""


class NullUIAdapter:
    """No-op UI adapter for headless runtime usage."""

    def attach(self, bot: TradingBot) -> None:
        return None

    def detach(self) -> None:
        return None


__all__ = ["RuntimeUIAdapter", "NullUIAdapter"]
