"""Runtime UI adapter for the TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpt_trader.app.runtime_ui_adapter import RuntimeUIAdapter
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.bot import TradingBot
    from gpt_trader.monitoring.status_reporter import BotStatus
    from gpt_trader.tui.app import TraderApp

logger = get_logger(__name__, component="tui")


class TuiRuntimeUIAdapter(RuntimeUIAdapter):
    """Bridge TradingBot status updates into the TUI."""

    def __init__(self, app: TraderApp) -> None:
        self._app = app
        self._bot: TradingBot | None = None
        self._attached = False

    def attach(self, bot: TradingBot) -> None:
        if self._attached and self._bot is bot:
            return

        self.detach()
        self._bot = bot

        if not hasattr(bot, "engine") or not hasattr(bot.engine, "status_reporter"):
            logger.warning("Cannot attach UI adapter: status reporter missing")
            return

        if hasattr(self._app, "_is_real_status_reporter"):
            try:
                if not self._app._is_real_status_reporter():
                    logger.debug("Skipping UI adapter attach for null status reporter")
                    return
            except Exception:
                pass

        bot.engine.status_reporter.add_observer(self._handle_status)
        self._attached = True
        logger.debug("Runtime UI adapter attached")

    def detach(self) -> None:
        if not self._attached or not self._bot:
            return

        try:
            if hasattr(self._bot.engine, "status_reporter"):
                self._bot.engine.status_reporter.remove_observer(self._handle_status)
        except Exception as exc:
            logger.debug("Failed to detach UI adapter: %s", exc)
        finally:
            self._attached = False
            self._bot = None

    def _handle_status(self, status: BotStatus) -> None:
        if hasattr(self._app, "_on_status_update"):
            self._app._on_status_update(status)


__all__ = ["TuiRuntimeUIAdapter"]
