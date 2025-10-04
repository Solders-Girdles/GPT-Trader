"""Signal manager for CLI run command.

Provides SignalManager to setup shutdown signal handlers with injectable handler class.
"""

from collections.abc import Callable
from typing import Any


class SignalManager:
    """Manages signal handler setup for bot lifecycle."""

    def __init__(self, handler_class: Callable[[Any], Any] | None = None) -> None:
        """
        Initialize signal manager.

        Args:
            handler_class: Class to instantiate for signal handling (default: ShutdownHandler)
        """
        if handler_class is None:
            from bot_v2.cli.handlers.shutdown import ShutdownHandler

            handler_class = ShutdownHandler

        self._handler_class = handler_class

    def setup_signals(self, bot: Any) -> None:
        """
        Setup signal handlers for bot.

        Args:
            bot: PerpsBot instance to handle shutdown signals
        """
        handler = self._handler_class(bot)
        handler.register_signals()
