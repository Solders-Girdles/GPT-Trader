"""Bot lifecycle controller for CLI run command.

Provides LifecycleController to manage async bot execution with injectable runner.
"""

from collections.abc import Callable
from typing import Any


class LifecycleController:
    """Controls bot lifecycle execution."""

    def __init__(self, runner: Callable[[Any], Any] | None = None) -> None:
        """
        Initialize lifecycle controller.

        Args:
            runner: Function to execute coroutine (default: asyncio.run)
        """
        if runner is None:
            import asyncio

            runner = asyncio.run

        self._runner = runner

    def execute(self, bot: Any, single_cycle: bool) -> int:
        """
        Execute bot lifecycle.

        Args:
            bot: PerpsBot instance with run() coroutine
            single_cycle: If True, run single cycle; if False, run continuously

        Returns:
            Exit code (0 for success)

        Raises:
            Exception: Propagates any exception from bot execution
        """
        self._runner(bot.run(single_cycle=single_cycle))
        return 0
