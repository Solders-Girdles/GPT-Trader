"""
Worker Service for TUI Async Operations.

Provides Worker-based async operation management using Textual's Worker API.
Workers run in background threads/tasks and don't block the UI event loop.

Benefits over raw asyncio.create_task():
- Automatic cancellation on app exit
- Built-in error handling and state tracking
- Exclusive mode prevents duplicate operations
- Integration with Textual's lifecycle
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from textual.worker import Worker, WorkerState

from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.app import TraderApp

logger = get_logger(__name__, component="tui")


class WorkerService:
    """
    Manages background workers for TUI async operations.

    Provides centralized worker management with proper lifecycle handling,
    cancellation support, and state tracking.

    Usage:
        worker_service = WorkerService(app)
        worker = worker_service.run_bot_async(bot)
        # Later...
        worker_service.cancel_bot_worker()
    """

    # Worker group names for exclusive operations
    BOT_GROUP = "bot_operations"
    DATA_GROUP = "data_fetch"
    SYNC_GROUP = "state_sync"

    def __init__(self, app: TraderApp) -> None:
        """
        Initialize WorkerService.

        Args:
            app: Reference to the TraderApp instance
        """
        self.app = app
        self._bot_worker: Worker[None] | None = None
        self._active_workers: dict[str, Worker[Any]] = {}

    def run_bot_async(self) -> Worker[None]:
        """
        Start the bot in a background worker.

        Uses exclusive mode - only one bot worker can run at a time.
        Subsequent calls will cancel the previous worker.

        Returns:
            Worker instance for tracking/cancellation
        """
        if self._bot_worker and self._bot_worker.state == WorkerState.RUNNING:
            logger.warning("Bot worker already running, cancelling previous")
            self._bot_worker.cancel()

        async def run_bot() -> None:
            """Worker function to run the bot."""
            logger.info("Bot worker started")
            try:
                await self.app.bot.run()
            except Exception as e:
                logger.error(f"Bot worker error: {e}", exc_info=True)
                raise

        self._bot_worker = self.app.run_worker(
            run_bot,
            name="bot_runner",
            group=self.BOT_GROUP,
            exclusive=True,
            description="Running trading bot",
        )

        # Add completion callback
        self._bot_worker.add_done_callback(self._on_bot_worker_done)

        logger.info(f"Bot worker created: {self._bot_worker.name}")
        return self._bot_worker

    def cancel_bot_worker(self) -> bool:
        """
        Cancel the running bot worker.

        Returns:
            True if a worker was cancelled, False if none was running
        """
        if self._bot_worker and self._bot_worker.state == WorkerState.RUNNING:
            self._bot_worker.cancel()
            logger.info("Bot worker cancellation requested")
            return True
        return False

    def is_bot_worker_running(self) -> bool:
        """Check if the bot worker is currently running."""
        return self._bot_worker is not None and self._bot_worker.state == WorkerState.RUNNING

    def get_bot_worker_state(self) -> WorkerState | None:
        """Get the current state of the bot worker."""
        if self._bot_worker:
            return self._bot_worker.state
        return None

    def run_data_fetch(
        self,
        fetch_func: Callable[[], Any],
        name: str = "data_fetch",
    ) -> Worker[Any]:
        """
        Run a data fetch operation in a background worker.

        Args:
            fetch_func: Async function to execute
            name: Name for the worker (for logging)

        Returns:
            Worker instance
        """

        async def do_fetch() -> Any:
            return await fetch_func()

        worker = self.app.run_worker(
            do_fetch,
            name=name,
            group=self.DATA_GROUP,
            exclusive=False,  # Allow multiple data fetches
            description=f"Fetching {name}",
        )

        self._active_workers[name] = worker
        logger.debug(f"Data fetch worker created: {name}")
        return worker

    def run_state_sync(self) -> Worker[None]:
        """
        Run a state synchronization in a background worker.

        Uses exclusive mode - only one sync can run at a time.

        Returns:
            Worker instance
        """

        async def sync_state() -> None:
            """Sync state from bot to TUI."""
            if hasattr(self.app, "_sync_state_from_bot"):
                self.app._sync_state_from_bot()
            logger.debug("State sync completed")

        worker = self.app.run_worker(
            sync_state,
            name="state_sync",
            group=self.SYNC_GROUP,
            exclusive=True,
            description="Syncing state",
        )

        logger.debug("State sync worker created")
        return worker

    def cancel_all_workers(self) -> int:
        """
        Cancel all active workers.

        Returns:
            Number of workers cancelled
        """
        cancelled = 0

        if self._bot_worker and self._bot_worker.state == WorkerState.RUNNING:
            self._bot_worker.cancel()
            cancelled += 1

        for name, worker in self._active_workers.items():
            if worker.state == WorkerState.RUNNING:
                worker.cancel()
                cancelled += 1

        if cancelled > 0:
            logger.info(f"Cancelled {cancelled} workers")

        return cancelled

    def cleanup(self) -> None:
        """Clean up all workers on service destruction."""
        self.cancel_all_workers()
        self._bot_worker = None
        self._active_workers.clear()
        logger.info("WorkerService cleaned up")

    def _on_bot_worker_done(self, worker: Worker[None]) -> None:
        """Callback when bot worker completes."""
        if worker.state == WorkerState.SUCCESS:
            logger.info("Bot worker completed successfully")
        elif worker.state == WorkerState.CANCELLED:
            logger.info("Bot worker was cancelled")
        elif worker.state == WorkerState.ERROR:
            logger.error(f"Bot worker failed: {worker.error}")

    def get_worker_stats(self) -> dict[str, Any]:
        """
        Get statistics about workers for debugging.

        Returns:
            Dict with worker state information
        """
        stats = {
            "bot_worker": None,
            "active_workers": {},
        }

        if self._bot_worker:
            stats["bot_worker"] = {
                "name": self._bot_worker.name,
                "state": self._bot_worker.state.name,
            }

        for name, worker in self._active_workers.items():
            stats["active_workers"][name] = {
                "state": worker.state.name,
            }

        return stats
