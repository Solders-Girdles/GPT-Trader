"""Bootstrap mixin for TraderApp.

Contains methods related to initial data loading and
read-only data feed startup.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from gpt_trader.tui.notification_helpers import notify_error, notify_warning
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.app import TraderApp

logger = get_logger(__name__, component="tui")


class TraderAppBootstrapMixin:
    """Mixin providing bootstrap methods for TraderApp.

    Methods:
    - _start_readonly_data_feed: Auto-start data feed for read-only mode
    - request_bootstrap_snapshot: Request one-time bootstrap snapshot
    - bootstrap_snapshot: Fetch balances and market snapshot
    """

    # Type hints for attributes from TraderApp
    if TYPE_CHECKING:
        bot: Any
        data_source_mode: str
        tui_state: Any
        ui_coordinator: Any
        worker_service: Any
        _bootstrap_snapshot_requested: bool
        _bootstrap_snapshot_inflight: bool

        def _sync_state_from_bot(self) -> None: ...

    async def _start_readonly_data_feed(self: TraderApp) -> None:
        """Auto-start data feed for read-only mode.

        Called via call_later() after initialization to fetch initial data
        without requiring the user to press 'S'.
        """
        try:
            logger.debug("Starting read-only data feed")

            # Request bootstrap snapshot for initial data
            request_bootstrap = getattr(self, "request_bootstrap_snapshot", None)
            if callable(request_bootstrap):
                request_bootstrap(force=True)

            # Sync state from StatusReporter
            if self.ui_coordinator:
                self.ui_coordinator.sync_state_from_bot()

            self.tui_state.data_fetching = False
            logger.info("Read-only data feed started successfully")
        except Exception as e:
            self.tui_state.data_fetching = False
            logger.error(f"Failed to start read-only data feed: {e}", exc_info=True)
            notify_error(self, f"Failed to start data feed: {e}", title="Error")

    def request_bootstrap_snapshot(self: TraderApp, *, force: bool = False) -> None:
        """Request a one-time bootstrap snapshot in the background.

        This is used on startup to populate account balances and a basic market
        snapshot even when the bot is STOPPED (manual-start policy).
        """
        if self.data_source_mode == "demo":
            return

        # Avoid scheduling bootstrap work when there is no real broker attached.
        # This also prevents MagicMock auto-attribute creation in unit tests.
        try:
            if self.bot is None:
                return
            bot_dict = getattr(self.bot, "__dict__", None)
            if isinstance(bot_dict, dict) and bot_dict.get("broker") is None:
                return
        except Exception:
            pass

        if self._bootstrap_snapshot_inflight:
            return

        if self._bootstrap_snapshot_requested and not force:
            return

        self._bootstrap_snapshot_requested = True

        if not self.worker_service:
            return

        async def fetch() -> None:
            await self.bootstrap_snapshot()

        self._bootstrap_snapshot_inflight = True
        self.worker_service.run_data_fetch(fetch, name="bootstrap_snapshot")

    async def bootstrap_snapshot(self: TraderApp) -> bool:
        """Fetch balances + a minimal market snapshot and push into the UI.

        Returns:
            True if any data was fetched and applied.
        """
        # Avoid doing work in demo mode or while bot is running.
        if self.data_source_mode == "demo":
            self._bootstrap_snapshot_inflight = False
            return False

        if self.bot is None:
            self._bootstrap_snapshot_inflight = False
            return False

        if bool(getattr(self.bot, "running", False)):
            self._bootstrap_snapshot_inflight = False
            return False

        # Avoid MagicMock attr auto-creation in tests by preferring __dict__.
        broker = None
        try:
            bot_dict = getattr(self.bot, "__dict__", {})
            broker = bot_dict.get("broker") if isinstance(bot_dict, dict) else None
        except Exception:
            broker = None

        # Fallback only if the bot doesn't expose __dict__ (avoid MagicMock attr creation).
        if broker is None and not isinstance(getattr(self.bot, "__dict__", None), dict):
            try:
                broker = getattr(self.bot, "broker", None)
            except Exception:
                broker = None

        if broker is None:
            self._bootstrap_snapshot_inflight = False
            return False

        # StatusReporter is the source of truth for sync_state_from_bot().
        reporter = None
        try:
            reporter = getattr(getattr(self.bot, "engine", None), "status_reporter", None)
        except Exception:
            reporter = None

        if reporter is None or getattr(reporter, "is_null_reporter", False):
            self._bootstrap_snapshot_inflight = False
            return False

        from gpt_trader.core import Balance as CoreBalance
        from gpt_trader.tui.formatting import safe_decimal

        stable_assets = {"USD", "USDC", "USDT", "DAI"}

        def fetch_sync() -> tuple[list[CoreBalance], dict[str, Decimal], Decimal]:
            """Run blocking broker calls in a single thread for session safety."""
            balances: list[CoreBalance] = []
            prices: dict[str, Decimal] = {}

            # 1) Balances
            try:
                balances = list(broker.list_balances() or [])
            except Exception as exc:
                raise RuntimeError(f"Failed to fetch balances: {exc}") from exc

            non_zero = [b for b in balances if getattr(b, "total", Decimal("0")) > 0]

            # 2) Market prices: bot symbols first, then holdings (cap for sanity)
            symbols: list[str] = []
            seen: set[str] = set()

            try:
                config_symbols = list(
                    getattr(getattr(self.bot, "config", None), "symbols", []) or []
                )
            except Exception:
                config_symbols = []

            for sym in config_symbols:
                sym_str = str(sym)
                if sym_str and sym_str not in seen:
                    seen.add(sym_str)
                    symbols.append(sym_str)

            for bal in non_zero:
                asset = str(getattr(bal, "asset", "") or "").upper()
                if not asset or asset in stable_assets:
                    continue
                for quote in ("USD", "USDC"):
                    product_id = f"{asset}-{quote}"
                    if product_id not in seen:
                        seen.add(product_id)
                        symbols.append(product_id)
                        break

            # Hard cap to avoid excessive HTTP calls on startup.
            symbols = symbols[:25]

            for product_id in symbols:
                try:
                    ticker = broker.get_ticker(product_id) or {}
                    price = safe_decimal(
                        ticker.get("price")
                        or ticker.get("last")
                        or ticker.get("trade_price")
                        or "0"
                    )
                    if price > 0:
                        prices[product_id] = price
                except Exception:
                    continue

            # 3) Estimate equity in USD using stable balances + known tickers.
            equity = Decimal("0")
            for bal in non_zero:
                asset = str(getattr(bal, "asset", "") or "").upper()
                total = safe_decimal(getattr(bal, "total", "0"))
                if total <= 0:
                    continue
                if asset in stable_assets:
                    equity += total
                    continue

                usd_price = prices.get(f"{asset}-USD")
                if usd_price is None:
                    usdc_price = prices.get(f"{asset}-USDC")
                    usd_price = usdc_price
                if usd_price is not None and usd_price > 0:
                    equity += total * usd_price

            return non_zero, prices, equity

        try:
            balances, prices, equity = await asyncio.to_thread(fetch_sync)
        except Exception as e:
            logger.warning("Bootstrap snapshot failed: %s", e)
            try:
                notify_warning(self, f"Failed to fetch account snapshot: {e}", title="Startup")
            except Exception:
                pass
            self._bootstrap_snapshot_inflight = False
            return False

        # Apply to StatusReporter (so subsequent syncs preserve it).
        try:
            reporter.update_account(balances, summary={})
            reporter.update_equity(equity)
            for symbol, price in prices.items():
                reporter.update_price(symbol, price)
        except Exception as e:
            logger.debug(
                "Failed applying bootstrap snapshot to StatusReporter: %s", e, exc_info=True
            )

        # Refresh TuiState/UI from StatusReporter.
        try:
            if self.ui_coordinator:
                self.ui_coordinator.sync_state_from_bot()
                self.ui_coordinator.update_main_screen()
            else:
                self._sync_state_from_bot()
        except Exception:
            pass

        self._bootstrap_snapshot_inflight = False
        return bool(balances or prices)
