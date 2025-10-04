"""Display loop controller for paper trading dashboard."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot_v2.features.paper_trade.dashboard.main import PaperTradingDashboard

logger = logging.getLogger(__name__)


class DisplayController:
    """Controls dashboard display loops and screen management."""

    def __init__(self, dashboard: PaperTradingDashboard, refresh_interval: int = 5) -> None:
        """
        Initialize display controller.

        Args:
            dashboard: Dashboard instance to control
            refresh_interval: Seconds between refreshes in continuous mode
        """
        self.dashboard = dashboard
        self.refresh_interval = refresh_interval

    def clear_screen(self) -> None:
        """Clear the console screen."""
        os.system("clear" if os.name == "posix" else "cls")

    def display_once(self) -> None:
        """Display dashboard once without clearing screen."""
        metrics = self.dashboard.calculate_metrics()

        self.dashboard.print_header()
        self.dashboard.print_portfolio_summary(metrics)
        self.dashboard.print_positions()
        self.dashboard.print_performance(metrics)
        self.dashboard.print_recent_trades()
        print("\n" + "=" * 80)

    def display_continuous(self, duration: int | None = None) -> None:
        """
        Display dashboard continuously with refresh.

        Args:
            duration: Total seconds to run, or None for infinite
        """
        start = time.time()

        try:
            while True:
                self.clear_screen()
                self.display_once()

                # Check duration
                if duration and (time.time() - start) >= duration:
                    break

                # Show refresh countdown
                print(f"\nRefreshing in {self.refresh_interval} seconds... (Press Ctrl+C to stop)")
                time.sleep(self.refresh_interval)

        except KeyboardInterrupt:
            print("\n\nDashboard stopped by user")
            logger.info("Dashboard stopped by user")
