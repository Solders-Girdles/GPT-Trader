"""
Console dashboard for paper trading monitoring.
Displays positions, equity, metrics in a clean format.
"""

import logging
from datetime import datetime
from typing import Any

from bot_v2.features.paper_trade.dashboard.console_renderer import ConsoleRenderer
from bot_v2.features.paper_trade.dashboard.display_controller import DisplayController
from bot_v2.features.paper_trade.dashboard.formatters import DashboardFormatter
from bot_v2.features.paper_trade.dashboard.html_report_generator import HTMLReportGenerator
from bot_v2.features.paper_trade.dashboard.metrics import DashboardMetricsAssembler

logger = logging.getLogger(__name__)


class PaperTradingDashboard:
    """Console dashboard for paper trading monitoring."""

    def __init__(self, engine: Any, refresh_interval: int = 5) -> None:
        """
        Initialize dashboard.

        Args:
            engine: PaperExecutionEngine instance
            refresh_interval: Seconds between updates
        """
        self.engine = engine
        self.refresh_interval = refresh_interval
        self.start_time = datetime.now()
        self.initial_equity = engine.initial_capital

        # Initialize components
        self.formatter = DashboardFormatter()
        self.metrics_assembler = DashboardMetricsAssembler(initial_equity=engine.initial_capital)
        self.renderer = ConsoleRenderer(formatter=self.formatter, start_time=self.start_time)
        self.controller = DisplayController(dashboard=self, refresh_interval=refresh_interval)
        self.html_generator = HTMLReportGenerator(formatter=self.formatter)

    def clear_screen(self) -> None:
        """Clear the console screen."""
        self.controller.clear_screen()

    def format_currency(self, value: float | int) -> str:
        """Backward-compatible currency formatter."""
        return self.formatter.currency_str(value)

    def format_pct(self, value: float | int) -> str:
        """Backward-compatible percentage formatter."""
        return self.formatter.percentage_str(value)

    def calculate_metrics(self) -> dict[str, Any]:
        """Calculate current metrics from engine state."""
        return self.metrics_assembler.calculate(self.engine)

    def print_header(self) -> None:
        """Print dashboard header."""
        self.renderer.render_header(bot_id=self.engine.bot_id)

    def print_portfolio_summary(self, metrics: dict) -> None:
        """Print portfolio summary section."""
        self.renderer.render_portfolio_summary(metrics)

    def print_positions(self) -> None:
        """Print open positions."""
        self.renderer.render_positions(self.engine.positions)

    def print_performance(self, metrics: dict) -> None:
        """Print performance metrics."""
        self.renderer.render_performance(metrics)

    def print_recent_trades(self, limit: int = 5) -> None:
        """Print recent trades."""
        self.renderer.render_recent_trades(self.engine.trades, limit=limit)

    def display_once(self) -> None:
        """Display dashboard once without clearing screen."""
        self.controller.display_once()

    def display_continuous(self, duration: int | None = None) -> None:
        """
        Display dashboard continuously with refresh.

        Args:
            duration: Total seconds to run, or None for infinite
        """
        self.controller.display_continuous(duration)

    def generate_html_summary(self, output_path: str | None = None) -> str:
        """
        Generate HTML summary report.

        Args:
            output_path: Path to save HTML file, or None to use default

        Returns:
            Path to saved HTML file
        """
        metrics = self.calculate_metrics()
        return self.html_generator.generate(
            bot_id=self.engine.bot_id,
            metrics=metrics,
            positions=self.engine.positions,
            trades=self.engine.trades,
            output_path=output_path,
        )
