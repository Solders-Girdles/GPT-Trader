"""
Live Performance Dashboard for GPT-Trader Production Trading

Real-time monitoring dashboard that provides comprehensive oversight of:
- Live trading performance and P&L tracking
- Strategy performance metrics and analytics
- Risk monitoring with position and portfolio metrics
- Market data quality and system health
- Order execution quality and transaction costs
- Circuit breaker status and safety monitoring

This is the primary operational dashboard for production trading oversight.
"""

import json
import logging
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


class DashboardStatus(Enum):
    """Dashboard status"""

    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics"""

    timestamp: datetime = field(default_factory=datetime.now)

    # P&L metrics
    total_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")

    # Portfolio metrics
    total_capital: Decimal = Decimal("100000")
    invested_capital: Decimal = Decimal("0")
    cash_balance: Decimal = Decimal("100000")
    portfolio_value: Decimal = Decimal("100000")

    # Position metrics
    total_positions: int = 0
    long_positions: int = 0
    short_positions: int = 0
    largest_position_pct: float = 0.0

    # Risk metrics
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    portfolio_beta: float = 1.0
    portfolio_volatility: float = 0.0
    sharpe_ratio: float = 0.0

    # Trading metrics
    total_trades_today: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    success_rate: float = 0.0
    avg_trade_pnl: Decimal = Decimal("0")

    # System metrics
    active_strategies: int = 0
    active_orders: int = 0
    system_uptime: timedelta = field(default_factory=lambda: timedelta())


@dataclass
class StrategyPerformance:
    """Strategy-specific performance metrics"""

    strategy_id: str

    # Performance
    total_pnl: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    trade_count: int = 0
    win_rate: float = 0.0
    avg_trade_pnl: Decimal = Decimal("0")

    # Risk
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0

    # Positions
    position_count: int = 0
    total_exposure: Decimal = Decimal("0")

    # Status
    is_active: bool = True
    last_trade_time: datetime | None = None
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class SystemAlert:
    """System alert/notification"""

    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    component: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False


class LivePerformanceDashboard:
    """Production live trading performance dashboard"""

    def __init__(
        self,
        dashboard_dir: str = "data/dashboard",
        update_interval: float = 1.0,
        max_alerts: int = 100,
    ) -> None:

        self.dashboard_dir = Path(dashboard_dir)
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.dashboard_dir / "snapshots").mkdir(exist_ok=True)
        (self.dashboard_dir / "alerts").mkdir(exist_ok=True)
        (self.dashboard_dir / "logs").mkdir(exist_ok=True)

        self.update_interval = update_interval
        self.max_alerts = max_alerts

        # Initialize database
        self.db_path = self.dashboard_dir / "dashboard.db"
        self._initialize_database()

        # Dashboard state
        self.status = DashboardStatus.STOPPED
        self.start_time = datetime.now()

        # Data sources (injected by system components)
        self.trading_engines = {}  # engine_id -> engine instance
        self.risk_monitor = None
        self.circuit_breakers = None
        self.streaming_manager = None

        # Performance tracking
        self.current_metrics = PerformanceMetrics()
        self.metrics_history: deque = deque(maxlen=3600)  # 1 hour at 1s intervals
        self.strategy_performance: dict[str, StrategyPerformance] = {}

        # Alert system
        self.active_alerts: deque = deque(maxlen=max_alerts)
        self.alert_history: deque = deque(maxlen=1000)

        # Threading for live updates
        self.update_thread = None
        self.is_running = False

        # Dashboard layout
        self.layout = None
        self.live_display = None

        logger.info("Live Performance Dashboard initialized")

    def _initialize_database(self) -> None:
        """Initialize SQLite database for dashboard data"""

        with sqlite3.connect(self.db_path) as conn:
            # Performance snapshots table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_pnl TEXT NOT NULL,
                    unrealized_pnl TEXT NOT NULL,
                    realized_pnl TEXT NOT NULL,
                    daily_pnl TEXT NOT NULL,
                    total_capital TEXT NOT NULL,
                    portfolio_value TEXT NOT NULL,
                    total_positions INTEGER,
                    total_trades INTEGER,
                    success_rate REAL,
                    current_drawdown REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    metrics_data TEXT
                )
            """
            )

            # Strategy performance table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    strategy_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    total_pnl TEXT NOT NULL,
                    daily_pnl TEXT NOT NULL,
                    trade_count INTEGER,
                    win_rate REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    position_count INTEGER,
                    total_exposure TEXT,
                    is_active BOOLEAN,
                    performance_data TEXT,
                    PRIMARY KEY (strategy_id, timestamp)
                )
            """
            )

            # System alerts table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS system_alerts (
                    alert_id TEXT PRIMARY KEY,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    component TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    resolved BOOLEAN DEFAULT FALSE,
                    alert_metadata TEXT
                )
            """
            )

            # Dashboard sessions table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dashboard_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration_seconds INTEGER,
                    total_updates INTEGER DEFAULT 0,
                    session_metadata TEXT
                )
            """
            )

            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_snapshots_time ON performance_snapshots (timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_strategy_perf_time ON strategy_performance (timestamp)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_time ON system_alerts (timestamp)")

            conn.commit()

    def register_trading_engine(self, engine_id: str, engine_instance) -> None:
        """Register a trading engine for monitoring"""

        self.trading_engines[engine_id] = engine_instance
        console.print(f"   🔗 Registered trading engine: {engine_id}")

    def register_risk_monitor(self, risk_monitor) -> None:
        """Register risk monitor for metrics"""

        self.risk_monitor = risk_monitor
        console.print("   📊 Risk monitor registered")

    def register_circuit_breakers(self, circuit_breakers) -> None:
        """Register circuit breaker system"""

        self.circuit_breakers = circuit_breakers
        console.print("   ⚡ Circuit breakers registered")

    def register_streaming_manager(self, streaming_manager) -> None:
        """Register streaming data manager"""

        self.streaming_manager = streaming_manager
        console.print("   📡 Streaming manager registered")

    def start_dashboard(self) -> None:
        """Start the live dashboard"""

        if self.is_running:
            console.print("⚠️  Dashboard is already running")
            return

        console.print("🚀 [bold green]Starting Live Performance Dashboard[/bold green]")

        self.is_running = True
        self.status = DashboardStatus.STARTING
        self.start_time = datetime.now()

        # Create dashboard layout
        self._create_dashboard_layout()

        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

        # Start live display
        self._start_live_display()

        self.status = DashboardStatus.RUNNING

        console.print("   ✅ Real-time monitoring started")
        console.print("   📊 Performance tracking active")
        console.print("   🎯 Dashboard display ready")

        logger.info("Live performance dashboard started successfully")

    def stop_dashboard(self) -> None:
        """Stop the live dashboard"""

        console.print("⏹️  [bold yellow]Stopping Live Performance Dashboard[/bold yellow]")

        self.is_running = False
        self.status = DashboardStatus.STOPPED

        # Stop live display
        if self.live_display:
            self.live_display.stop()

        # Wait for update thread
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)

        # Save final snapshot
        self._save_performance_snapshot()

        console.print("   ✅ Dashboard stopped")

        logger.info("Live performance dashboard stopped")

    def _create_dashboard_layout(self) -> None:
        """Create the dashboard layout"""

        self.layout = Layout()

        # Split into main sections
        self.layout.split(
            Layout(name="header", size=3), Layout(name="body"), Layout(name="footer", size=5)
        )

        # Split body into columns
        self.layout["body"].split_row(Layout(name="left"), Layout(name="right"))

        # Split left column
        self.layout["left"].split(
            Layout(name="performance", ratio=2),
            Layout(name="positions", ratio=2),
            Layout(name="trading", ratio=1),
        )

        # Split right column
        self.layout["right"].split(
            Layout(name="strategies"), Layout(name="risk"), Layout(name="system")
        )

    def _start_live_display(self) -> None:
        """Start the live Rich display"""

        try:
            self.live_display = Live(self.layout, refresh_per_second=2, screen=True)

            # Run in separate thread
            display_thread = threading.Thread(target=self._run_live_display, daemon=True)
            display_thread.start()

        except Exception as e:
            logger.error(f"Failed to start live display: {str(e)}")
            self.status = DashboardStatus.ERROR

    def _run_live_display(self) -> None:
        """Run the live display loop"""

        try:
            with self.live_display:
                while self.is_running:
                    time.sleep(0.5)  # 2 FPS refresh rate
        except Exception as e:
            logger.error(f"Live display error: {str(e)}")

    def _update_loop(self) -> None:
        """Main dashboard update loop"""

        update_count = 0

        while self.is_running:
            try:
                # Update performance metrics
                self._update_performance_metrics()

                # Update strategy performance
                self._update_strategy_performance()

                # Check for new alerts
                self._check_system_alerts()

                # Update dashboard layout
                self._update_dashboard_display()

                # Save periodic snapshot
                update_count += 1
                if update_count % 60 == 0:  # Every minute
                    self._save_performance_snapshot()

                # Sleep until next update
                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Dashboard update error: {str(e)}")
                time.sleep(5)  # Longer pause on error

    def _update_performance_metrics(self) -> None:
        """Update current performance metrics"""

        try:
            # Initialize new metrics
            new_metrics = PerformanceMetrics()
            new_metrics.timestamp = datetime.now()
            new_metrics.system_uptime = new_metrics.timestamp - self.start_time

            # Aggregate data from trading engines
            total_pnl = Decimal("0")
            unrealized_pnl = Decimal("0")
            realized_pnl = Decimal("0")
            total_positions = 0
            long_positions = 0
            short_positions = 0
            total_trades = 0
            successful_trades = 0
            active_orders = 0
            portfolio_value = Decimal("100000")  # Starting capital

            for _engine_id, engine in self.trading_engines.items():
                if hasattr(engine, "get_trading_status"):
                    status = engine.get_trading_status()

                    # Aggregate trading metrics
                    total_trades += status.get("total_trades", 0)
                    successful_trades += status.get("successful_trades", 0)
                    active_orders += status.get("active_orders", 0)
                    portfolio_value = Decimal(str(status.get("current_capital", 100000)))

                if hasattr(engine, "positions"):
                    for position in engine.positions.values():
                        total_positions += 1
                        if position.quantity > 0:
                            long_positions += 1
                        else:
                            short_positions += 1

                        unrealized_pnl += position.unrealized_pnl
                        realized_pnl += position.realized_pnl

            total_pnl = unrealized_pnl + realized_pnl

            # Update metrics
            new_metrics.total_pnl = total_pnl
            new_metrics.unrealized_pnl = unrealized_pnl
            new_metrics.realized_pnl = realized_pnl
            new_metrics.portfolio_value = portfolio_value
            new_metrics.total_positions = total_positions
            new_metrics.long_positions = long_positions
            new_metrics.short_positions = short_positions
            new_metrics.total_trades_today = total_trades
            new_metrics.successful_trades = successful_trades
            new_metrics.active_orders = active_orders

            # Calculate derived metrics
            if total_trades > 0:
                new_metrics.success_rate = successful_trades / total_trades
                new_metrics.failed_trades = total_trades - successful_trades

            # Get risk metrics if available
            if self.risk_monitor:
                risk_metrics = self.risk_monitor.get_current_risk_metrics()
                if risk_metrics:
                    new_metrics.current_drawdown = risk_metrics.current_drawdown
                    new_metrics.max_drawdown = risk_metrics.max_drawdown
                    new_metrics.portfolio_volatility = risk_metrics.portfolio_volatility_30d
                    new_metrics.sharpe_ratio = risk_metrics.sharpe_ratio_30d
                    new_metrics.portfolio_beta = risk_metrics.portfolio_beta

            # Store current metrics
            self.current_metrics = new_metrics
            self.metrics_history.append(new_metrics)

        except Exception as e:
            logger.error(f"Performance metrics update error: {str(e)}")

    def _update_strategy_performance(self) -> None:
        """Update strategy-specific performance"""

        try:
            current_strategies = set()

            for _engine_id, engine in self.trading_engines.items():
                if hasattr(engine, "positions"):
                    for position in engine.positions.values():
                        strategy_id = position.strategy_id
                        current_strategies.add(strategy_id)

                        if strategy_id not in self.strategy_performance:
                            self.strategy_performance[strategy_id] = StrategyPerformance(
                                strategy_id=strategy_id
                            )

                        strategy_perf = self.strategy_performance[strategy_id]

                        # Update strategy metrics
                        strategy_perf.position_count += 1
                        strategy_perf.total_exposure += abs(position.market_value)
                        strategy_perf.total_pnl += position.unrealized_pnl + position.realized_pnl
                        strategy_perf.last_update = datetime.now()

            # Mark inactive strategies
            for strategy_id in self.strategy_performance:
                if strategy_id not in current_strategies:
                    self.strategy_performance[strategy_id].is_active = False

            # Update active strategy count
            self.current_metrics.active_strategies = len(
                [s for s in self.strategy_performance.values() if s.is_active]
            )

        except Exception as e:
            logger.error(f"Strategy performance update error: {str(e)}")

    def _check_system_alerts(self) -> None:
        """Check for new system alerts"""

        try:
            # Check circuit breaker status
            if self.circuit_breakers:
                breaker_status = self.circuit_breakers.get_breaker_status()

                if breaker_status.get("triggered_breakers", 0) > 0:
                    self._add_alert(
                        severity=AlertSeverity.CRITICAL,
                        title="Circuit Breaker Triggered",
                        message=f"{breaker_status['triggered_breakers']} circuit breakers are currently triggered",
                        component="circuit_breakers",
                    )

            # Check risk metrics for warnings
            if self.risk_monitor:
                risk_metrics = self.risk_monitor.get_current_risk_metrics()
                if risk_metrics:
                    if risk_metrics.current_drawdown > 0.10:  # 10% drawdown warning
                        self._add_alert(
                            severity=AlertSeverity.WARNING,
                            title="High Drawdown Warning",
                            message=f"Current drawdown: {risk_metrics.current_drawdown:.1%}",
                            component="risk_monitor",
                        )

            # Check trading engine health
            for engine_id, engine in self.trading_engines.items():
                if hasattr(engine, "get_trading_status"):
                    status = engine.get_trading_status()

                    if not status.get("is_running", False):
                        self._add_alert(
                            severity=AlertSeverity.WARNING,
                            title="Trading Engine Stopped",
                            message=f"Trading engine {engine_id} is not running",
                            component=engine_id,
                        )

        except Exception as e:
            logger.error(f"System alerts check error: {str(e)}")

    def _add_alert(self, severity: AlertSeverity, title: str, message: str, component: str) -> None:
        """Add a new system alert"""

        alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{component}"

        # Check if similar alert already exists
        for existing_alert in self.active_alerts:
            if (
                existing_alert.title == title
                and existing_alert.component == component
                and not existing_alert.resolved
            ):
                return  # Don't duplicate alerts

        alert = SystemAlert(
            alert_id=alert_id, severity=severity, title=title, message=message, component=component
        )

        self.active_alerts.append(alert)
        self.alert_history.append(alert)

        # Store in database
        self._store_system_alert(alert)

    def _update_dashboard_display(self) -> None:
        """Update the dashboard display layout"""

        if not self.layout:
            return

        try:
            # Update header
            self.layout["header"].update(self._create_header_panel())

            # Update performance panel
            self.layout["performance"].update(self._create_performance_panel())

            # Update positions panel
            self.layout["positions"].update(self._create_positions_panel())

            # Update trading panel
            self.layout["trading"].update(self._create_trading_panel())

            # Update strategies panel
            self.layout["strategies"].update(self._create_strategies_panel())

            # Update risk panel
            self.layout["risk"].update(self._create_risk_panel())

            # Update system panel
            self.layout["system"].update(self._create_system_panel())

            # Update footer
            self.layout["footer"].update(self._create_footer_panel())

        except Exception as e:
            logger.error(f"Dashboard display update error: {str(e)}")

    def _create_header_panel(self) -> Panel:
        """Create header panel"""

        status_color = {
            DashboardStatus.STARTING: "yellow",
            DashboardStatus.RUNNING: "green",
            DashboardStatus.STOPPED: "red",
            DashboardStatus.ERROR: "red",
        }.get(self.status, "white")

        uptime = datetime.now() - self.start_time
        uptime_str = (
            f"{int(uptime.total_seconds()//3600)}h {int((uptime.total_seconds()%3600)//60)}m"
        )

        header_text = "[bold blue]GPT-Trader Live Performance Dashboard[/bold blue]\n"
        header_text += f"Status: [{status_color}]{self.status.value.upper()}[/{status_color}] | "
        header_text += f"Uptime: {uptime_str} | "
        header_text += f"Last Update: {datetime.now().strftime('%H:%M:%S')}"

        return Panel(header_text, title="🚀 Live Trading Dashboard")

    def _create_performance_panel(self) -> Panel:
        """Create performance metrics panel"""

        metrics = self.current_metrics

        # P&L colors
        pnl_color = "green" if metrics.total_pnl >= 0 else "red"
        daily_pnl_color = "green" if metrics.daily_pnl >= 0 else "red"

        performance_text = f"Portfolio Value: [bold]${metrics.portfolio_value:,.2f}[/bold]\n"
        performance_text += f"Total P&L: [{pnl_color}]${metrics.total_pnl:,.2f}[/{pnl_color}]\n"
        performance_text += f"  Unrealized: ${metrics.unrealized_pnl:,.2f}\n"
        performance_text += f"  Realized: ${metrics.realized_pnl:,.2f}\n"
        performance_text += (
            f"Daily P&L: [{daily_pnl_color}]${metrics.daily_pnl:,.2f}[/{daily_pnl_color}]\n"
        )
        performance_text += (
            f"Drawdown: {metrics.current_drawdown:.1%} (Max: {metrics.max_drawdown:.1%})\n"
        )
        performance_text += f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}"

        return Panel(performance_text, title="💰 Performance")

    def _create_positions_panel(self) -> Panel:
        """Create positions panel"""

        metrics = self.current_metrics

        positions_text = f"Total Positions: [bold]{metrics.total_positions}[/bold]\n"
        positions_text += f"  Long: {metrics.long_positions}\n"
        positions_text += f"  Short: {metrics.short_positions}\n"
        positions_text += f"Largest Position: {metrics.largest_position_pct:.1%}\n"
        positions_text += f"Portfolio Beta: {metrics.portfolio_beta:.2f}\n"
        positions_text += f"Volatility (30D): {metrics.portfolio_volatility:.1%}"

        return Panel(positions_text, title="📊 Positions")

    def _create_trading_panel(self) -> Panel:
        """Create trading activity panel"""

        metrics = self.current_metrics

        trading_text = f"Active Orders: [bold]{metrics.active_orders}[/bold]\n"
        trading_text += f"Trades Today: {metrics.total_trades_today}\n"
        trading_text += f"Success Rate: {metrics.success_rate:.1%}\n"
        trading_text += f"Avg Trade P&L: ${metrics.avg_trade_pnl:,.2f}"

        return Panel(trading_text, title="📈 Trading")

    def _create_strategies_panel(self) -> Panel:
        """Create strategies panel"""

        if not self.strategy_performance:
            return Panel("No active strategies", title="🎯 Strategies")

        strategies_table = Table.grid()
        strategies_table.add_column(style="cyan")
        strategies_table.add_column(justify="right")
        strategies_table.add_column(justify="right")

        strategies_table.add_row("Strategy", "P&L", "Pos")
        strategies_table.add_row("─────────", "──────", "───")

        # Show top 5 strategies by P&L
        top_strategies = sorted(
            self.strategy_performance.values(), key=lambda x: x.total_pnl, reverse=True
        )[:5]

        for strategy in top_strategies:
            pnl_color = "green" if strategy.total_pnl >= 0 else "red"
            short_id = (
                strategy.strategy_id[-12:]
                if len(strategy.strategy_id) > 12
                else strategy.strategy_id
            )

            strategies_table.add_row(
                short_id,
                f"[{pnl_color}]${strategy.total_pnl:,.0f}[/{pnl_color}]",
                str(strategy.position_count),
            )

        return Panel(strategies_table, title="🎯 Strategies")

    def _create_risk_panel(self) -> Panel:
        """Create risk monitoring panel"""

        risk_text = ""

        if self.risk_monitor:
            risk_metrics = self.risk_monitor.get_current_risk_metrics()
            if risk_metrics:
                risk_text += f"Portfolio VaR (95%): ${risk_metrics.portfolio_var_95:,.0f}\n"
                risk_text += f"Expected Shortfall: ${risk_metrics.portfolio_cvar_95:,.0f}\n"
                risk_text += f"Correlation Risk: {risk_metrics.avg_correlation:.2f}\n"

        # Circuit breaker status
        if self.circuit_breakers:
            breaker_status = self.circuit_breakers.get_breaker_status()
            active = breaker_status.get("active_breakers", 0)
            triggered = breaker_status.get("triggered_breakers", 0)

            breaker_color = "red" if triggered > 0 else "green"
            risk_text += f"Circuit Breakers: [{breaker_color}]{active} Active, {triggered} Triggered[/{breaker_color}]"

        if not risk_text:
            risk_text = "Risk monitoring not available"

        return Panel(risk_text, title="⚡ Risk")

    def _create_system_panel(self) -> Panel:
        """Create system status panel"""

        system_text = f"Trading Engines: {len(self.trading_engines)}\n"

        # Data feed status
        if self.streaming_manager:
            stats = self.streaming_manager.get_streaming_statistics()
            connected_feeds = sum(
                1 for s in stats.values() if s.connection_status.value == "connected"
            )
            system_text += f"Data Feeds: {connected_feeds}/{len(stats)}\n"

        # Alert status
        critical_alerts = sum(
            1 for a in self.active_alerts if a.severity == AlertSeverity.CRITICAL and not a.resolved
        )
        warning_alerts = sum(
            1 for a in self.active_alerts if a.severity == AlertSeverity.WARNING and not a.resolved
        )

        alert_color = "red" if critical_alerts > 0 else "yellow" if warning_alerts > 0 else "green"
        system_text += f"Alerts: [{alert_color}]{critical_alerts} Critical, {warning_alerts} Warning[/{alert_color}]"

        return Panel(system_text, title="🖥️ System")

    def _create_footer_panel(self) -> Panel:
        """Create footer panel with recent alerts"""

        if not self.active_alerts:
            return Panel("No active alerts", title="🚨 Recent Alerts")

        alerts_table = Table.grid()
        alerts_table.add_column(style="dim")
        alerts_table.add_column()
        alerts_table.add_column(style="dim")

        # Show last 3 alerts
        recent_alerts = list(self.active_alerts)[-3:]

        for alert in recent_alerts:
            elapsed = datetime.now() - alert.timestamp
            elapsed_str = (
                f"{int(elapsed.total_seconds())}s ago"
                if elapsed.total_seconds() < 60
                else f"{int(elapsed.total_seconds()/60)}m ago"
            )

            severity_color = {
                AlertSeverity.CRITICAL: "red",
                AlertSeverity.WARNING: "yellow",
                AlertSeverity.INFO: "blue",
            }.get(alert.severity, "white")

            alerts_table.add_row(
                elapsed_str,
                f"[{severity_color}]{alert.title}[/{severity_color}]: {alert.message[:50]}{'...' if len(alert.message) > 50 else ''}",
                alert.component,
            )

        return Panel(alerts_table, title="🚨 Recent Alerts")

    def _save_performance_snapshot(self) -> None:
        """Save current performance snapshot to database"""

        try:
            metrics = self.current_metrics

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO performance_snapshots (
                        timestamp, total_pnl, unrealized_pnl, realized_pnl, daily_pnl,
                        total_capital, portfolio_value, total_positions, total_trades,
                        success_rate, current_drawdown, max_drawdown, sharpe_ratio,
                        metrics_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        metrics.timestamp.isoformat(),
                        str(metrics.total_pnl),
                        str(metrics.unrealized_pnl),
                        str(metrics.realized_pnl),
                        str(metrics.daily_pnl),
                        str(metrics.total_capital),
                        str(metrics.portfolio_value),
                        metrics.total_positions,
                        metrics.total_trades_today,
                        metrics.success_rate,
                        metrics.current_drawdown,
                        metrics.max_drawdown,
                        metrics.sharpe_ratio,
                        json.dumps(
                            {
                                "long_positions": metrics.long_positions,
                                "short_positions": metrics.short_positions,
                                "active_orders": metrics.active_orders,
                                "active_strategies": metrics.active_strategies,
                                "portfolio_beta": metrics.portfolio_beta,
                                "portfolio_volatility": metrics.portfolio_volatility,
                            }
                        ),
                    ),
                )
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to save performance snapshot: {str(e)}")

    def _store_system_alert(self, alert: SystemAlert) -> None:
        """Store system alert in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO system_alerts (
                    alert_id, severity, title, message, component, timestamp,
                    acknowledged, resolved, alert_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    alert.alert_id,
                    alert.severity.value,
                    alert.title,
                    alert.message,
                    alert.component,
                    alert.timestamp.isoformat(),
                    alert.acknowledged,
                    alert.resolved,
                    json.dumps({}),
                ),
            )
            conn.commit()

    def get_dashboard_status(self) -> dict[str, Any]:
        """Get current dashboard status"""

        return {
            "status": self.status.value,
            "uptime_seconds": int((datetime.now() - self.start_time).total_seconds()),
            "registered_engines": len(self.trading_engines),
            "has_risk_monitor": self.risk_monitor is not None,
            "has_circuit_breakers": self.circuit_breakers is not None,
            "has_streaming_manager": self.streaming_manager is not None,
            "active_alerts": len(self.active_alerts),
            "total_alerts": len(self.alert_history),
            "metrics_history_size": len(self.metrics_history),
            "current_performance": {
                "total_pnl": float(self.current_metrics.total_pnl),
                "portfolio_value": float(self.current_metrics.portfolio_value),
                "total_positions": self.current_metrics.total_positions,
                "success_rate": self.current_metrics.success_rate,
            },
        }


def create_live_dashboard(
    dashboard_dir: str = "data/dashboard", update_interval: float = 1.0, max_alerts: int = 100
) -> LivePerformanceDashboard:
    """Factory function to create live dashboard"""
    return LivePerformanceDashboard(
        dashboard_dir=dashboard_dir, update_interval=update_interval, max_alerts=max_alerts
    )


if __name__ == "__main__":
    # Example usage
    dashboard = create_live_dashboard()

    try:
        # Start dashboard
        dashboard.start_dashboard()

        # Let it run for demo
        time.sleep(30)

        # Display status
        status = dashboard.get_dashboard_status()
        console.print(f"\nDashboard Status: {status}")

    finally:
        dashboard.stop_dashboard()

    print("Live Performance Dashboard created successfully!")
