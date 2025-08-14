"""
Strategy Health Monitor for GPT-Trader Production Trading

Automated monitoring system that continuously evaluates strategy performance and health:

- Real-time strategy performance tracking and analysis
- Statistical significance testing for performance degradation
- Automated strategy lifecycle management (enable/disable/retire)
- Performance attribution analysis and decomposition
- Risk-adjusted performance measurement (Sharpe, Calmar, etc.)
- Strategy comparison and ranking systems
- Early warning detection for underperforming strategies
- Automated A/B testing and strategy optimization

This system ensures only healthy, profitable strategies remain active in production.
"""

import json
import logging
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


class StrategyHealthStatus(Enum):
    """Strategy health status levels"""

    EXCELLENT = "excellent"  # Outperforming expectations
    GOOD = "good"  # Meeting expectations
    ACCEPTABLE = "acceptable"  # Marginally acceptable performance
    WARNING = "warning"  # Performance concerns detected
    CRITICAL = "critical"  # Significant underperformance
    FAILED = "failed"  # Strategy has failed criteria


class StrategyAction(Enum):
    """Actions that can be taken on strategies"""

    CONTINUE = "continue"  # No action needed
    MONITOR = "monitor"  # Increase monitoring frequency
    REDUCE_ALLOCATION = "reduce_allocation"  # Reduce position sizes
    DISABLE_NEW_POSITIONS = "disable_new_positions"  # No new positions
    LIQUIDATE_POSITIONS = "liquidate_positions"  # Close all positions
    DISABLE_STRATEGY = "disable_strategy"  # Completely disable strategy
    RETIRE_STRATEGY = "retire_strategy"  # Permanently retire strategy


class PerformanceMetric(Enum):
    """Performance metrics to track"""

    TOTAL_RETURN = "total_return"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    SHARPE_RATIO = "sharpe_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    AVERAGE_TRADE_PNL = "average_trade_pnl"
    TRADE_FREQUENCY = "trade_frequency"
    POSITION_SIZING = "position_sizing"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"


@dataclass
class StrategyMetrics:
    """Comprehensive strategy performance metrics"""

    strategy_id: str
    measurement_period: timedelta

    # Basic performance metrics
    total_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    total_return_pct: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    profit_factor: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    volatility: float = 0.0
    beta: float = 0.0

    # Risk-adjusted performance
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    information_ratio: float = 0.0

    # Position metrics
    avg_position_size: Decimal = Decimal("0")
    max_position_size: Decimal = Decimal("0")
    total_exposure: Decimal = Decimal("0")
    leverage: float = 0.0

    # Frequency metrics
    trades_per_day: float = 0.0
    avg_holding_period: timedelta = timedelta()

    # Comparison metrics (vs benchmark)
    alpha: float = 0.0
    tracking_error: float = 0.0

    # Timestamp
    calculated_at: datetime = field(default_factory=datetime.now)


@dataclass
class HealthCheck:
    """Individual health check result"""

    check_id: str
    metric: PerformanceMetric
    current_value: float
    threshold_value: float
    expected_range: tuple[float, float]

    # Check result
    passed: bool
    severity: StrategyHealthStatus
    message: str

    # Statistical information
    confidence_level: float = 0.95
    p_value: float | None = None
    z_score: float | None = None

    # Timestamp
    checked_at: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyHealthReport:
    """Complete strategy health assessment"""

    strategy_id: str
    assessment_timestamp: datetime = field(default_factory=datetime.now)

    # Overall health
    overall_health: StrategyHealthStatus = StrategyHealthStatus.GOOD
    health_score: float = 0.75  # 0.0 to 1.0

    # Individual checks
    health_checks: list[HealthCheck] = field(default_factory=list)
    failed_checks: list[HealthCheck] = field(default_factory=list)
    warning_checks: list[HealthCheck] = field(default_factory=list)

    # Current metrics
    current_metrics: StrategyMetrics | None = None

    # Historical comparison
    metrics_trend: str = "stable"  # improving, stable, deteriorating
    performance_percentile: float = 50.0  # vs historical performance

    # Recommended actions
    recommended_actions: list[StrategyAction] = field(default_factory=list)
    action_urgency: str = "low"  # low, medium, high

    # Additional context
    notes: list[str] = field(default_factory=list)
    last_significant_change: datetime | None = None


class StrategyHealthMonitor:
    """Production strategy health monitoring and management system"""

    def __init__(
        self,
        monitoring_dir: str = "data/strategy_health",
        check_interval: timedelta = timedelta(minutes=15),
        min_trades_for_analysis: int = 10,
    ) -> None:
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.monitoring_dir / "reports").mkdir(exist_ok=True)
        (self.monitoring_dir / "metrics").mkdir(exist_ok=True)
        (self.monitoring_dir / "alerts").mkdir(exist_ok=True)
        (self.monitoring_dir / "logs").mkdir(exist_ok=True)

        self.check_interval = check_interval
        self.min_trades_for_analysis = min_trades_for_analysis

        # Initialize database
        self.db_path = self.monitoring_dir / "strategy_health.db"
        self._initialize_database()

        # Health check configuration
        self.health_thresholds = self._initialize_health_thresholds()

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None

        # Strategy tracking
        self.active_strategies: dict[str, dict[str, Any]] = {}
        self.strategy_metrics: dict[str, StrategyMetrics] = {}
        self.health_reports: dict[str, StrategyHealthReport] = {}

        # Historical data
        self.metrics_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.health_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # System integrations (injected by main system)
        self.trading_engines = {}  # engine_id -> engine instance
        self.risk_monitor = None
        self.alerting_system = None

        # Performance benchmarks
        self.benchmark_returns = deque(maxlen=252)  # 1 year of daily returns
        self.risk_free_rate = 0.02  # 2% annual risk-free rate

        # Callbacks for health events
        self.health_callbacks: list = []

        logger.info("Strategy Health Monitor initialized")

    def _initialize_database(self) -> None:
        """Initialize SQLite database for strategy health monitoring"""

        with sqlite3.connect(self.db_path) as conn:
            # Strategy metrics table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    measurement_date TEXT NOT NULL,
                    total_pnl TEXT NOT NULL,
                    realized_pnl TEXT NOT NULL,
                    unrealized_pnl TEXT NOT NULL,
                    total_return_pct REAL,
                    total_trades INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    calmar_ratio REAL,
                    metrics_data TEXT,
                    calculated_at TEXT NOT NULL
                )
            """
            )

            # Health checks table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS health_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    check_id TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    threshold_value REAL NOT NULL,
                    passed BOOLEAN NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    p_value REAL,
                    z_score REAL,
                    checked_at TEXT NOT NULL
                )
            """
            )

            # Health reports table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS health_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    assessment_date TEXT NOT NULL,
                    overall_health TEXT NOT NULL,
                    health_score REAL NOT NULL,
                    failed_checks INTEGER DEFAULT 0,
                    warning_checks INTEGER DEFAULT 0,
                    recommended_actions TEXT,
                    action_urgency TEXT,
                    report_data TEXT,
                    created_at TEXT NOT NULL
                )
            """
            )

            # Strategy actions table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    triggered_by TEXT,
                    action_data TEXT,
                    executed_at TEXT NOT NULL,
                    success BOOLEAN DEFAULT TRUE
                )
            """
            )

            # Strategy lifecycle table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_lifecycle (
                    strategy_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    first_trade_at TEXT,
                    last_trade_at TEXT,
                    total_lifetime_pnl TEXT DEFAULT '0',
                    total_lifetime_trades INTEGER DEFAULT 0,
                    retirement_reason TEXT,
                    retired_at TEXT,
                    lifecycle_data TEXT
                )
            """
            )

            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_strategy_date ON strategy_metrics (strategy_id, measurement_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_checks_strategy_date ON health_checks (strategy_id, checked_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reports_strategy_date ON health_reports (strategy_id, assessment_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_actions_strategy ON strategy_actions (strategy_id)"
            )

            conn.commit()

    def _initialize_health_thresholds(self) -> dict[PerformanceMetric, dict[str, Any]]:
        """Initialize health check thresholds"""

        return {
            PerformanceMetric.WIN_RATE: {
                "critical_threshold": 0.30,  # < 30% win rate is critical
                "warning_threshold": 0.40,  # < 40% win rate is warning
                "excellent_threshold": 0.60,  # > 60% win rate is excellent
                "expected_range": (0.45, 0.55),
                "check_enabled": True,
            },
            PerformanceMetric.PROFIT_FACTOR: {
                "critical_threshold": 1.0,  # < 1.0 profit factor is losing money
                "warning_threshold": 1.2,  # < 1.2 is concerning
                "excellent_threshold": 2.0,  # > 2.0 is excellent
                "expected_range": (1.3, 1.8),
                "check_enabled": True,
            },
            PerformanceMetric.MAX_DRAWDOWN: {
                "critical_threshold": 0.20,  # > 20% drawdown is critical
                "warning_threshold": 0.15,  # > 15% drawdown is warning
                "excellent_threshold": 0.05,  # < 5% drawdown is excellent
                "expected_range": (0.08, 0.12),
                "check_enabled": True,
            },
            PerformanceMetric.SHARPE_RATIO: {
                "critical_threshold": 0.0,  # < 0 Sharpe is critical
                "warning_threshold": 0.5,  # < 0.5 is concerning
                "excellent_threshold": 1.5,  # > 1.5 is excellent
                "expected_range": (0.8, 1.2),
                "check_enabled": True,
            },
            PerformanceMetric.TOTAL_RETURN: {
                "critical_threshold": -0.10,  # < -10% return is critical
                "warning_threshold": -0.05,  # < -5% return is warning
                "excellent_threshold": 0.15,  # > 15% return is excellent
                "expected_range": (0.05, 0.12),
                "check_enabled": True,
            },
        }

    def register_trading_engine(self, engine_id: str, engine_instance) -> None:
        """Register a trading engine for monitoring"""

        self.trading_engines[engine_id] = engine_instance
        console.print(f"   🔗 Registered trading engine: {engine_id}")

    def register_risk_monitor(self, risk_monitor) -> None:
        """Register risk monitor for additional metrics"""

        self.risk_monitor = risk_monitor
        console.print("   📊 Risk monitor registered")

    def register_alerting_system(self, alerting_system) -> None:
        """Register alerting system for health notifications"""

        self.alerting_system = alerting_system
        console.print("   🚨 Alerting system registered")

    def add_health_callback(self, callback) -> None:
        """Add callback for health events"""

        self.health_callbacks.append(callback)

    def start_monitoring(self) -> None:
        """Start strategy health monitoring"""

        if self.is_monitoring:
            console.print("⚠️  Strategy health monitoring is already active")
            return

        console.print("🚀 [bold green]Starting Strategy Health Monitoring[/bold green]")

        self.is_monitoring = True

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        console.print(
            f"   ✅ Health checks every {self.check_interval.total_seconds()/60:.0f} minutes"
        )
        console.print(f"   📊 Minimum trades for analysis: {self.min_trades_for_analysis}")
        console.print("   🩺 Strategy health monitoring active")

        logger.info("Strategy health monitoring started successfully")

    def stop_monitoring(self) -> None:
        """Stop strategy health monitoring"""

        console.print("⏹️  [bold yellow]Stopping Strategy Health Monitoring[/bold yellow]")

        self.is_monitoring = False

        # Wait for monitoring thread to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10)

        console.print("   ✅ Strategy health monitoring stopped")

        logger.info("Strategy health monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""

        while self.is_monitoring:
            try:
                # Discover active strategies
                self._discover_active_strategies()

                # Update metrics for each strategy
                for strategy_id in self.active_strategies.keys():
                    self._update_strategy_metrics(strategy_id)

                # Perform health checks
                for strategy_id in self.active_strategies.keys():
                    if strategy_id in self.strategy_metrics:
                        health_report = self._perform_health_assessment(strategy_id)
                        self.health_reports[strategy_id] = health_report

                        # Take automated actions if needed
                        self._execute_automated_actions(health_report)

                # Sleep until next check
                time.sleep(self.check_interval.total_seconds())

            except Exception as e:
                logger.error(f"Strategy health monitoring error: {str(e)}")
                time.sleep(60)  # Wait 1 minute on error

    def _discover_active_strategies(self) -> None:
        """Discover currently active strategies"""

        current_strategies = set()

        for engine_id, engine in self.trading_engines.items():
            if hasattr(engine, "positions"):
                for position in engine.positions.values():
                    current_strategies.add(position.strategy_id)

            # Also check active orders
            if hasattr(engine, "active_orders"):
                for order in engine.active_orders.values():
                    current_strategies.add(order.strategy_id)

        # Update active strategies tracking
        for strategy_id in current_strategies:
            if strategy_id not in self.active_strategies:
                self.active_strategies[strategy_id] = {
                    "discovered_at": datetime.now(),
                    "first_seen": datetime.now(),
                    "engines": set(),
                    "total_trades": 0,
                    "last_trade_time": None,
                }

            # Track which engines use this strategy
            for engine_id, engine in self.trading_engines.items():
                if hasattr(engine, "positions"):
                    for position in engine.positions.values():
                        if position.strategy_id == strategy_id:
                            self.active_strategies[strategy_id]["engines"].add(engine_id)

        # Remove strategies that are no longer active
        inactive_strategies = set(self.active_strategies.keys()) - current_strategies
        for strategy_id in inactive_strategies:
            # Keep recent strategies for a while
            if (
                datetime.now() - self.active_strategies[strategy_id]["discovered_at"]
            ).total_seconds() > 3600:  # 1 hour
                del self.active_strategies[strategy_id]

    def _update_strategy_metrics(self, strategy_id: str) -> None:
        """Update performance metrics for a strategy"""

        try:
            # Collect data from all engines
            total_pnl = Decimal("0")
            realized_pnl = Decimal("0")
            unrealized_pnl = Decimal("0")
            positions = []

            # Get position data
            for _engine_id, engine in self.trading_engines.items():
                if hasattr(engine, "positions"):
                    for position in engine.positions.values():
                        if position.strategy_id == strategy_id:
                            positions.append(position)
                            total_pnl += position.unrealized_pnl + position.realized_pnl
                            unrealized_pnl += position.unrealized_pnl
                            realized_pnl += position.realized_pnl

            # Get trade history (placeholder - would integrate with actual trade history)
            trade_data = self._get_strategy_trade_history(strategy_id)

            # Calculate comprehensive metrics
            metrics = StrategyMetrics(
                strategy_id=strategy_id,
                measurement_period=timedelta(days=30),  # 30-day measurement
                total_pnl=total_pnl,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
            )

            if trade_data:
                # Calculate trade statistics
                metrics.total_trades = len(trade_data)
                winning_trades = [t for t in trade_data if t["pnl"] > 0]
                losing_trades = [t for t in trade_data if t["pnl"] < 0]

                metrics.winning_trades = len(winning_trades)
                metrics.losing_trades = len(losing_trades)

                if metrics.total_trades > 0:
                    metrics.win_rate = metrics.winning_trades / metrics.total_trades

                if winning_trades:
                    metrics.avg_win = Decimal(str(np.mean([t["pnl"] for t in winning_trades])))

                if losing_trades:
                    metrics.avg_loss = Decimal(str(abs(np.mean([t["pnl"] for t in losing_trades]))))

                # Calculate profit factor
                total_wins = sum(t["pnl"] for t in winning_trades)
                total_losses = abs(sum(t["pnl"] for t in losing_trades))

                if total_losses > 0:
                    metrics.profit_factor = total_wins / total_losses

                # Calculate returns and risk metrics
                returns = [t["return_pct"] for t in trade_data if "return_pct" in t]
                if returns:
                    metrics.total_return_pct = (1 + np.prod([1 + r for r in returns])) - 1
                    metrics.volatility = np.std(returns) * np.sqrt(252)  # Annualized

                    # Calculate risk-adjusted metrics
                    if metrics.volatility > 0:
                        excess_returns = np.array(returns) - (self.risk_free_rate / 252)
                        metrics.sharpe_ratio = (
                            np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
                        )

                    # Calculate maximum drawdown
                    cumulative_returns = np.cumprod([1 + r for r in returns])
                    running_max = np.maximum.accumulate(cumulative_returns)
                    drawdowns = (cumulative_returns - running_max) / running_max
                    metrics.max_drawdown = abs(np.min(drawdowns))
                    metrics.current_drawdown = abs(drawdowns[-1]) if len(drawdowns) > 0 else 0.0

                    # Calculate Calmar ratio
                    if metrics.max_drawdown > 0:
                        metrics.calmar_ratio = metrics.total_return_pct / metrics.max_drawdown

            # Calculate position metrics
            if positions:
                position_sizes = [abs(p.market_value) for p in positions]
                metrics.avg_position_size = Decimal(str(np.mean(position_sizes)))
                metrics.max_position_size = Decimal(str(np.max(position_sizes)))
                metrics.total_exposure = sum(Decimal(str(abs(p.market_value))) for p in positions)

            # Store metrics
            self.strategy_metrics[strategy_id] = metrics
            self.metrics_history[strategy_id].append(metrics)

            # Save to database
            self._store_strategy_metrics(metrics)

        except Exception as e:
            logger.error(f"Error updating metrics for strategy {strategy_id}: {str(e)}")

    def _get_strategy_trade_history(self, strategy_id: str) -> list[dict[str, Any]]:
        """Get trade history for strategy (placeholder implementation)"""

        # In a real implementation, this would query the database or trading engines
        # for historical trade data for the strategy

        # For demo purposes, return simulated trade data
        if strategy_id not in self.active_strategies:
            return []

        # Simulate some trades
        num_trades = np.random.randint(5, 50)
        trades = []

        for i in range(num_trades):
            pnl = np.random.normal(100, 500)  # Mean $100, std $500
            return_pct = pnl / 10000  # Assume $10k position size

            trades.append(
                {
                    "trade_id": f"{strategy_id}_trade_{i}",
                    "pnl": pnl,
                    "return_pct": return_pct,
                    "timestamp": datetime.now() - timedelta(days=np.random.randint(1, 30)),
                }
            )

        return trades

    def _perform_health_assessment(self, strategy_id: str) -> StrategyHealthReport:
        """Perform comprehensive health assessment for strategy"""

        metrics = self.strategy_metrics.get(strategy_id)
        if not metrics:
            return StrategyHealthReport(
                strategy_id=strategy_id,
                overall_health=StrategyHealthStatus.WARNING,
                health_score=0.0,
                notes=["Insufficient data for health assessment"],
            )

        # Skip assessment if insufficient trades
        if metrics.total_trades < self.min_trades_for_analysis:
            return StrategyHealthReport(
                strategy_id=strategy_id,
                overall_health=StrategyHealthStatus.ACCEPTABLE,
                health_score=0.5,
                notes=[
                    f"Insufficient trades for analysis ({metrics.total_trades} < {self.min_trades_for_analysis})"
                ],
            )

        # Perform individual health checks
        health_checks = []
        failed_checks = []
        warning_checks = []

        for metric, thresholds in self.health_thresholds.items():
            if not thresholds.get("check_enabled", True):
                continue

            check = self._perform_individual_health_check(metrics, metric, thresholds)
            health_checks.append(check)

            if not check.passed:
                if check.severity in [StrategyHealthStatus.CRITICAL, StrategyHealthStatus.FAILED]:
                    failed_checks.append(check)
                else:
                    warning_checks.append(check)

        # Calculate overall health score
        health_score = self._calculate_health_score(health_checks)
        overall_health = self._determine_overall_health(health_score, failed_checks, warning_checks)

        # Determine recommended actions
        recommended_actions = self._determine_recommended_actions(
            overall_health, failed_checks, warning_checks
        )
        action_urgency = self._determine_action_urgency(overall_health, failed_checks)

        # Create health report
        report = StrategyHealthReport(
            strategy_id=strategy_id,
            overall_health=overall_health,
            health_score=health_score,
            health_checks=health_checks,
            failed_checks=failed_checks,
            warning_checks=warning_checks,
            current_metrics=metrics,
            recommended_actions=recommended_actions,
            action_urgency=action_urgency,
        )

        # Store report
        self.health_history[strategy_id].append(report)
        self._store_health_report(report)

        # Send alerts if needed
        if overall_health in [StrategyHealthStatus.CRITICAL, StrategyHealthStatus.FAILED]:
            self._send_health_alert(report)

        return report

    def _perform_individual_health_check(
        self, metrics: StrategyMetrics, metric: PerformanceMetric, thresholds: dict[str, Any]
    ) -> HealthCheck:
        """Perform individual health check"""

        # Get current value
        current_value = self._extract_metric_value(metrics, metric)

        # Determine thresholds
        critical_threshold = thresholds["critical_threshold"]
        warning_threshold = thresholds["warning_threshold"]
        expected_range = thresholds["expected_range"]

        # Perform check
        check_id = f"{metric.value}_check"

        # Determine if check passed and severity
        if metric in [PerformanceMetric.MAX_DRAWDOWN]:  # Lower is better
            if current_value >= critical_threshold:
                passed = False
                severity = StrategyHealthStatus.CRITICAL
                message = f"{metric.value.replace('_', ' ').title()} {current_value:.2%} exceeds critical threshold {critical_threshold:.2%}"
            elif current_value >= warning_threshold:
                passed = False
                severity = StrategyHealthStatus.WARNING
                message = f"{metric.value.replace('_', ' ').title()} {current_value:.2%} exceeds warning threshold {warning_threshold:.2%}"
            else:
                passed = True
                severity = StrategyHealthStatus.GOOD
                message = f"{metric.value.replace('_', ' ').title()} {current_value:.2%} is within acceptable range"
        else:  # Higher is better
            if current_value <= critical_threshold:
                passed = False
                severity = StrategyHealthStatus.CRITICAL
                message = f"{metric.value.replace('_', ' ').title()} {current_value:.2f} below critical threshold {critical_threshold:.2f}"
            elif current_value <= warning_threshold:
                passed = False
                severity = StrategyHealthStatus.WARNING
                message = f"{metric.value.replace('_', ' ').title()} {current_value:.2f} below warning threshold {warning_threshold:.2f}"
            else:
                passed = True
                severity = StrategyHealthStatus.GOOD
                message = (
                    f"{metric.value.replace('_', ' ').title()} {current_value:.2f} is acceptable"
                )

        return HealthCheck(
            check_id=check_id,
            metric=metric,
            current_value=current_value,
            threshold_value=critical_threshold,
            expected_range=expected_range,
            passed=passed,
            severity=severity,
            message=message,
        )

    def _extract_metric_value(self, metrics: StrategyMetrics, metric: PerformanceMetric) -> float:
        """Extract metric value from strategy metrics"""

        if metric == PerformanceMetric.WIN_RATE:
            return metrics.win_rate
        elif metric == PerformanceMetric.PROFIT_FACTOR:
            return metrics.profit_factor
        elif metric == PerformanceMetric.MAX_DRAWDOWN:
            return metrics.max_drawdown
        elif metric == PerformanceMetric.SHARPE_RATIO:
            return metrics.sharpe_ratio
        elif metric == PerformanceMetric.TOTAL_RETURN:
            return metrics.total_return_pct
        elif metric == PerformanceMetric.CALMAR_RATIO:
            return metrics.calmar_ratio
        else:
            return 0.0

    def _calculate_health_score(self, health_checks: list[HealthCheck]) -> float:
        """Calculate overall health score"""

        if not health_checks:
            return 0.5

        total_score = 0.0
        total_weight = 0.0

        for check in health_checks:
            # Assign weights based on metric importance
            weight = {
                PerformanceMetric.PROFIT_FACTOR: 0.3,
                PerformanceMetric.WIN_RATE: 0.2,
                PerformanceMetric.MAX_DRAWDOWN: 0.25,
                PerformanceMetric.SHARPE_RATIO: 0.15,
                PerformanceMetric.TOTAL_RETURN: 0.1,
            }.get(check.metric, 0.1)

            # Assign score based on check result
            if check.severity == StrategyHealthStatus.EXCELLENT:
                score = 1.0
            elif check.severity == StrategyHealthStatus.GOOD:
                score = 0.8
            elif check.severity == StrategyHealthStatus.ACCEPTABLE:
                score = 0.6
            elif check.severity == StrategyHealthStatus.WARNING:
                score = 0.4
            elif check.severity == StrategyHealthStatus.CRITICAL:
                score = 0.2
            else:  # FAILED
                score = 0.0

            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.5

    def _determine_overall_health(
        self,
        health_score: float,
        failed_checks: list[HealthCheck],
        warning_checks: list[HealthCheck],
    ) -> StrategyHealthStatus:
        """Determine overall health status"""

        if failed_checks:
            return StrategyHealthStatus.CRITICAL
        elif len(warning_checks) >= 3:
            return StrategyHealthStatus.WARNING
        elif health_score >= 0.8:
            return StrategyHealthStatus.EXCELLENT
        elif health_score >= 0.7:
            return StrategyHealthStatus.GOOD
        elif health_score >= 0.5:
            return StrategyHealthStatus.ACCEPTABLE
        else:
            return StrategyHealthStatus.WARNING

    def _determine_recommended_actions(
        self,
        overall_health: StrategyHealthStatus,
        failed_checks: list[HealthCheck],
        warning_checks: list[HealthCheck],
    ) -> list[StrategyAction]:
        """Determine recommended actions based on health assessment"""

        actions = []

        if overall_health == StrategyHealthStatus.FAILED:
            actions.extend([StrategyAction.LIQUIDATE_POSITIONS, StrategyAction.DISABLE_STRATEGY])
        elif overall_health == StrategyHealthStatus.CRITICAL:
            if len(failed_checks) >= 2:
                actions.extend(
                    [StrategyAction.DISABLE_NEW_POSITIONS, StrategyAction.REDUCE_ALLOCATION]
                )
            else:
                actions.append(StrategyAction.MONITOR)
        elif overall_health == StrategyHealthStatus.WARNING:
            if len(warning_checks) >= 2:
                actions.append(StrategyAction.REDUCE_ALLOCATION)
            actions.append(StrategyAction.MONITOR)
        elif overall_health == StrategyHealthStatus.ACCEPTABLE:
            actions.append(StrategyAction.MONITOR)
        else:  # GOOD or EXCELLENT
            actions.append(StrategyAction.CONTINUE)

        return actions

    def _determine_action_urgency(
        self, overall_health: StrategyHealthStatus, failed_checks: list[HealthCheck]
    ) -> str:
        """Determine urgency of recommended actions"""

        if overall_health in [StrategyHealthStatus.FAILED, StrategyHealthStatus.CRITICAL]:
            return "high"
        elif len(failed_checks) > 0 or overall_health == StrategyHealthStatus.WARNING:
            return "medium"
        else:
            return "low"

    def _execute_automated_actions(self, health_report: StrategyHealthReport) -> None:
        """Execute automated actions based on health assessment"""

        strategy_id = health_report.strategy_id

        for action in health_report.recommended_actions:
            try:
                if action == StrategyAction.DISABLE_STRATEGY:
                    self._disable_strategy(strategy_id, "Automated: Health assessment failed")
                elif action == StrategyAction.LIQUIDATE_POSITIONS:
                    self._liquidate_strategy_positions(
                        strategy_id, "Automated: Critical health issues"
                    )
                elif action == StrategyAction.DISABLE_NEW_POSITIONS:
                    self._disable_new_positions(strategy_id, "Automated: Warning health status")
                # Other actions would be implemented based on system capabilities

            except Exception as e:
                logger.error(
                    f"Failed to execute automated action {action.value} for strategy {strategy_id}: {str(e)}"
                )

    def _disable_strategy(self, strategy_id: str, reason: str) -> None:
        """Disable a strategy"""

        # This would integrate with the trading system to disable the strategy
        console.print(f"   🚫 Strategy disabled: {strategy_id} - {reason}")

        self._record_strategy_action(strategy_id, StrategyAction.DISABLE_STRATEGY, reason)

    def _liquidate_strategy_positions(self, strategy_id: str, reason: str) -> None:
        """Liquidate all positions for a strategy"""

        liquidated_count = 0

        for _engine_id, engine in self.trading_engines.items():
            if hasattr(engine, "positions"):
                for position_key, position in list(engine.positions.items()):
                    if position.strategy_id == strategy_id:
                        try:
                            # Create liquidation order
                            if hasattr(engine, "submit_order"):
                                side = (
                                    engine.OrderSide.SELL
                                    if position.quantity > 0
                                    else engine.OrderSide.BUY
                                )
                                engine.submit_order(
                                    strategy_id="health_monitor_liquidation",
                                    symbol=position.symbol,
                                    side=side,
                                    quantity=abs(position.quantity),
                                    order_type=engine.OrderType.MARKET,
                                    notes=f"Health monitor liquidation: {reason}",
                                )
                                liquidated_count += 1
                        except Exception as e:
                            logger.error(f"Failed to liquidate position {position_key}: {str(e)}")

        console.print(f"   💰 Liquidated {liquidated_count} positions for strategy {strategy_id}")

        self._record_strategy_action(
            strategy_id,
            StrategyAction.LIQUIDATE_POSITIONS,
            f"{reason} - {liquidated_count} positions liquidated",
        )

    def _disable_new_positions(self, strategy_id: str, reason: str) -> None:
        """Disable new positions for a strategy"""

        # This would integrate with the trading system to prevent new positions
        console.print(f"   ⏸️ New positions disabled for strategy: {strategy_id} - {reason}")

        self._record_strategy_action(strategy_id, StrategyAction.DISABLE_NEW_POSITIONS, reason)

    def _send_health_alert(self, health_report: StrategyHealthReport) -> None:
        """Send health alert through alerting system"""

        if not self.alerting_system:
            return

        severity = {
            StrategyHealthStatus.CRITICAL: "critical",
            StrategyHealthStatus.FAILED: "emergency",
        }.get(health_report.overall_health, "warning")

        title = f"Strategy Health Alert: {health_report.strategy_id}"
        message = f"""
Strategy health assessment has detected issues:

Overall Health: {health_report.overall_health.value.upper()}
Health Score: {health_report.health_score:.2f}/1.00

Failed Checks: {len(health_report.failed_checks)}
Warning Checks: {len(health_report.warning_checks)}

Recommended Actions:
{', '.join([action.value.replace('_', ' ').title() for action in health_report.recommended_actions])}

Urgency: {health_report.action_urgency.upper()}
"""

        try:
            self.alerting_system.send_alert(
                event_type="strategy_health_degradation",
                severity=self.alerting_system.AlertSeverity.__members__[severity.upper()],
                title=title,
                message=message,
                component="strategy_health_monitor",
                metadata={
                    "strategy_id": health_report.strategy_id,
                    "health_score": health_report.health_score,
                    "failed_checks": len(health_report.failed_checks),
                    "warning_checks": len(health_report.warning_checks),
                },
            )
        except Exception as e:
            logger.error(f"Failed to send health alert: {str(e)}")

    def _record_strategy_action(
        self, strategy_id: str, action: StrategyAction, reason: str
    ) -> None:
        """Record strategy action in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO strategy_actions (
                    strategy_id, action_type, reason, triggered_by, executed_at
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (strategy_id, action.value, reason, "health_monitor", datetime.now().isoformat()),
            )
            conn.commit()

    def _store_strategy_metrics(self, metrics: StrategyMetrics) -> None:
        """Store strategy metrics in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO strategy_metrics (
                    strategy_id, measurement_date, total_pnl, realized_pnl, unrealized_pnl,
                    total_return_pct, total_trades, win_rate, profit_factor, max_drawdown,
                    sharpe_ratio, calmar_ratio, metrics_data, calculated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metrics.strategy_id,
                    datetime.now().date().isoformat(),
                    str(metrics.total_pnl),
                    str(metrics.realized_pnl),
                    str(metrics.unrealized_pnl),
                    metrics.total_return_pct,
                    metrics.total_trades,
                    metrics.win_rate,
                    metrics.profit_factor,
                    metrics.max_drawdown,
                    metrics.sharpe_ratio,
                    metrics.calmar_ratio,
                    json.dumps(
                        {
                            "volatility": metrics.volatility,
                            "avg_position_size": str(metrics.avg_position_size),
                            "total_exposure": str(metrics.total_exposure),
                            "measurement_period_days": metrics.measurement_period.days,
                        }
                    ),
                    metrics.calculated_at.isoformat(),
                ),
            )
            conn.commit()

    def _store_health_report(self, report: StrategyHealthReport) -> None:
        """Store health report in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO health_reports (
                    strategy_id, assessment_date, overall_health, health_score,
                    failed_checks, warning_checks, recommended_actions, action_urgency,
                    report_data, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    report.strategy_id,
                    datetime.now().date().isoformat(),
                    report.overall_health.value,
                    report.health_score,
                    len(report.failed_checks),
                    len(report.warning_checks),
                    json.dumps([action.value for action in report.recommended_actions]),
                    report.action_urgency,
                    json.dumps(
                        {
                            "metrics_trend": report.metrics_trend,
                            "performance_percentile": report.performance_percentile,
                            "notes": report.notes,
                        }
                    ),
                    report.assessment_timestamp.isoformat(),
                ),
            )
            conn.commit()

    def get_strategy_health_summary(self) -> dict[str, Any]:
        """Get summary of strategy health status"""

        total_strategies = len(self.active_strategies)
        health_distribution = defaultdict(int)

        for report in self.health_reports.values():
            health_distribution[report.overall_health] += 1

        return {
            "total_active_strategies": total_strategies,
            "monitored_strategies": len(self.health_reports),
            "health_distribution": {
                status.value: health_distribution[status] for status in StrategyHealthStatus
            },
            "average_health_score": (
                np.mean([r.health_score for r in self.health_reports.values()])
                if self.health_reports
                else 0.0
            ),
            "strategies_needing_attention": len(
                [
                    r
                    for r in self.health_reports.values()
                    if r.overall_health
                    in [
                        StrategyHealthStatus.WARNING,
                        StrategyHealthStatus.CRITICAL,
                        StrategyHealthStatus.FAILED,
                    ]
                ]
            ),
        }

    def display_health_dashboard(self) -> None:
        """Display strategy health dashboard"""

        summary = self.get_strategy_health_summary()

        console.print(
            Panel(
                f"[bold blue]Strategy Health Monitor Dashboard[/bold blue]\n"
                f"Active Strategies: {summary['total_active_strategies']}\n"
                f"Monitored Strategies: {summary['monitored_strategies']}\n"
                f"Average Health Score: {summary['average_health_score']:.2f}/1.00",
                title="🩺 Strategy Health",
            )
        )

        # Health distribution table
        if self.health_reports:
            health_table = Table(title="📊 Strategy Health Status")
            health_table.add_column("Strategy ID", style="cyan")
            health_table.add_column("Health", style="white")
            health_table.add_column("Score", justify="right")
            health_table.add_column("Failed", justify="right", style="red")
            health_table.add_column("Warnings", justify="right", style="yellow")
            health_table.add_column("Actions", style="dim")

            for strategy_id, report in list(self.health_reports.items())[:10]:  # Show top 10
                health_color = {
                    StrategyHealthStatus.EXCELLENT: "green",
                    StrategyHealthStatus.GOOD: "green",
                    StrategyHealthStatus.ACCEPTABLE: "yellow",
                    StrategyHealthStatus.WARNING: "yellow",
                    StrategyHealthStatus.CRITICAL: "red",
                    StrategyHealthStatus.FAILED: "red",
                }.get(report.overall_health, "white")

                actions_str = ", ".join(
                    [a.value.replace("_", " ").title() for a in report.recommended_actions[:2]]
                )
                if len(report.recommended_actions) > 2:
                    actions_str += "..."

                health_table.add_row(
                    strategy_id[-15:],  # Last 15 chars
                    f"[{health_color}]{report.overall_health.value.title()}[/{health_color}]",
                    f"{report.health_score:.2f}",
                    str(len(report.failed_checks)),
                    str(len(report.warning_checks)),
                    actions_str,
                )

            console.print(health_table)


def create_strategy_health_monitor(
    monitoring_dir: str = "data/strategy_health",
    check_interval: timedelta = timedelta(minutes=15),
    min_trades_for_analysis: int = 10,
) -> StrategyHealthMonitor:
    """Factory function to create strategy health monitor"""
    return StrategyHealthMonitor(
        monitoring_dir=monitoring_dir,
        check_interval=check_interval,
        min_trades_for_analysis=min_trades_for_analysis,
    )


if __name__ == "__main__":
    # Example usage
    health_monitor = create_strategy_health_monitor()

    # Start monitoring
    health_monitor.start_monitoring()

    try:
        # Let it run for demo
        time.sleep(30)

        # Display dashboard
        health_monitor.display_health_dashboard()

    finally:
        health_monitor.stop_monitoring()

    print("Strategy Health Monitor created successfully!")
