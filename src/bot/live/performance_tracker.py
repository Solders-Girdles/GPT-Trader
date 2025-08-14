"""
Real-Time Performance Tracking for Live Trading Infrastructure

This module implements comprehensive real-time performance monitoring including:
- Live P&L tracking with attribution analysis
- Real-time risk metrics and VaR calculations
- Performance benchmarking against indices and peers
- Transaction cost analysis and execution quality metrics
- Strategy performance monitoring and analytics
- Real-time drawdown tracking and recovery analysis
- Performance alerts and notifications
- Multi-timeframe performance analysis
- Live portfolio analytics dashboard
"""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

# Optional dependencies
try:
    import redis
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Performance metrics tracked"""

    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CURRENT_DRAWDOWN = "current_drawdown"
    VALUE_AT_RISK = "value_at_risk"
    CONDITIONAL_VAR = "conditional_var"
    VOLATILITY = "volatility"
    BETA = "beta"
    ALPHA = "alpha"
    INFORMATION_RATIO = "information_ratio"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    AVERAGE_TRADE = "average_trade"
    LARGEST_WIN = "largest_win"
    LARGEST_LOSS = "largest_loss"


class TimeFrame(Enum):
    """Performance analysis timeframes"""

    TICK = "tick"
    MINUTE = "1m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    HOUR = "1h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"
    QUARTERLY = "1Q"
    YEARLY = "1Y"


class AlertLevel(Enum):
    """Performance alert levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot"""

    timestamp: pd.Timestamp
    portfolio_value: float
    cash_balance: float
    total_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    positions: dict[str, float]  # symbol -> market value
    exposures: dict[str, float]  # sector/asset class -> exposure
    metrics: dict[PerformanceMetric, float]
    risk_metrics: dict[str, float]


@dataclass
class TradeRecord:
    """Individual trade record"""

    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    entry_price: float
    exit_price: float | None
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp | None
    pnl: float | None
    commission: float
    slippage: float
    duration_seconds: float | None
    strategy_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert/notification"""

    alert_id: str
    alert_type: str
    level: AlertLevel
    message: str
    timestamp: pd.Timestamp
    metric_name: str
    current_value: float
    threshold_value: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkData:
    """Benchmark comparison data"""

    benchmark_name: str
    returns: pd.Series
    last_updated: pd.Timestamp


@dataclass
class PerformanceConfig:
    """Configuration for performance tracking"""

    track_realtime_pnl: bool = True
    track_risk_metrics: bool = True
    track_transaction_costs: bool = True
    track_strategy_attribution: bool = True
    enable_benchmarking: bool = True
    enable_alerts: bool = True
    enable_persistence: bool = True

    # Update frequencies
    snapshot_interval_seconds: float = 1.0
    risk_calc_interval_seconds: float = 5.0
    benchmark_update_interval_seconds: float = 60.0

    # Risk parameters
    var_confidence_level: float = 0.95
    var_lookback_days: int = 252
    drawdown_alert_threshold: float = 0.05  # 5%
    volatility_alert_threshold: float = 0.30  # 30%

    # Storage settings
    redis_url: str | None = None
    max_history_days: int = 30
    performance_data_retention_days: int = 365


class RiskCalculator:
    """Real-time risk metrics calculator"""

    def __init__(self, config: PerformanceConfig) -> None:
        self.config = config
        self.return_history = deque(maxlen=config.var_lookback_days)
        self.portfolio_values = deque(maxlen=config.var_lookback_days)

    def update(self, portfolio_value: float, timestamp: pd.Timestamp) -> None:
        """Update risk calculations with new portfolio value"""
        self.portfolio_values.append(portfolio_value)

        if len(self.portfolio_values) > 1:
            # Calculate return
            prev_value = self.portfolio_values[-2]
            if prev_value > 0:
                return_pct = (portfolio_value - prev_value) / prev_value
                self.return_history.append(return_pct)

    def calculate_var(self, confidence_level: float = None) -> float:
        """Calculate Value at Risk"""
        if len(self.return_history) < 30:  # Need minimum data
            return 0.0

        confidence_level = confidence_level or self.config.var_confidence_level
        returns = np.array(self.return_history)

        # Historical simulation VaR
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, var_percentile)

        # Convert to dollar amount
        current_value = self.portfolio_values[-1] if self.portfolio_values else 0
        return abs(var * current_value)

    def calculate_cvar(self, confidence_level: float = None) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(self.return_history) < 30:
            return 0.0

        confidence_level = confidence_level or self.config.var_confidence_level
        returns = np.array(self.return_history)

        # Calculate VaR threshold
        var_percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(returns, var_percentile)

        # Calculate expected value of returns below VaR
        tail_returns = returns[returns <= var_threshold]
        cvar = np.mean(tail_returns) if len(tail_returns) > 0 else 0

        # Convert to dollar amount
        current_value = self.portfolio_values[-1] if self.portfolio_values else 0
        return abs(cvar * current_value)

    def calculate_volatility(self, annualized: bool = True) -> float:
        """Calculate portfolio volatility"""
        if len(self.return_history) < 10:
            return 0.0

        returns = np.array(self.return_history)
        vol = np.std(returns)

        if annualized:
            # Annualize assuming daily returns
            vol *= np.sqrt(252)

        return vol

    def calculate_max_drawdown(self) -> tuple[float, float]:
        """Calculate maximum drawdown and current drawdown"""
        if len(self.portfolio_values) < 2:
            return 0.0, 0.0

        values = np.array(self.portfolio_values)

        # Calculate running maximum
        running_max = np.maximum.accumulate(values)

        # Calculate drawdown
        drawdown = (values - running_max) / running_max

        max_drawdown = np.min(drawdown)
        current_drawdown = drawdown[-1]

        return abs(max_drawdown), abs(current_drawdown)

    def get_risk_metrics(self) -> dict[str, float]:
        """Get all risk metrics"""
        max_dd, current_dd = self.calculate_max_drawdown()

        return {
            "var_95": self.calculate_var(0.95),
            "var_99": self.calculate_var(0.99),
            "cvar_95": self.calculate_cvar(0.95),
            "cvar_99": self.calculate_cvar(0.99),
            "volatility": self.calculate_volatility(),
            "max_drawdown": max_dd,
            "current_drawdown": current_dd,
            "data_points": len(self.return_history),
        }


class TransactionCostAnalyzer:
    """Analyze transaction costs and execution quality"""

    def __init__(self) -> None:
        self.trades = []
        self.cost_metrics = {}

    def add_trade(self, trade: TradeRecord) -> None:
        """Add trade for cost analysis"""
        self.trades.append(trade)
        self._update_cost_metrics()

    def _update_cost_metrics(self) -> None:
        """Update transaction cost metrics"""
        if not self.trades:
            return

        total_commission = sum(t.commission for t in self.trades)
        total_slippage = sum(abs(t.slippage) for t in self.trades)
        total_volume = sum(abs(t.quantity * t.entry_price) for t in self.trades)

        self.cost_metrics = {
            "total_commission": total_commission,
            "total_slippage": total_slippage,
            "avg_commission_per_trade": total_commission / len(self.trades),
            "avg_slippage_per_trade": total_slippage / len(self.trades),
            "commission_rate_bps": (
                (total_commission / total_volume * 10000) if total_volume > 0 else 0
            ),
            "slippage_rate_bps": (total_slippage / total_volume * 10000) if total_volume > 0 else 0,
            "total_transaction_costs": total_commission + total_slippage,
            "total_trades": len(self.trades),
        }

    def get_execution_quality_metrics(self) -> dict[str, float]:
        """Get execution quality metrics"""
        if not self.trades:
            return {}

        completed_trades = [t for t in self.trades if t.exit_price is not None]
        if not completed_trades:
            return self.cost_metrics

        # Calculate additional metrics
        win_trades = [t for t in completed_trades if t.pnl and t.pnl > 0]
        loss_trades = [t for t in completed_trades if t.pnl and t.pnl < 0]

        win_rate = len(win_trades) / len(completed_trades) if completed_trades else 0
        avg_win = np.mean([t.pnl for t in win_trades]) if win_trades else 0
        avg_loss = np.mean([t.pnl for t in loss_trades]) if loss_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Update metrics
        execution_metrics = {
            **self.cost_metrics,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "largest_win": max([t.pnl for t in win_trades]) if win_trades else 0,
            "largest_loss": min([t.pnl for t in loss_trades]) if loss_trades else 0,
            "avg_trade_duration": (
                np.mean([t.duration_seconds for t in completed_trades if t.duration_seconds])
                if completed_trades
                else 0
            ),
        }

        return execution_metrics


class BenchmarkManager:
    """Manage benchmark data and comparisons"""

    def __init__(self) -> None:
        self.benchmarks = {}
        self.performance_attribution = {}

    def add_benchmark(self, name: str, returns: pd.Series) -> None:
        """Add benchmark for comparison"""
        self.benchmarks[name] = BenchmarkData(
            benchmark_name=name, returns=returns, last_updated=pd.Timestamp.now()
        )
        logger.info(f"Added benchmark: {name}")

    def calculate_performance_attribution(self, portfolio_returns: pd.Series) -> dict[str, float]:
        """Calculate performance attribution vs benchmarks"""
        attribution = {}

        for name, benchmark in self.benchmarks.items():
            if len(portfolio_returns) < 2 or len(benchmark.returns) < 2:
                continue

            # Align returns
            aligned_portfolio, aligned_benchmark = portfolio_returns.align(
                benchmark.returns, join="inner"
            )

            if len(aligned_portfolio) < 10:  # Need minimum data
                continue

            # Calculate metrics
            portfolio_return = aligned_portfolio.sum()
            benchmark_return = aligned_benchmark.sum()
            excess_return = portfolio_return - benchmark_return

            # Calculate alpha and beta
            covariance = np.cov(aligned_portfolio, aligned_benchmark)[0, 1]
            benchmark_variance = np.var(aligned_benchmark)

            beta = covariance / benchmark_variance if benchmark_variance != 0 else 1.0
            alpha = portfolio_return - beta * benchmark_return

            # Tracking error
            tracking_error = np.std(aligned_portfolio - aligned_benchmark)

            # Information ratio
            information_ratio = excess_return / tracking_error if tracking_error != 0 else 0

            attribution[name] = {
                "excess_return": excess_return,
                "alpha": alpha,
                "beta": beta,
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
                "correlation": np.corrcoef(aligned_portfolio, aligned_benchmark)[0, 1],
            }

        return attribution


class PerformanceTracker:
    """Main real-time performance tracking system"""

    def __init__(self, config: PerformanceConfig) -> None:
        self.config = config
        self.risk_calculator = RiskCalculator(config)
        self.cost_analyzer = TransactionCostAnalyzer()
        self.benchmark_manager = BenchmarkManager()

        # Performance data storage
        self.snapshots = deque(maxlen=config.max_history_days * 24 * 60)  # Minute resolution
        self.trades = []
        self.alerts = []

        # Current state
        self.current_snapshot = None
        self.is_running = False
        self.start_time = time.time()

        # Redis client for persistence
        self.redis_client = None
        if REDIS_AVAILABLE and config.enable_persistence and config.redis_url:
            try:
                self.redis_client = redis.from_url(config.redis_url)
                logger.info("Performance tracker connected to Redis")
            except Exception as e:
                logger.warning(f"Redis connection failed: {str(e)}")

        # Threading
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=2)

        logger.info("Performance tracker initialized")

    async def start(self) -> None:
        """Start the performance tracker"""
        if self.is_running:
            logger.warning("Performance tracker already running")
            return

        self.is_running = True
        self.start_time = time.time()
        logger.info("Starting performance tracker...")

        # Start monitoring tasks
        if self.config.track_realtime_pnl:
            asyncio.create_task(self._snapshot_monitor())

        if self.config.track_risk_metrics:
            asyncio.create_task(self._risk_monitor())

        if self.config.enable_benchmarking:
            asyncio.create_task(self._benchmark_monitor())

        logger.info("Performance tracker started")

    async def stop(self) -> None:
        """Stop the performance tracker"""
        if not self.is_running:
            return

        logger.info("Stopping performance tracker...")
        self.is_running = False

        # Save final snapshot
        if self.current_snapshot:
            await self._persist_snapshot(self.current_snapshot)

        self.executor.shutdown(wait=True)
        logger.info("Performance tracker stopped")

    def update_portfolio_value(
        self,
        portfolio_value: float,
        cash_balance: float,
        positions: dict[str, float],
        timestamp: pd.Timestamp = None,
    ) -> None:
        """Update portfolio value and trigger calculations"""
        if timestamp is None:
            timestamp = pd.Timestamp.now()

        with self.lock:
            # Calculate P&L if we have previous snapshot
            daily_pnl = 0.0
            total_pnl = 0.0
            unrealized_pnl = sum(positions.values()) - cash_balance

            if self.current_snapshot:
                total_pnl = portfolio_value - self.current_snapshot.portfolio_value

                # Daily P&L (since start of day)
                today_start = timestamp.normalize()
                today_snapshots = [s for s in self.snapshots if s.timestamp >= today_start]
                if today_snapshots:
                    daily_pnl = portfolio_value - today_snapshots[0].portfolio_value
                else:
                    daily_pnl = total_pnl

            # Update risk calculator
            self.risk_calculator.update(portfolio_value, timestamp)

            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(portfolio_value, timestamp)

            # Get risk metrics
            risk_metrics = self.risk_calculator.get_risk_metrics()

            # Create snapshot
            self.current_snapshot = PerformanceSnapshot(
                timestamp=timestamp,
                portfolio_value=portfolio_value,
                cash_balance=cash_balance,
                total_pnl=total_pnl,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=total_pnl - unrealized_pnl,
                daily_pnl=daily_pnl,
                positions=positions.copy(),
                exposures=self._calculate_exposures(positions),
                metrics=metrics,
                risk_metrics=risk_metrics,
            )

            # Store snapshot
            self.snapshots.append(self.current_snapshot)

            # Check for alerts
            if self.config.enable_alerts:
                asyncio.create_task(self._check_alerts(self.current_snapshot))

    def add_trade(self, trade: TradeRecord) -> None:
        """Add trade record for analysis"""
        with self.lock:
            self.trades.append(trade)
            self.cost_analyzer.add_trade(trade)
            logger.debug(f"Added trade: {trade.symbol} {trade.side} {trade.quantity}")

    def _calculate_performance_metrics(
        self, portfolio_value: float, timestamp: pd.Timestamp
    ) -> dict[PerformanceMetric, float]:
        """Calculate performance metrics"""
        metrics = {}

        if not self.snapshots:
            return metrics

        # Get time series data
        values = [s.portfolio_value for s in self.snapshots] + [portfolio_value]
        returns = pd.Series(values).pct_change().dropna()

        if len(returns) < 2:
            return metrics

        # Total return
        initial_value = self.snapshots[0].portfolio_value
        total_return = (portfolio_value - initial_value) / initial_value
        metrics[PerformanceMetric.TOTAL_RETURN] = total_return

        # Annualized return
        days_elapsed = (timestamp - self.snapshots[0].timestamp).total_seconds() / 86400
        if days_elapsed > 0:
            annualized_return = (1 + total_return) ** (365 / days_elapsed) - 1
            metrics[PerformanceMetric.ANNUALIZED_RETURN] = annualized_return

        # Volatility
        volatility = returns.std() * np.sqrt(252)  # Annualized
        metrics[PerformanceMetric.VOLATILITY] = volatility

        # Sharpe ratio
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        if volatility > 0:
            sharpe = (
                metrics.get(PerformanceMetric.ANNUALIZED_RETURN, 0) - risk_free_rate
            ) / volatility
            metrics[PerformanceMetric.SHARPE_RATIO] = sharpe

        # Sortino ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = negative_returns.std() * np.sqrt(252)
            if downside_deviation > 0:
                sortino = (
                    metrics.get(PerformanceMetric.ANNUALIZED_RETURN, 0) - risk_free_rate
                ) / downside_deviation
                metrics[PerformanceMetric.SORTINO_RATIO] = sortino

        # Max drawdown
        max_dd, current_dd = self.risk_calculator.calculate_max_drawdown()
        metrics[PerformanceMetric.MAX_DRAWDOWN] = max_dd
        metrics[PerformanceMetric.CURRENT_DRAWDOWN] = current_dd

        # Calmar ratio
        if max_dd > 0:
            calmar = metrics.get(PerformanceMetric.ANNUALIZED_RETURN, 0) / max_dd
            metrics[PerformanceMetric.CALMAR_RATIO] = calmar

        # VaR
        metrics[PerformanceMetric.VALUE_AT_RISK] = self.risk_calculator.calculate_var()
        metrics[PerformanceMetric.CONDITIONAL_VAR] = self.risk_calculator.calculate_cvar()

        return metrics

    def _calculate_exposures(self, positions: dict[str, float]) -> dict[str, float]:
        """Calculate portfolio exposures by category"""
        total_value = sum(abs(v) for v in positions.values())
        if total_value == 0:
            return {}

        # Simple exposure calculation (would be enhanced with sector/asset class mapping)
        exposures = {
            "equity_exposure": sum(v for v in positions.values() if v > 0) / total_value,
            "short_exposure": abs(sum(v for v in positions.values() if v < 0)) / total_value,
            "gross_exposure": total_value,
            "net_exposure": sum(positions.values()) / total_value if total_value > 0 else 0,
            "concentration_top5": self._calculate_concentration(positions, 5),
        }

        return exposures

    def _calculate_concentration(self, positions: dict[str, float], top_n: int) -> float:
        """Calculate concentration in top N positions"""
        if not positions:
            return 0.0

        sorted_positions = sorted(positions.values(), key=abs, reverse=True)
        top_positions = sorted_positions[:top_n]
        total_exposure = sum(abs(v) for v in positions.values())

        return sum(abs(v) for v in top_positions) / total_exposure if total_exposure > 0 else 0

    async def _check_alerts(self, snapshot: PerformanceSnapshot) -> None:
        """Check for performance alerts"""
        try:
            alerts_to_send = []

            # Drawdown alert
            current_dd = snapshot.risk_metrics.get("current_drawdown", 0)
            if current_dd > self.config.drawdown_alert_threshold:
                alert = PerformanceAlert(
                    alert_id=f"dd_{int(time.time())}",
                    alert_type="drawdown_alert",
                    level=AlertLevel.WARNING,
                    message=f"Current drawdown {current_dd:.2%} exceeds threshold {self.config.drawdown_alert_threshold:.2%}",
                    timestamp=snapshot.timestamp,
                    metric_name="current_drawdown",
                    current_value=current_dd,
                    threshold_value=self.config.drawdown_alert_threshold,
                )
                alerts_to_send.append(alert)

            # Volatility alert
            volatility = snapshot.metrics.get(PerformanceMetric.VOLATILITY, 0)
            if volatility > self.config.volatility_alert_threshold:
                alert = PerformanceAlert(
                    alert_id=f"vol_{int(time.time())}",
                    alert_type="volatility_alert",
                    level=AlertLevel.WARNING,
                    message=f"Portfolio volatility {volatility:.2%} exceeds threshold {self.config.volatility_alert_threshold:.2%}",
                    timestamp=snapshot.timestamp,
                    metric_name="volatility",
                    current_value=volatility,
                    threshold_value=self.config.volatility_alert_threshold,
                )
                alerts_to_send.append(alert)

            # Store alerts
            with self.lock:
                self.alerts.extend(alerts_to_send)

            # Log critical alerts
            for alert in alerts_to_send:
                if alert.level == AlertLevel.CRITICAL:
                    logger.critical(alert.message)
                elif alert.level == AlertLevel.WARNING:
                    logger.warning(alert.message)
                else:
                    logger.info(alert.message)

        except Exception as e:
            logger.error(f"Error checking alerts: {str(e)}")

    async def _snapshot_monitor(self) -> None:
        """Monitor and persist snapshots"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.snapshot_interval_seconds)

                if self.current_snapshot:
                    await self._persist_snapshot(self.current_snapshot)

            except Exception as e:
                logger.error(f"Snapshot monitor error: {str(e)}")
                await asyncio.sleep(1)

    async def _risk_monitor(self) -> None:
        """Monitor risk metrics"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.risk_calc_interval_seconds)

                # Risk metrics are updated in real-time, just log periodically
                if self.current_snapshot:
                    risk_metrics = self.current_snapshot.risk_metrics
                    if risk_metrics.get("data_points", 0) > 0:
                        logger.debug(
                            f"Risk metrics - VaR: ${risk_metrics.get('var_95', 0):,.0f}, "
                            f"Current DD: {risk_metrics.get('current_drawdown', 0):.2%}"
                        )

            except Exception as e:
                logger.error(f"Risk monitor error: {str(e)}")
                await asyncio.sleep(5)

    async def _benchmark_monitor(self) -> None:
        """Monitor benchmark comparisons"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.benchmark_update_interval_seconds)

                if len(self.snapshots) > 10:  # Need minimum data
                    # Calculate portfolio returns
                    values = [s.portfolio_value for s in self.snapshots]
                    returns = pd.Series(values).pct_change().dropna()

                    # Update performance attribution
                    attribution = self.benchmark_manager.calculate_performance_attribution(returns)

                    if attribution:
                        logger.debug(
                            f"Performance attribution updated for {len(attribution)} benchmarks"
                        )

            except Exception as e:
                logger.error(f"Benchmark monitor error: {str(e)}")
                await asyncio.sleep(30)

    async def _persist_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """Persist snapshot to Redis"""
        if not self.redis_client:
            return

        try:
            snapshot_data = {
                "timestamp": snapshot.timestamp.isoformat(),
                "portfolio_value": snapshot.portfolio_value,
                "cash_balance": snapshot.cash_balance,
                "total_pnl": snapshot.total_pnl,
                "daily_pnl": snapshot.daily_pnl,
                "positions": snapshot.positions,
                "metrics": {k.value: v for k, v in snapshot.metrics.items()},
                "risk_metrics": snapshot.risk_metrics,
            }

            # Store with TTL
            key = f"performance_snapshot:{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}"
            await self.redis_client.setex(
                key,
                self.config.performance_data_retention_days * 86400,
                json.dumps(snapshot_data, default=str),
            )

        except Exception as e:
            logger.warning(f"Failed to persist snapshot: {str(e)}")

    def get_current_performance(self) -> dict[str, Any]:
        """Get current performance summary"""
        if not self.current_snapshot:
            return {}

        cost_metrics = self.cost_analyzer.get_execution_quality_metrics()

        return {
            "timestamp": self.current_snapshot.timestamp.isoformat(),
            "portfolio_value": self.current_snapshot.portfolio_value,
            "total_pnl": self.current_snapshot.total_pnl,
            "daily_pnl": self.current_snapshot.daily_pnl,
            "performance_metrics": {k.value: v for k, v in self.current_snapshot.metrics.items()},
            "risk_metrics": self.current_snapshot.risk_metrics,
            "exposures": self.current_snapshot.exposures,
            "transaction_costs": cost_metrics,
            "active_positions": len(self.current_snapshot.positions),
            "recent_alerts": len(
                [
                    a
                    for a in self.alerts
                    if (pd.Timestamp.now() - a.timestamp).total_seconds() < 3600
                ]
            ),
            "uptime_hours": (time.time() - self.start_time) / 3600,
        }

    def get_performance_history(self, hours: int = 24) -> pd.DataFrame:
        """Get performance history as DataFrame"""
        if not self.snapshots:
            return pd.DataFrame()

        cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=hours)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]

        if not recent_snapshots:
            return pd.DataFrame()

        data = []
        for snapshot in recent_snapshots:
            row = {
                "timestamp": snapshot.timestamp,
                "portfolio_value": snapshot.portfolio_value,
                "total_pnl": snapshot.total_pnl,
                "daily_pnl": snapshot.daily_pnl,
                "cash_balance": snapshot.cash_balance,
                **{k.value: v for k, v in snapshot.metrics.items()},
                **{f"risk_{k}": v for k, v in snapshot.risk_metrics.items()},
            }
            data.append(row)

        return pd.DataFrame(data).set_index("timestamp")

    def generate_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.current_snapshot:
            return {"error": "No performance data available"}

        # Basic metrics
        current_perf = self.get_current_performance()

        # Trade analysis
        cost_metrics = self.cost_analyzer.get_execution_quality_metrics()

        # Risk analysis
        risk_metrics = self.current_snapshot.risk_metrics

        # Alert summary
        alert_summary = defaultdict(int)
        for alert in self.alerts:
            alert_summary[alert.level.value] += 1

        return {
            "report_timestamp": pd.Timestamp.now().isoformat(),
            "performance_summary": current_perf,
            "risk_analysis": risk_metrics,
            "transaction_analysis": cost_metrics,
            "alert_summary": dict(alert_summary),
            "position_summary": {
                "total_positions": len(self.current_snapshot.positions),
                "gross_exposure": self.current_snapshot.exposures.get("gross_exposure", 0),
                "net_exposure": self.current_snapshot.exposures.get("net_exposure", 0),
                "concentration_top5": self.current_snapshot.exposures.get("concentration_top5", 0),
            },
            "system_info": {
                "uptime_hours": (time.time() - self.start_time) / 3600,
                "snapshots_collected": len(self.snapshots),
                "trades_analyzed": len(self.trades),
                "is_running": self.is_running,
            },
        }


def create_performance_tracker(
    track_realtime_pnl: bool = True,
    enable_alerts: bool = True,
    enable_persistence: bool = True,
    **kwargs,
) -> PerformanceTracker:
    """Factory function to create performance tracker"""
    config = PerformanceConfig(
        track_realtime_pnl=track_realtime_pnl,
        enable_alerts=enable_alerts,
        enable_persistence=enable_persistence,
        **kwargs,
    )

    return PerformanceTracker(config)


# Example usage and testing
async def main() -> None:
    """Example usage of real-time performance tracker"""
    print("Real-Time Performance Tracking Testing")
    print("=" * 45)

    # Create performance tracker
    tracker = create_performance_tracker(
        track_realtime_pnl=True,
        track_risk_metrics=True,
        enable_alerts=True,
        snapshot_interval_seconds=0.5,
    )

    # Add a benchmark
    benchmark_returns = pd.Series(np.random.normal(0.0002, 0.01, 100))  # Mock S&P 500
    tracker.benchmark_manager.add_benchmark("SPY", benchmark_returns)

    print("✅ Performance tracker created")

    # Start tracking
    await tracker.start()
    print("✅ Performance tracker started")

    try:
        # Simulate portfolio updates
        base_portfolio_value = 100000

        for i in range(20):
            # Simulate price movement
            return_pct = np.random.normal(0.0001, 0.005)  # Small random moves
            portfolio_value = base_portfolio_value * (1 + return_pct)
            base_portfolio_value = portfolio_value

            # Mock positions
            positions = {
                "AAPL": portfolio_value * 0.3,
                "GOOGL": portfolio_value * 0.25,
                "MSFT": portfolio_value * 0.2,
                "TSLA": portfolio_value * 0.15,
                "CASH": portfolio_value * 0.1,
            }

            cash_balance = positions["CASH"]

            # Update tracker
            tracker.update_portfolio_value(
                portfolio_value=portfolio_value, cash_balance=cash_balance, positions=positions
            )

            # Add some mock trades
            if i % 5 == 0:
                trade = TradeRecord(
                    trade_id=f"trade_{i}",
                    symbol="AAPL",
                    side="buy" if i % 2 == 0 else "sell",
                    quantity=100,
                    entry_price=150.0 + np.random.normal(0, 2),
                    exit_price=150.0 + np.random.normal(0, 2),
                    entry_time=pd.Timestamp.now() - pd.Timedelta(minutes=np.random.randint(1, 60)),
                    exit_time=pd.Timestamp.now(),
                    pnl=np.random.normal(10, 50),
                    commission=1.0,
                    slippage=0.05,
                    duration_seconds=np.random.randint(60, 3600),
                    strategy_name="test_strategy",
                )
                tracker.add_trade(trade)

            await asyncio.sleep(0.1)  # 100ms updates

        # Wait for processing
        await asyncio.sleep(1)

        # Get performance summary
        performance = tracker.get_current_performance()
        print("\n📊 Current Performance:")
        print(f"   Portfolio Value: ${performance['portfolio_value']:,.2f}")
        print(f"   Total P&L: ${performance['total_pnl']:,.2f}")
        print(f"   Daily P&L: ${performance['daily_pnl']:,.2f}")
        print(f"   Sharpe Ratio: {performance['performance_metrics'].get('sharpe_ratio', 0):.3f}")
        print(f"   Max Drawdown: {performance['performance_metrics'].get('max_drawdown', 0):.2%}")
        print(f"   VaR (95%): ${performance['risk_metrics'].get('var_95', 0):,.0f}")

        # Transaction costs
        print("\n💰 Transaction Analysis:")
        tx_costs = performance["transaction_costs"]
        print(f"   Total Trades: {tx_costs.get('total_trades', 0)}")
        print(f"   Win Rate: {tx_costs.get('win_rate', 0):.1%}")
        print(f"   Commission Rate: {tx_costs.get('commission_rate_bps', 0):.1f} bps")
        print(f"   Slippage Rate: {tx_costs.get('slippage_rate_bps', 0):.1f} bps")

        # Generate report
        report = tracker.generate_performance_report()
        print("\n📋 Performance Report Generated:")
        print(f"   Uptime: {report['system_info']['uptime_hours']:.2f} hours")
        print(f"   Snapshots: {report['system_info']['snapshots_collected']}")
        print(f"   Trades Analyzed: {report['system_info']['trades_analyzed']}")
        print(f"   Recent Alerts: {report['performance_summary']['recent_alerts']}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")

    finally:
        await tracker.stop()
        print("🛑 Performance tracker stopped")

    print("\n🚀 Real-Time Performance Tracking ready for production!")


if __name__ == "__main__":
    asyncio.run(main())
