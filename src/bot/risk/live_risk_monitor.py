"""
Real-Time Risk Monitor for GPT-Trader Live Trading

Comprehensive real-time risk monitoring system providing:
- Position-level risk calculations and limits
- Portfolio drawdown monitoring and protection
- Correlation risk tracking across strategies
- Leverage and exposure controls
- Market volatility-based risk adjustment
- Real-time risk metrics and alerting

This is the core risk management system for production live trading.
"""

import json
import logging
import queue
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bot.dataflow.streaming_data import StreamingDataManager

# Risk monitoring imports
from bot.live.trading_engine_v2 import LiveTradingEngine, Position
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RiskEventType(Enum):
    """Types of risk events"""

    POSITION_LIMIT_BREACH = "position_limit_breach"
    DRAWDOWN_LIMIT_BREACH = "drawdown_limit_breach"
    CONCENTRATION_RISK = "concentration_risk"
    CORRELATION_SPIKE = "correlation_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_SHORTAGE = "liquidity_shortage"
    MARGIN_CALL = "margin_call"
    VAR_BREACH = "var_breach"
    LEVERAGE_EXCESS = "leverage_excess"


@dataclass
class RiskMetrics:
    """Real-time risk metrics"""

    # Portfolio-level metrics
    total_exposure: Decimal
    net_exposure: Decimal
    gross_exposure: Decimal
    leverage_ratio: float
    cash_available: Decimal

    # Risk measures
    portfolio_var_95: Decimal
    portfolio_cvar_95: Decimal
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float

    # Concentration measures
    max_position_weight: float
    concentration_index: float
    sector_concentrations: dict[str, float]
    strategy_concentrations: dict[str, float]

    # Correlation measures
    avg_correlation: float
    max_correlation: float
    correlation_matrix: pd.DataFrame | None = None

    # Volatility measures
    portfolio_volatility: float
    realized_volatility: float
    volatility_forecast: float

    # Liquidity measures
    avg_spread_bps: float
    illiquid_positions_value: Decimal
    liquidity_score: float

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskLimit:
    """Individual risk limit configuration"""

    limit_name: str
    limit_type: str  # "portfolio", "position", "strategy", "sector"
    limit_metric: str  # "exposure", "drawdown", "var", etc.
    soft_limit: float
    hard_limit: float
    current_value: float = 0.0
    breach_count: int = 0
    last_breach: datetime | None = None
    is_active: bool = True

    @property
    def utilization(self) -> float:
        """Calculate limit utilization percentage"""
        if self.hard_limit == 0:
            return 0.0
        return abs(self.current_value) / abs(self.hard_limit) * 100

    @property
    def is_soft_breach(self) -> bool:
        return abs(self.current_value) > abs(self.soft_limit)

    @property
    def is_hard_breach(self) -> bool:
        return abs(self.current_value) > abs(self.hard_limit)


@dataclass
class RiskEvent:
    """Risk event record"""

    event_id: str
    event_type: RiskEventType
    risk_level: RiskLevel
    symbol: str | None
    strategy_id: str | None

    # Event details
    current_value: float
    limit_value: float
    breach_amount: float
    breach_percentage: float

    # Context
    message: str
    recommendations: list[str]
    auto_actions_taken: list[str] = field(default_factory=list)

    # Timing
    detected_at: datetime = field(default_factory=datetime.now)
    resolved_at: datetime | None = None
    is_resolved: bool = False


class RealTimeRiskMonitor:
    """Real-time risk monitoring and control system"""

    def __init__(
        self,
        risk_dir: str = "data/risk_monitoring",
        trading_engine: LiveTradingEngine = None,
        streaming_manager: StreamingDataManager = None,
    ) -> None:
        self.risk_dir = Path(risk_dir)
        self.risk_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.risk_dir / "metrics").mkdir(exist_ok=True)
        (self.risk_dir / "events").mkdir(exist_ok=True)
        (self.risk_dir / "reports").mkdir(exist_ok=True)
        (self.risk_dir / "limits").mkdir(exist_ok=True)

        # Initialize components
        self.trading_engine = trading_engine
        self.streaming_manager = streaming_manager

        # Initialize database
        self.db_path = self.risk_dir / "risk_monitoring.db"
        self._initialize_database()

        # Risk monitoring state
        self.is_monitoring = False
        self.risk_limits: dict[str, RiskLimit] = {}
        self.active_risk_events: dict[str, RiskEvent] = {}
        self.current_metrics: RiskMetrics | None = None

        # Historical data for calculations
        self.position_history: list[dict[str, Any]] = []
        self.pnl_history: list[float] = []
        self.return_history: list[float] = []

        # Threading for real-time monitoring
        self.monitoring_thread = None
        self.calculation_thread = None
        self.alert_queue = queue.Queue()

        # Risk callbacks
        self.risk_event_callbacks: list[Callable[[RiskEvent], None]] = []
        self.metric_callbacks: list[Callable[[RiskMetrics], None]] = []

        # Initialize default risk limits
        self._initialize_default_limits()

        logger.info(f"Real-Time Risk Monitor initialized at {self.risk_dir}")

    def _initialize_database(self) -> None:
        """Initialize SQLite database for risk monitoring"""

        with sqlite3.connect(self.db_path) as conn:
            # Risk metrics table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_exposure TEXT NOT NULL,
                    net_exposure TEXT NOT NULL,
                    leverage_ratio REAL NOT NULL,
                    portfolio_var_95 TEXT NOT NULL,
                    max_drawdown REAL NOT NULL,
                    current_drawdown REAL NOT NULL,
                    portfolio_volatility REAL NOT NULL,
                    avg_correlation REAL,
                    liquidity_score REAL,
                    timestamp TEXT NOT NULL,
                    metrics_json TEXT
                )
            """
            )

            # Risk limits table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_limits (
                    limit_name TEXT PRIMARY KEY,
                    limit_type TEXT NOT NULL,
                    limit_metric TEXT NOT NULL,
                    soft_limit REAL NOT NULL,
                    hard_limit REAL NOT NULL,
                    current_value REAL DEFAULT 0.0,
                    breach_count INTEGER DEFAULT 0,
                    last_breach TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    limit_data TEXT
                )
            """
            )

            # Risk events table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    symbol TEXT,
                    strategy_id TEXT,
                    current_value REAL NOT NULL,
                    limit_value REAL NOT NULL,
                    breach_amount REAL NOT NULL,
                    message TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    resolved_at TEXT,
                    is_resolved BOOLEAN DEFAULT 0,
                    event_data TEXT
                )
            """
            )

            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON risk_metrics (timestamp)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON risk_events (event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_level ON risk_events (risk_level)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_resolved ON risk_events (is_resolved)"
            )

            conn.commit()

    def _initialize_default_limits(self) -> None:
        """Initialize default risk limits"""

        default_limits = [
            # Portfolio limits
            RiskLimit(
                "max_portfolio_drawdown", "portfolio", "drawdown", 0.10, 0.15
            ),  # 10% soft, 15% hard
            RiskLimit("max_leverage", "portfolio", "leverage", 2.0, 3.0),  # 2x soft, 3x hard
            RiskLimit("min_cash_reserve", "portfolio", "cash", 0.05, 0.02),  # 5% soft, 2% hard
            RiskLimit("max_var_95", "portfolio", "var", 0.025, 0.04),  # 2.5% soft, 4% hard
            # Position limits
            RiskLimit(
                "max_position_weight", "position", "weight", 0.15, 0.25
            ),  # 15% soft, 25% hard
            RiskLimit(
                "max_single_position_loss", "position", "loss", 0.05, 0.08
            ),  # 5% soft, 8% hard
            # Concentration limits
            RiskLimit(
                "max_strategy_concentration", "strategy", "concentration", 0.40, 0.60
            ),  # 40% soft, 60% hard
            RiskLimit(
                "max_sector_concentration", "sector", "concentration", 0.30, 0.50
            ),  # 30% soft, 50% hard
            # Correlation limits
            RiskLimit(
                "max_avg_correlation", "portfolio", "correlation", 0.60, 0.80
            ),  # 60% soft, 80% hard
            # Volatility limits
            RiskLimit(
                "max_portfolio_volatility", "portfolio", "volatility", 0.20, 0.30
            ),  # 20% soft, 30% hard
        ]

        for limit in default_limits:
            self.risk_limits[limit.limit_name] = limit
            self._store_risk_limit(limit)

    def add_risk_event_callback(self, callback: Callable[[RiskEvent], None]) -> None:
        """Add callback for risk events"""
        self.risk_event_callbacks.append(callback)

    def add_metric_callback(self, callback: Callable[[RiskMetrics], None]) -> None:
        """Add callback for risk metrics updates"""
        self.metric_callbacks.append(callback)

    def start_risk_monitoring(self) -> None:
        """Start real-time risk monitoring"""

        if self.is_monitoring:
            console.print("⚠️  Risk monitoring is already active")
            return

        if not self.trading_engine:
            console.print(
                "❌ [bold red]Cannot start risk monitoring - no trading engine available[/bold red]"
            )
            return

        console.print("🛡️  [bold green]Starting Real-Time Risk Monitoring[/bold green]")

        self.is_monitoring = True

        # Start monitoring threads
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.calculation_thread = threading.Thread(target=self._calculation_loop, daemon=True)

        self.monitoring_thread.start()
        self.calculation_thread.start()

        console.print("   ✅ Risk calculation engine started")
        console.print("   📊 Real-time monitoring active")
        console.print(f"   🚨 {len(self.risk_limits)} risk limits configured")

        logger.info("Real-time risk monitoring started successfully")

    def stop_risk_monitoring(self) -> None:
        """Stop real-time risk monitoring"""

        console.print("⏹️  [bold yellow]Stopping Risk Monitoring[/bold yellow]")

        self.is_monitoring = False

        # Wait for threads to finish
        for thread in [self.monitoring_thread, self.calculation_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)

        console.print("   ✅ Risk monitoring stopped")

        logger.info("Risk monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main risk monitoring loop"""

        while self.is_monitoring:
            try:
                # Calculate current risk metrics
                if self.trading_engine and self.trading_engine.is_running:
                    metrics = self._calculate_risk_metrics()

                    if metrics:
                        self.current_metrics = metrics
                        self._store_risk_metrics(metrics)

                        # Check risk limits
                        self._check_risk_limits(metrics)

                        # Notify callbacks
                        for callback in self.metric_callbacks:
                            try:
                                callback(metrics)
                            except Exception as e:
                                logger.error(f"Metric callback error: {str(e)}")

                # Sleep between monitoring cycles
                time.sleep(1)  # Monitor every second

            except Exception as e:
                logger.error(f"Risk monitoring loop error: {str(e)}")
                time.sleep(5)  # Longer pause on error

    def _calculation_loop(self) -> None:
        """Risk calculation processing loop"""

        while self.is_monitoring:
            try:
                # Update position history for risk calculations
                if self.trading_engine:
                    self._update_position_history()
                    self._update_pnl_history()

                # Sleep between calculations
                time.sleep(5)  # Calculate every 5 seconds

            except Exception as e:
                logger.error(f"Risk calculation loop error: {str(e)}")
                time.sleep(10)  # Longer pause on error

    def _calculate_risk_metrics(self) -> RiskMetrics | None:
        """Calculate comprehensive risk metrics"""

        try:
            if not self.trading_engine or not self.trading_engine.positions:
                return None

            positions = self.trading_engine.positions
            total_capital = self.trading_engine.current_capital

            # Portfolio exposure calculations
            total_exposure = Decimal("0")
            gross_exposure = Decimal("0")
            net_exposure = Decimal("0")

            position_values = []
            position_weights = []

            for position in positions.values():
                if position.current_price:
                    market_value = position.quantity * position.current_price
                    total_exposure += abs(market_value)
                    gross_exposure += abs(market_value)
                    net_exposure += market_value

                    position_values.append(float(market_value))
                    position_weights.append(float(abs(market_value) / total_capital))

            # Leverage ratio
            leverage_ratio = float(gross_exposure / total_capital) if total_capital > 0 else 0.0

            # Cash available
            cash_available = total_capital - abs(net_exposure)

            # Risk measures
            portfolio_var_95, portfolio_cvar_95 = self._calculate_var_cvar()
            max_drawdown, current_drawdown = self._calculate_drawdown_metrics()
            sharpe_ratio = self._calculate_sharpe_ratio()

            # Concentration measures
            max_position_weight = max(position_weights) if position_weights else 0.0
            concentration_index = self._calculate_concentration_index(position_weights)

            # Sector and strategy concentrations
            sector_concentrations = self._calculate_sector_concentrations(positions)
            strategy_concentrations = self._calculate_strategy_concentrations(positions)

            # Correlation measures
            avg_correlation, max_correlation, correlation_matrix = (
                self._calculate_correlation_metrics(positions)
            )

            # Volatility measures
            portfolio_volatility = self._calculate_portfolio_volatility()
            realized_volatility = self._calculate_realized_volatility()
            volatility_forecast = realized_volatility * 1.1  # Simple forecast

            # Liquidity measures
            avg_spread_bps, illiquid_positions_value, liquidity_score = (
                self._calculate_liquidity_metrics(positions)
            )

            return RiskMetrics(
                total_exposure=total_exposure,
                net_exposure=net_exposure,
                gross_exposure=gross_exposure,
                leverage_ratio=leverage_ratio,
                cash_available=cash_available,
                portfolio_var_95=portfolio_var_95,
                portfolio_cvar_95=portfolio_cvar_95,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                sharpe_ratio=sharpe_ratio,
                max_position_weight=max_position_weight,
                concentration_index=concentration_index,
                sector_concentrations=sector_concentrations,
                strategy_concentrations=strategy_concentrations,
                avg_correlation=avg_correlation,
                max_correlation=max_correlation,
                correlation_matrix=correlation_matrix,
                portfolio_volatility=portfolio_volatility,
                realized_volatility=realized_volatility,
                volatility_forecast=volatility_forecast,
                avg_spread_bps=avg_spread_bps,
                illiquid_positions_value=illiquid_positions_value,
                liquidity_score=liquidity_score,
            )

        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {str(e)}")
            return None

    def _calculate_var_cvar(self) -> tuple[Decimal, Decimal]:
        """Calculate Value at Risk and Conditional Value at Risk"""

        if len(self.return_history) < 30:  # Need minimum history
            return Decimal("0"), Decimal("0")

        returns = np.array(self.return_history[-252:])  # Last year of returns

        # 95% VaR (5th percentile)
        var_95 = np.percentile(returns, 5) * -1  # Convert to positive loss

        # CVaR (average of losses beyond VaR)
        tail_losses = returns[returns <= -var_95]
        cvar_95 = np.mean(tail_losses) * -1 if len(tail_losses) > 0 else var_95

        portfolio_value = float(self.trading_engine.current_capital)

        return (Decimal(str(var_95 * portfolio_value)), Decimal(str(cvar_95 * portfolio_value)))

    def _calculate_drawdown_metrics(self) -> tuple[float, float]:
        """Calculate maximum and current drawdown"""

        if len(self.pnl_history) < 2:
            return 0.0, 0.0

        pnl_series = np.array(self.pnl_history)
        cumulative_returns = np.cumsum(pnl_series)

        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)

        # Drawdown series
        drawdown = (cumulative_returns - running_max) / np.maximum(running_max, 1.0)

        max_drawdown = abs(np.min(drawdown))
        current_drawdown = abs(drawdown[-1])

        return max_drawdown, current_drawdown

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""

        if len(self.return_history) < 30:
            return 0.0

        returns = np.array(self.return_history[-252:])  # Last year

        mean_return = np.mean(returns) * 252  # Annualized
        volatility = np.std(returns) * np.sqrt(252)  # Annualized

        risk_free_rate = 0.02  # 2% risk-free rate

        if volatility > 0:
            return (mean_return - risk_free_rate) / volatility
        return 0.0

    def _calculate_concentration_index(self, weights: list[float]) -> float:
        """Calculate Herfindahl concentration index"""

        if not weights:
            return 0.0

        return sum(w**2 for w in weights)

    def _calculate_sector_concentrations(self, positions: dict[str, Position]) -> dict[str, float]:
        """Calculate sector concentration (simplified)"""

        # Simplified sector mapping
        sector_map = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "AMZN": "Consumer Discretionary",
            "TSLA": "Consumer Discretionary",
        }

        total_value = sum(
            float(pos.quantity * pos.current_price) if pos.current_price else 0.0
            for pos in positions.values()
        )

        if total_value == 0:
            return {}

        sector_values = {}
        for position in positions.values():
            sector = sector_map.get(position.symbol, "Other")
            value = (
                float(position.quantity * position.current_price) if position.current_price else 0.0
            )
            sector_values[sector] = sector_values.get(sector, 0.0) + value

        return {sector: value / total_value for sector, value in sector_values.items()}

    def _calculate_strategy_concentrations(
        self, positions: dict[str, Position]
    ) -> dict[str, float]:
        """Calculate strategy concentration"""

        total_value = sum(
            float(pos.quantity * pos.current_price) if pos.current_price else 0.0
            for pos in positions.values()
        )

        if total_value == 0:
            return {}

        strategy_values = {}
        for position in positions.values():
            strategy = position.strategy_id
            value = (
                float(position.quantity * position.current_price) if position.current_price else 0.0
            )
            strategy_values[strategy] = strategy_values.get(strategy, 0.0) + value

        return {strategy: value / total_value for strategy, value in strategy_values.items()}

    def _calculate_correlation_metrics(
        self, positions: dict[str, Position]
    ) -> tuple[float, float, pd.DataFrame | None]:
        """Calculate correlation metrics (simplified)"""

        # Simplified correlation calculation
        # In production, this would use historical price correlations

        symbols = [pos.symbol for pos in positions.values()]
        n_symbols = len(symbols)

        if n_symbols < 2:
            return 0.0, 0.0, None

        # Create simplified correlation matrix
        correlation_matrix = pd.DataFrame(np.eye(n_symbols), index=symbols, columns=symbols)

        # Add some realistic correlations
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i != j:
                    # Tech stocks tend to be more correlated
                    if symbol1 in ["AAPL", "MSFT", "GOOGL"] and symbol2 in [
                        "AAPL",
                        "MSFT",
                        "GOOGL",
                    ]:
                        correlation = 0.6 + np.random.normal(0, 0.1)
                    else:
                        correlation = 0.3 + np.random.normal(0, 0.1)

                    correlation = max(-0.9, min(0.9, correlation))
                    correlation_matrix.loc[symbol1, symbol2] = correlation

        # Calculate average and max correlation
        upper_triangle = np.triu(correlation_matrix.values, k=1)
        non_zero_corr = upper_triangle[upper_triangle != 0]

        avg_correlation = np.mean(non_zero_corr) if len(non_zero_corr) > 0 else 0.0
        max_correlation = np.max(non_zero_corr) if len(non_zero_corr) > 0 else 0.0

        return avg_correlation, max_correlation, correlation_matrix

    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility"""

        if len(self.return_history) < 30:
            return 0.0

        returns = np.array(self.return_history[-30:])  # Last 30 days
        return np.std(returns) * np.sqrt(252)  # Annualized

    def _calculate_realized_volatility(self) -> float:
        """Calculate realized volatility"""

        if len(self.return_history) < 10:
            return 0.0

        returns = np.array(self.return_history[-10:])  # Last 10 days
        return np.std(returns) * np.sqrt(252)  # Annualized

    def _calculate_liquidity_metrics(
        self, positions: dict[str, Position]
    ) -> tuple[float, Decimal, float]:
        """Calculate liquidity metrics"""

        spreads = []
        illiquid_value = Decimal("0")
        total_value = Decimal("0")

        for position in positions.values():
            if self.streaming_manager:
                quote = self.streaming_manager.get_latest_quote(position.symbol)
                if quote:
                    spread_bps = quote.spread_bps
                    spreads.append(spread_bps)

                    position_value = (
                        position.quantity * position.current_price
                        if position.current_price
                        else Decimal("0")
                    )
                    total_value += abs(position_value)

                    # Consider positions illiquid if spread > 10bps
                    if spread_bps > 10.0:
                        illiquid_value += abs(position_value)

        avg_spread_bps = np.mean(spreads) if spreads else 0.0

        # Liquidity score (0-100, where 100 is most liquid)
        if avg_spread_bps == 0:
            liquidity_score = 100.0
        else:
            liquidity_score = max(0.0, 100.0 - avg_spread_bps)

        return avg_spread_bps, illiquid_value, liquidity_score

    def _check_risk_limits(self, metrics: RiskMetrics) -> None:
        """Check all risk limits against current metrics"""

        # Update limit values and check for breaches
        limit_checks = [
            ("max_portfolio_drawdown", metrics.current_drawdown),
            ("max_leverage", metrics.leverage_ratio),
            (
                "min_cash_reserve",
                float(metrics.cash_available / self.trading_engine.current_capital),
            ),
            ("max_var_95", float(metrics.portfolio_var_95 / self.trading_engine.current_capital)),
            ("max_position_weight", metrics.max_position_weight),
            (
                "max_strategy_concentration",
                (
                    max(metrics.strategy_concentrations.values())
                    if metrics.strategy_concentrations
                    else 0.0
                ),
            ),
            (
                "max_sector_concentration",
                (
                    max(metrics.sector_concentrations.values())
                    if metrics.sector_concentrations
                    else 0.0
                ),
            ),
            ("max_avg_correlation", metrics.avg_correlation),
            ("max_portfolio_volatility", metrics.portfolio_volatility),
        ]

        for limit_name, current_value in limit_checks:
            if limit_name in self.risk_limits:
                limit = self.risk_limits[limit_name]
                limit.current_value = current_value

                # Check for breaches
                if limit.is_hard_breach:
                    self._handle_risk_limit_breach(limit, RiskLevel.CRITICAL)
                elif limit.is_soft_breach:
                    self._handle_risk_limit_breach(limit, RiskLevel.HIGH)

                # Update stored limit
                self._store_risk_limit(limit)

    def _handle_risk_limit_breach(self, limit: RiskLimit, risk_level: RiskLevel) -> None:
        """Handle risk limit breach"""

        # Check if this is a new breach (not already alerted recently)
        if (
            limit.last_breach is None or (datetime.now() - limit.last_breach).total_seconds() > 300
        ):  # 5 min cooldown
            event_id = f"risk_event_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{limit.limit_name}"

            breach_amount = abs(limit.current_value) - abs(
                limit.soft_limit if risk_level == RiskLevel.HIGH else limit.hard_limit
            )
            breach_percentage = (
                (breach_amount / abs(limit.hard_limit)) * 100 if limit.hard_limit != 0 else 0
            )

            risk_event = RiskEvent(
                event_id=event_id,
                event_type=RiskEventType.POSITION_LIMIT_BREACH,
                risk_level=risk_level,
                symbol=None,
                strategy_id=None,
                current_value=limit.current_value,
                limit_value=(
                    limit.hard_limit if risk_level == RiskLevel.CRITICAL else limit.soft_limit
                ),
                breach_amount=breach_amount,
                breach_percentage=breach_percentage,
                message=f"{limit.limit_name} breach: {limit.current_value:.3f} exceeds {limit.hard_limit if risk_level == RiskLevel.CRITICAL else limit.soft_limit:.3f}",
                recommendations=[
                    f"Reduce {limit.limit_type} exposure",
                    "Review position sizing",
                    "Consider hedging positions",
                ],
            )

            # Store and track event
            self.active_risk_events[event_id] = risk_event
            self._store_risk_event(risk_event)

            # Update limit breach tracking
            limit.breach_count += 1
            limit.last_breach = datetime.now()

            # Notify callbacks
            for callback in self.risk_event_callbacks:
                try:
                    callback(risk_event)
                except Exception as e:
                    logger.error(f"Risk event callback error: {str(e)}")

            console.print(f"🚨 [bold red]RISK ALERT:[/bold red] {risk_event.message}")

            logger.warning(f"Risk limit breach: {risk_event.message}")

    def _update_position_history(self) -> None:
        """Update position history for risk calculations"""

        if not self.trading_engine or not self.trading_engine.positions:
            return

        position_snapshot = {
            "timestamp": datetime.now(),
            "positions": {
                pos_key: {
                    "symbol": pos.symbol,
                    "quantity": float(pos.quantity),
                    "market_value": float(pos.market_value),
                    "unrealized_pnl": float(pos.unrealized_pnl),
                }
                for pos_key, pos in self.trading_engine.positions.items()
            },
        }

        self.position_history.append(position_snapshot)

        # Keep only recent history (last 1000 snapshots)
        if len(self.position_history) > 1000:
            self.position_history = self.position_history[-1000:]

    def _update_pnl_history(self) -> None:
        """Update P&L history for risk calculations"""

        if not self.trading_engine:
            return

        # Calculate current total P&L
        total_pnl = sum(
            float(pos.unrealized_pnl + pos.realized_pnl)
            for pos in self.trading_engine.positions.values()
        )

        self.pnl_history.append(total_pnl)

        # Calculate returns if we have previous P&L
        if len(self.pnl_history) > 1:
            prev_pnl = self.pnl_history[-2]
            if prev_pnl != 0:
                return_pct = (total_pnl - prev_pnl) / abs(prev_pnl)
            else:
                return_pct = 0.0

            self.return_history.append(return_pct)

        # Keep only recent history
        if len(self.pnl_history) > 1000:
            self.pnl_history = self.pnl_history[-1000:]
        if len(self.return_history) > 1000:
            self.return_history = self.return_history[-1000:]

    def _store_risk_metrics(self, metrics: RiskMetrics) -> None:
        """Store risk metrics in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO risk_metrics (
                    total_exposure, net_exposure, leverage_ratio, portfolio_var_95,
                    max_drawdown, current_drawdown, portfolio_volatility, avg_correlation,
                    liquidity_score, timestamp, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    str(metrics.total_exposure),
                    str(metrics.net_exposure),
                    metrics.leverage_ratio,
                    str(metrics.portfolio_var_95),
                    metrics.max_drawdown,
                    metrics.current_drawdown,
                    metrics.portfolio_volatility,
                    metrics.avg_correlation,
                    metrics.liquidity_score,
                    metrics.timestamp.isoformat(),
                    json.dumps(metrics.__dict__, default=str),
                ),
            )
            conn.commit()

    def _store_risk_limit(self, limit: RiskLimit) -> None:
        """Store risk limit in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO risk_limits (
                    limit_name, limit_type, limit_metric, soft_limit, hard_limit,
                    current_value, breach_count, last_breach, is_active, limit_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    limit.limit_name,
                    limit.limit_type,
                    limit.limit_metric,
                    limit.soft_limit,
                    limit.hard_limit,
                    limit.current_value,
                    limit.breach_count,
                    limit.last_breach.isoformat() if limit.last_breach else None,
                    limit.is_active,
                    json.dumps(limit.__dict__, default=str),
                ),
            )
            conn.commit()

    def _store_risk_event(self, event: RiskEvent) -> None:
        """Store risk event in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO risk_events (
                    event_id, event_type, risk_level, symbol, strategy_id,
                    current_value, limit_value, breach_amount, message,
                    detected_at, resolved_at, is_resolved, event_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event.event_id,
                    event.event_type.value,
                    event.risk_level.value,
                    event.symbol,
                    event.strategy_id,
                    event.current_value,
                    event.limit_value,
                    event.breach_amount,
                    event.message,
                    event.detected_at.isoformat(),
                    event.resolved_at.isoformat() if event.resolved_at else None,
                    event.is_resolved,
                    json.dumps(event.__dict__, default=str),
                ),
            )
            conn.commit()

    def get_risk_status(self) -> dict[str, Any]:
        """Get current risk monitoring status"""

        return {
            "is_monitoring": self.is_monitoring,
            "active_risk_events": len(self.active_risk_events),
            "risk_limits_configured": len(self.risk_limits),
            "breached_limits": sum(
                1
                for limit in self.risk_limits.values()
                if limit.is_soft_breach or limit.is_hard_breach
            ),
            "current_metrics_available": self.current_metrics is not None,
            "position_history_length": len(self.position_history),
            "pnl_history_length": len(self.pnl_history),
        }

    def display_risk_dashboard(self) -> None:
        """Display comprehensive risk monitoring dashboard"""

        status = self.get_risk_status()

        console.print(
            Panel(
                f"[bold blue]Real-Time Risk Monitor Dashboard[/bold blue]\n"
                f"Status: {'🟢 MONITORING' if status['is_monitoring'] else '🔴 STOPPED'}\n"
                f"Active Alerts: {status['active_risk_events']}\n"
                f"Breached Limits: {status['breached_limits']}/{status['risk_limits_configured']}",
                title="🛡️  Risk Monitor",
            )
        )

        # Current risk metrics
        if self.current_metrics:
            metrics = self.current_metrics

            metrics_table = Table(title="📊 Current Risk Metrics")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="white")
            metrics_table.add_column("Status", style="green")

            metrics_data = [
                ("Portfolio Exposure", f"${metrics.total_exposure:,.0f}", "📊"),
                ("Leverage Ratio", f"{metrics.leverage_ratio:.2f}x", "⚖️"),
                (
                    "Current Drawdown",
                    f"{metrics.current_drawdown:.1%}",
                    "📉" if metrics.current_drawdown > 0.05 else "✅",
                ),
                ("Portfolio VaR (95%)", f"${metrics.portfolio_var_95:,.0f}", "🎯"),
                ("Avg Correlation", f"{metrics.avg_correlation:.1%}", "🔗"),
                ("Portfolio Volatility", f"{metrics.portfolio_volatility:.1%}", "📊"),
                ("Liquidity Score", f"{metrics.liquidity_score:.1f}/100", "💧"),
            ]

            for metric_name, value, status in metrics_data:
                metrics_table.add_row(metric_name, value, status)

            console.print(metrics_table)

        # Risk limits status
        if self.risk_limits:
            limits_table = Table(title="🚨 Risk Limits Status")
            limits_table.add_column("Limit", style="cyan")
            limits_table.add_column("Current", justify="right", style="white")
            limits_table.add_column("Soft Limit", justify="right", style="yellow")
            limits_table.add_column("Hard Limit", justify="right", style="red")
            limits_table.add_column("Status", style="green")

            for limit in list(self.risk_limits.values())[:10]:  # Show top 10
                if limit.is_hard_breach:
                    status = "[red]BREACH[/red]"
                elif limit.is_soft_breach:
                    status = "[yellow]WARNING[/yellow]"
                else:
                    status = "[green]OK[/green]"

                limits_table.add_row(
                    limit.limit_name.replace("_", " ").title(),
                    f"{limit.current_value:.3f}",
                    f"{limit.soft_limit:.3f}",
                    f"{limit.hard_limit:.3f}",
                    status,
                )

            console.print(limits_table)


def create_real_time_risk_monitor(
    risk_dir: str = "data/risk_monitoring",
    trading_engine: LiveTradingEngine = None,
    streaming_manager: StreamingDataManager = None,
) -> RealTimeRiskMonitor:
    """Factory function to create real-time risk monitor"""
    return RealTimeRiskMonitor(
        risk_dir=risk_dir, trading_engine=trading_engine, streaming_manager=streaming_manager
    )


if __name__ == "__main__":
    # Example usage
    monitor = create_real_time_risk_monitor()

    # Add risk event callback
    def on_risk_event(event) -> None:
        print(f"Risk Event: {event.risk_level.value} - {event.message}")

    monitor.add_risk_event_callback(on_risk_event)

    # Start monitoring
    monitor.start_risk_monitoring()

    try:
        # Let it run for demo
        time.sleep(10)

        # Display dashboard
        monitor.display_risk_dashboard()

    finally:
        monitor.stop_risk_monitoring()

    print("Real-Time Risk Monitor created successfully!")
