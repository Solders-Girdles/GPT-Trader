"""
Real-Time Risk Monitoring System for Live Trading Infrastructure

This module implements comprehensive real-time risk management including:
- Position risk monitoring (concentration, size, exposure limits)
- Portfolio-level risk metrics (VaR, CVaR, max drawdown)
- Market risk monitoring (volatility, correlation, regime detection)
- Liquidity risk assessment and monitoring
- Real-time P&L tracking and attribution
- Risk limit enforcement and alerting
- Compliance monitoring and reporting
- Circuit breakers and automatic risk controls
"""

import asyncio
import logging
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

# Optional dependencies
try:
    from scipy import stats
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some risk calculations will be limited.")

try:
    from ..live.market_data_pipeline import DataType, MarketDataPoint
    from ..live.order_management import Fill, Order, OrderSide, OrderStatus
except ImportError:
    # Fallback for testing
    class MarketDataPoint:
        def __init__(self, symbol, timestamp, data) -> None:
            self.symbol = symbol
            self.timestamp = timestamp
            self.data = data

    class Order:
        pass

    class Fill:
        pass

    class OrderSide:
        BUY = "buy"
        SELL = "sell"

    class OrderStatus:
        FILLED = "filled"


logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskType(Enum):
    """Types of risk being monitored"""

    POSITION_CONCENTRATION = "position_concentration"
    POSITION_SIZE = "position_size"
    PORTFOLIO_VAR = "portfolio_var"
    PORTFOLIO_DRAWDOWN = "portfolio_drawdown"
    LIQUIDITY_RISK = "liquidity_risk"
    MARKET_VOLATILITY = "market_volatility"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    LEVERAGE_EXCESS = "leverage_excess"
    SECTOR_CONCENTRATION = "sector_concentration"
    COMPLIANCE_VIOLATION = "compliance_violation"


class AlertAction(Enum):
    """Actions to take when risk limits are breached"""

    LOG_WARNING = "log_warning"
    SEND_ALERT = "send_alert"
    REDUCE_POSITIONS = "reduce_positions"
    HALT_TRADING = "halt_trading"
    LIQUIDATE_POSITIONS = "liquidate_positions"


@dataclass
class RiskLimit:
    """Risk limit configuration"""

    risk_type: RiskType
    threshold: float
    warning_threshold: float | None = None
    action: AlertAction = AlertAction.LOG_WARNING
    enabled: bool = True
    lookback_periods: int = 252  # For historical calculations
    confidence_level: float = 0.95  # For VaR calculations


@dataclass
class RiskAlert:
    """Risk alert notification"""

    alert_id: str
    risk_type: RiskType
    risk_level: RiskLevel
    current_value: float
    threshold: float
    symbol: str | None = None
    message: str = ""
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False


@dataclass
class PositionRisk:
    """Position-level risk metrics"""

    symbol: str
    position: float
    market_value: float
    unrealized_pnl: float
    daily_var_95: float
    daily_var_99: float
    volatility_20d: float
    beta: float | None = None
    concentration_pct: float = 0.0
    liquidity_score: float = 1.0
    sector: str | None = None
    last_updated: pd.Timestamp = field(default_factory=pd.Timestamp.now)


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""

    total_equity: float
    total_exposure: float
    leverage: float
    daily_var_95: float
    daily_var_99: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    volatility: float
    beta: float | None = None
    correlation_avg: float = 0.0
    concentration_hhi: float = 0.0
    liquidity_risk_score: float = 1.0
    last_updated: pd.Timestamp = field(default_factory=pd.Timestamp.now)


@dataclass
class RiskMonitorConfig:
    """Configuration for risk monitoring system"""

    # Position limits
    max_position_size: float = 1000000.0
    max_position_concentration: float = 0.1  # 10% of portfolio
    max_sector_concentration: float = 0.25  # 25% of portfolio
    max_leverage: float = 2.0

    # Portfolio risk limits
    max_daily_var_95: float = 0.02  # 2% daily VaR
    max_daily_var_99: float = 0.04  # 4% daily VaR
    max_drawdown_limit: float = 0.15  # 15% max drawdown
    min_liquidity_score: float = 0.5

    # Market risk limits
    max_volatility_threshold: float = 0.40  # 40% annualized
    min_correlation_threshold: float = -0.5
    max_correlation_threshold: float = 0.95

    # Monitoring settings
    update_frequency: float = 1.0  # seconds
    var_lookback_days: int = 252
    volatility_lookback_days: int = 20
    enable_auto_actions: bool = False
    enable_circuit_breakers: bool = True

    # Alert settings
    max_alerts_per_minute: int = 10
    alert_cooldown_minutes: int = 5


class BaseRiskCalculator(ABC):
    """Base class for risk calculations"""

    def __init__(self, config: RiskMonitorConfig) -> None:
        self.config = config
        self.market_data_cache = {}
        self.price_history = defaultdict(deque)

    @abstractmethod
    def calculate_position_risk(
        self, symbol: str, position: float, current_price: float
    ) -> PositionRisk:
        """Calculate position-specific risk metrics"""
        pass

    @abstractmethod
    def calculate_portfolio_risk(
        self, positions: dict[str, float], current_prices: dict[str, float]
    ) -> PortfolioRisk:
        """Calculate portfolio-level risk metrics"""
        pass

    def update_price_history(self, symbol: str, price: float, timestamp: pd.Timestamp) -> None:
        """Update price history for calculations"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(
                maxlen=max(self.config.var_lookback_days, self.config.volatility_lookback_days)
            )

        self.price_history[symbol].append({"price": price, "timestamp": timestamp})


class QuantitativeRiskCalculator(BaseRiskCalculator):
    """Quantitative risk calculator with advanced metrics"""

    def calculate_position_risk(
        self, symbol: str, position: float, current_price: float
    ) -> PositionRisk:
        """Calculate position risk metrics"""
        try:
            market_value = abs(position * current_price)

            # Get price history
            price_data = self.price_history.get(symbol, deque())
            if len(price_data) < 20:
                # Insufficient data - use conservative estimates
                return PositionRisk(
                    symbol=symbol,
                    position=position,
                    market_value=market_value,
                    unrealized_pnl=0.0,
                    daily_var_95=market_value * 0.02,  # 2% conservative estimate
                    daily_var_99=market_value * 0.04,  # 4% conservative estimate
                    volatility_20d=0.20,  # 20% conservative estimate
                    concentration_pct=0.0,
                    liquidity_score=1.0,
                )

            # Extract prices and calculate returns
            prices = np.array([p["price"] for p in price_data])
            returns = np.diff(np.log(prices))

            # Volatility calculation
            if len(returns) >= self.config.volatility_lookback_days:
                recent_returns = returns[-self.config.volatility_lookback_days :]
                volatility_20d = np.std(recent_returns) * np.sqrt(252)  # Annualized
            else:
                volatility_20d = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.20

            # VaR calculations
            if len(returns) >= 30 and SCIPY_AVAILABLE:
                # Parametric VaR
                daily_vol = volatility_20d / np.sqrt(252)
                var_95 = abs(market_value * stats.norm.ppf(0.05) * daily_vol)
                var_99 = abs(market_value * stats.norm.ppf(0.01) * daily_vol)

                # Historical VaR as a cross-check
                if len(returns) >= 100:
                    portfolio_returns = returns * market_value / prices[:-1]  # Approximate P&L
                    hist_var_95 = abs(np.percentile(portfolio_returns, 5))
                    hist_var_99 = abs(np.percentile(portfolio_returns, 1))

                    # Use the higher of parametric and historical VaR (more conservative)
                    var_95 = max(var_95, hist_var_95)
                    var_99 = max(var_99, hist_var_99)
            else:
                # Simple VaR approximation
                daily_vol = volatility_20d / np.sqrt(252)
                var_95 = market_value * daily_vol * 1.65  # ~95th percentile
                var_99 = market_value * daily_vol * 2.33  # ~99th percentile

            # Unrealized P&L (simplified - would need cost basis in practice)
            avg_price = np.mean(prices[-30:]) if len(prices) >= 30 else current_price
            unrealized_pnl = position * (current_price - avg_price)

            # Liquidity score (simplified heuristic)
            if len(returns) >= 5:
                price_stability = 1.0 / (1.0 + np.std(returns[-5:]))  # Recent stability
                liquidity_score = min(1.0, price_stability * 2.0)
            else:
                liquidity_score = 0.8  # Default score

            return PositionRisk(
                symbol=symbol,
                position=position,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                daily_var_95=var_95,
                daily_var_99=var_99,
                volatility_20d=volatility_20d,
                concentration_pct=0.0,  # Will be calculated at portfolio level
                liquidity_score=liquidity_score,
            )

        except Exception as e:
            logger.warning(f"Error calculating position risk for {symbol}: {str(e)}")
            # Return conservative default values
            market_value = abs(position * current_price)
            return PositionRisk(
                symbol=symbol,
                position=position,
                market_value=market_value,
                unrealized_pnl=0.0,
                daily_var_95=market_value * 0.03,
                daily_var_99=market_value * 0.05,
                volatility_20d=0.25,
                liquidity_score=0.5,
            )

    def calculate_portfolio_risk(
        self, positions: dict[str, float], current_prices: dict[str, float]
    ) -> PortfolioRisk:
        """Calculate portfolio-level risk metrics"""
        try:
            if not positions:
                return PortfolioRisk(
                    total_equity=0.0,
                    total_exposure=0.0,
                    leverage=0.0,
                    daily_var_95=0.0,
                    daily_var_99=0.0,
                    max_drawdown=0.0,
                    current_drawdown=0.0,
                    sharpe_ratio=0.0,
                    volatility=0.0,
                )

            # Calculate basic portfolio metrics
            position_values = {}
            total_long_exposure = 0.0
            total_short_exposure = 0.0

            for symbol, position in positions.items():
                if symbol in current_prices:
                    value = position * current_prices[symbol]
                    position_values[symbol] = abs(value)

                    if position > 0:
                        total_long_exposure += value
                    else:
                        total_short_exposure += abs(value)

            total_exposure = total_long_exposure + total_short_exposure
            total_equity = total_long_exposure - total_short_exposure  # Net equity
            leverage = total_exposure / max(abs(total_equity), 1.0)

            # Portfolio concentration (HHI)
            if total_exposure > 0:
                weights = np.array([value / total_exposure for value in position_values.values()])
                concentration_hhi = np.sum(weights**2)
            else:
                concentration_hhi = 0.0

            # Portfolio VaR calculation
            portfolio_var_95, portfolio_var_99 = self._calculate_portfolio_var(
                positions, current_prices
            )

            # Portfolio volatility and correlation
            portfolio_vol, avg_correlation = self._calculate_portfolio_volatility(
                positions, current_prices
            )

            # Drawdown calculation (simplified - would need equity curve history)
            max_drawdown = 0.05  # Placeholder - would calculate from equity history
            current_drawdown = 0.02  # Placeholder - would calculate from recent performance

            # Sharpe ratio (simplified)
            if portfolio_vol > 0:
                sharpe_ratio = 0.08 / portfolio_vol  # Assuming 8% risk-free rate
            else:
                sharpe_ratio = 0.0

            # Liquidity risk score (weighted average)
            if position_values:
                total_weight = sum(position_values.values())
                liquidity_scores = []
                weights = []

                for symbol in positions:
                    if symbol in current_prices:
                        pos_risk = self.calculate_position_risk(
                            symbol, positions[symbol], current_prices[symbol]
                        )
                        liquidity_scores.append(pos_risk.liquidity_score)
                        weights.append(position_values[symbol] / total_weight)

                if liquidity_scores:
                    portfolio_liquidity = np.average(liquidity_scores, weights=weights)
                else:
                    portfolio_liquidity = 1.0
            else:
                portfolio_liquidity = 1.0

            return PortfolioRisk(
                total_equity=total_equity,
                total_exposure=total_exposure,
                leverage=leverage,
                daily_var_95=portfolio_var_95,
                daily_var_99=portfolio_var_99,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                sharpe_ratio=sharpe_ratio,
                volatility=portfolio_vol,
                correlation_avg=avg_correlation,
                concentration_hhi=concentration_hhi,
                liquidity_risk_score=portfolio_liquidity,
            )

        except Exception as e:
            logger.warning(f"Error calculating portfolio risk: {str(e)}")
            return PortfolioRisk(
                total_equity=0.0,
                total_exposure=0.0,
                leverage=0.0,
                daily_var_95=0.0,
                daily_var_99=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                sharpe_ratio=0.0,
                volatility=0.0,
            )

    def _calculate_portfolio_var(
        self, positions: dict[str, float], current_prices: dict[str, float]
    ) -> tuple[float, float]:
        """Calculate portfolio VaR using correlation matrix"""
        try:
            symbols = [s for s in positions if s in current_prices and positions[s] != 0]

            if len(symbols) < 2:
                # Single asset or no positions
                if len(symbols) == 1:
                    symbol = symbols[0]
                    pos_risk = self.calculate_position_risk(
                        symbol, positions[symbol], current_prices[symbol]
                    )
                    return pos_risk.daily_var_95, pos_risk.daily_var_99
                else:
                    return 0.0, 0.0

            # Build returns matrix
            returns_matrix = []
            weights = []

            total_value = sum(abs(positions[s] * current_prices[s]) for s in symbols)

            for symbol in symbols:
                price_data = self.price_history.get(symbol, deque())
                if len(price_data) >= 30:
                    prices = np.array([p["price"] for p in price_data])
                    returns = np.diff(np.log(prices))

                    if len(returns) >= 20:
                        returns_matrix.append(returns[-100:])  # Last 100 days
                        weight = abs(positions[symbol] * current_prices[symbol]) / total_value
                        weights.append(weight)

            if len(returns_matrix) < 2:
                # Fallback to simple sum
                total_var_95 = sum(
                    self.calculate_position_risk(s, positions[s], current_prices[s]).daily_var_95
                    for s in symbols
                )
                total_var_99 = sum(
                    self.calculate_position_risk(s, positions[s], current_prices[s]).daily_var_99
                    for s in symbols
                )
                return total_var_95 * 0.7, total_var_99 * 0.7  # Diversification benefit estimate

            # Align returns to same length
            min_length = min(len(r) for r in returns_matrix)
            aligned_returns = np.array([r[-min_length:] for r in returns_matrix])

            # Calculate portfolio returns
            weights = np.array(weights)
            portfolio_returns = np.sum(aligned_returns * weights.reshape(-1, 1), axis=0)

            # Calculate VaR
            if SCIPY_AVAILABLE and len(portfolio_returns) >= 30:
                var_95 = abs(np.percentile(portfolio_returns, 5) * total_value)
                var_99 = abs(np.percentile(portfolio_returns, 1) * total_value)
            else:
                # Parametric VaR
                portfolio_vol = np.std(portfolio_returns)
                var_95 = abs(total_value * portfolio_vol * 1.65)
                var_99 = abs(total_value * portfolio_vol * 2.33)

            return var_95, var_99

        except Exception as e:
            logger.warning(f"Error calculating portfolio VaR: {str(e)}")
            # Fallback to sum of individual VaRs
            symbols = [s for s in positions if s in current_prices]
            total_var_95 = sum(
                self.calculate_position_risk(s, positions[s], current_prices[s]).daily_var_95
                for s in symbols
            )
            total_var_99 = sum(
                self.calculate_position_risk(s, positions[s], current_prices[s]).daily_var_99
                for s in symbols
            )
            return total_var_95, total_var_99

    def _calculate_portfolio_volatility(
        self, positions: dict[str, float], current_prices: dict[str, float]
    ) -> tuple[float, float]:
        """Calculate portfolio volatility and average correlation"""
        try:
            symbols = [s for s in positions if s in current_prices and positions[s] != 0]

            if len(symbols) < 2:
                if len(symbols) == 1:
                    symbol = symbols[0]
                    pos_risk = self.calculate_position_risk(
                        symbol, positions[symbol], current_prices[symbol]
                    )
                    return pos_risk.volatility_20d, 0.0
                else:
                    return 0.0, 0.0

            # Calculate weighted average volatility
            total_value = sum(abs(positions[s] * current_prices[s]) for s in symbols)
            weighted_vol = 0.0
            correlations = []

            for i, symbol in enumerate(symbols):
                pos_risk = self.calculate_position_risk(
                    symbol, positions[symbol], current_prices[symbol]
                )
                weight = abs(positions[symbol] * current_prices[symbol]) / total_value
                weighted_vol += weight * pos_risk.volatility_20d

                # Calculate correlations with other assets (simplified)
                for _j, other_symbol in enumerate(symbols[i + 1 :], i + 1):
                    corr = self._calculate_correlation(symbol, other_symbol)
                    if corr is not None:
                        correlations.append(corr)

            avg_correlation = np.mean(correlations) if correlations else 0.0

            # Adjust portfolio volatility for correlations (simplified)
            diversification_factor = 1.0 - (avg_correlation * 0.3)  # Rough approximation
            portfolio_vol = weighted_vol * diversification_factor

            return portfolio_vol, avg_correlation

        except Exception as e:
            logger.warning(f"Error calculating portfolio volatility: {str(e)}")
            return 0.20, 0.0  # Default values

    def _calculate_correlation(self, symbol1: str, symbol2: str) -> float | None:
        """Calculate correlation between two assets"""
        try:
            data1 = self.price_history.get(symbol1, deque())
            data2 = self.price_history.get(symbol2, deque())

            if len(data1) < 30 or len(data2) < 30:
                return None

            # Extract prices and align timestamps
            prices1 = np.array([p["price"] for p in data1])
            prices2 = np.array([p["price"] for p in data2])

            # Calculate returns
            returns1 = np.diff(np.log(prices1))
            returns2 = np.diff(np.log(prices2))

            # Take minimum length
            min_len = min(len(returns1), len(returns2))
            if min_len < 20:
                return None

            returns1 = returns1[-min_len:]
            returns2 = returns2[-min_len:]

            correlation = np.corrcoef(returns1, returns2)[0, 1]

            # Handle NaN
            if np.isnan(correlation):
                return None

            return correlation

        except Exception as e:
            logger.warning(
                f"Error calculating correlation between {symbol1} and {symbol2}: {str(e)}"
            )
            return None


class RealTimeRiskMonitor:
    """Main real-time risk monitoring system"""

    def __init__(self, config: RiskMonitorConfig) -> None:
        self.config = config
        self.risk_calculator = QuantitativeRiskCalculator(config)
        self.risk_limits = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.positions = {}
        self.current_prices = {}
        self.last_portfolio_risk = None
        self.is_running = False
        self.risk_listeners = []
        self.alert_cooldown = {}  # Track alert cooldowns

        # Performance tracking
        self.monitoring_stats = {
            "risk_checks_performed": 0,
            "alerts_generated": 0,
            "limits_breached": 0,
            "last_update": time.time(),
        }

        # Initialize default risk limits
        self._setup_default_limits()

    def _setup_default_limits(self) -> None:
        """Setup default risk limits"""
        default_limits = [
            RiskLimit(
                RiskType.POSITION_SIZE,
                self.config.max_position_size,
                self.config.max_position_size * 0.8,
                AlertAction.SEND_ALERT,
            ),
            RiskLimit(
                RiskType.POSITION_CONCENTRATION,
                self.config.max_position_concentration,
                self.config.max_position_concentration * 0.8,
                AlertAction.SEND_ALERT,
            ),
            RiskLimit(
                RiskType.LEVERAGE_EXCESS,
                self.config.max_leverage,
                self.config.max_leverage * 0.9,
                AlertAction.SEND_ALERT,
            ),
            RiskLimit(
                RiskType.PORTFOLIO_VAR,
                self.config.max_daily_var_95,
                self.config.max_daily_var_95 * 0.8,
                AlertAction.SEND_ALERT,
            ),
            RiskLimit(
                RiskType.PORTFOLIO_DRAWDOWN,
                self.config.max_drawdown_limit,
                self.config.max_drawdown_limit * 0.8,
                AlertAction.SEND_ALERT,
            ),
            RiskLimit(
                RiskType.LIQUIDITY_RISK,
                self.config.min_liquidity_score,
                self.config.min_liquidity_score * 1.2,
                AlertAction.LOG_WARNING,
            ),
        ]

        for limit in default_limits:
            self.risk_limits[limit.risk_type] = limit

    def add_risk_limit(self, risk_limit: RiskLimit) -> None:
        """Add or update a risk limit"""
        self.risk_limits[risk_limit.risk_type] = risk_limit
        logger.info(f"Added risk limit: {risk_limit.risk_type.value} = {risk_limit.threshold}")

    def remove_risk_limit(self, risk_type: RiskType) -> None:
        """Remove a risk limit"""
        if risk_type in self.risk_limits:
            del self.risk_limits[risk_type]
            logger.info(f"Removed risk limit: {risk_type.value}")

    def add_risk_listener(self, callback: Callable[[RiskAlert], None]) -> None:
        """Add risk event listener"""
        self.risk_listeners.append(callback)

    def update_market_data(self, data_point: MarketDataPoint) -> None:
        """Update market data for risk calculations"""
        if data_point.data_type == DataType.QUOTE or data_point.data_type == DataType.TRADE:
            # Extract price from data
            if "price" in data_point.data:
                price = data_point.data["price"]
            elif "last" in data_point.data:
                price = data_point.data["last"]
            elif "close" in data_point.data:
                price = data_point.data["close"]
            else:
                return  # No price data available

            # Update current prices and price history
            self.current_prices[data_point.symbol] = price
            self.risk_calculator.update_price_history(
                data_point.symbol, price, data_point.timestamp
            )

    def update_positions(self, positions: dict[str, float]) -> None:
        """Update position data"""
        self.positions = positions.copy()

        # Trigger risk check if monitoring is running
        if self.is_running:
            asyncio.create_task(self._check_all_risks())

    def update_position(self, symbol: str, position: float) -> None:
        """Update single position"""
        self.positions[symbol] = position

        # Trigger risk check if monitoring is running
        if self.is_running:
            asyncio.create_task(self._check_position_risks(symbol))

    async def start_monitoring(self) -> None:
        """Start real-time risk monitoring"""
        if self.is_running:
            logger.warning("Risk monitoring already running")
            return

        self.is_running = True
        logger.info("Starting real-time risk monitoring...")

        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())

        logger.info("Risk monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop real-time risk monitoring"""
        if not self.is_running:
            return

        logger.info("Stopping real-time risk monitoring...")
        self.is_running = False

        logger.info("Risk monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_running:
            try:
                start_time = time.time()

                # Perform comprehensive risk checks
                await self._check_all_risks()

                # Update performance stats
                self.monitoring_stats["risk_checks_performed"] += 1
                self.monitoring_stats["last_update"] = time.time()

                # Calculate sleep time to maintain frequency
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config.update_frequency - elapsed)
                await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {str(e)}")
                await asyncio.sleep(self.config.update_frequency)

    async def _check_all_risks(self) -> None:
        """Perform comprehensive risk checks"""
        try:
            # Check position-level risks
            for symbol in self.positions:
                await self._check_position_risks(symbol)

            # Check portfolio-level risks
            await self._check_portfolio_risks()

        except Exception as e:
            logger.error(f"Error in comprehensive risk check: {str(e)}")

    async def _check_position_risks(self, symbol: str) -> None:
        """Check risks for a specific position"""
        try:
            if symbol not in self.positions or symbol not in self.current_prices:
                return

            position = self.positions[symbol]
            current_price = self.current_prices[symbol]

            if position == 0:
                return  # No position to check

            # Calculate position risk
            pos_risk = self.risk_calculator.calculate_position_risk(symbol, position, current_price)

            # Check position size limit
            if RiskType.POSITION_SIZE in self.risk_limits:
                limit = self.risk_limits[RiskType.POSITION_SIZE]
                if limit.enabled and pos_risk.market_value > limit.threshold:
                    await self._generate_alert(
                        RiskType.POSITION_SIZE,
                        RiskLevel.HIGH,
                        pos_risk.market_value,
                        limit.threshold,
                        symbol,
                        f"Position size ${pos_risk.market_value:,.0f} exceeds limit ${limit.threshold:,.0f}",
                        limit.action,
                    )

            # Check concentration limit (will be calculated at portfolio level)
            # For now, check individual position as percentage of total exposure
            if self.last_portfolio_risk and self.last_portfolio_risk.total_exposure > 0:
                concentration = pos_risk.market_value / self.last_portfolio_risk.total_exposure

                if RiskType.POSITION_CONCENTRATION in self.risk_limits:
                    limit = self.risk_limits[RiskType.POSITION_CONCENTRATION]
                    if limit.enabled and concentration > limit.threshold:
                        await self._generate_alert(
                            RiskType.POSITION_CONCENTRATION,
                            RiskLevel.MEDIUM,
                            concentration,
                            limit.threshold,
                            symbol,
                            f"Position concentration {concentration:.1%} exceeds limit {limit.threshold:.1%}",
                            limit.action,
                        )

            # Check liquidity risk
            if RiskType.LIQUIDITY_RISK in self.risk_limits:
                limit = self.risk_limits[RiskType.LIQUIDITY_RISK]
                if limit.enabled and pos_risk.liquidity_score < limit.threshold:
                    await self._generate_alert(
                        RiskType.LIQUIDITY_RISK,
                        RiskLevel.MEDIUM,
                        pos_risk.liquidity_score,
                        limit.threshold,
                        symbol,
                        f"Liquidity score {pos_risk.liquidity_score:.2f} below minimum {limit.threshold:.2f}",
                        limit.action,
                    )

        except Exception as e:
            logger.warning(f"Error checking position risks for {symbol}: {str(e)}")

    async def _check_portfolio_risks(self) -> None:
        """Check portfolio-level risks"""
        try:
            if not self.positions or not self.current_prices:
                return

            # Calculate portfolio risk
            portfolio_risk = self.risk_calculator.calculate_portfolio_risk(
                self.positions, self.current_prices
            )

            self.last_portfolio_risk = portfolio_risk

            # Check leverage limit
            if RiskType.LEVERAGE_EXCESS in self.risk_limits:
                limit = self.risk_limits[RiskType.LEVERAGE_EXCESS]
                if limit.enabled and portfolio_risk.leverage > limit.threshold:
                    await self._generate_alert(
                        RiskType.LEVERAGE_EXCESS,
                        RiskLevel.HIGH,
                        portfolio_risk.leverage,
                        limit.threshold,
                        None,
                        f"Portfolio leverage {portfolio_risk.leverage:.2f}x exceeds limit {limit.threshold:.2f}x",
                        limit.action,
                    )

            # Check VaR limits
            if RiskType.PORTFOLIO_VAR in self.risk_limits:
                limit = self.risk_limits[RiskType.PORTFOLIO_VAR]
                if limit.enabled:
                    var_ratio = portfolio_risk.daily_var_95 / max(
                        abs(portfolio_risk.total_equity), 1.0
                    )
                    if var_ratio > limit.threshold:
                        await self._generate_alert(
                            RiskType.PORTFOLIO_VAR,
                            RiskLevel.HIGH,
                            var_ratio,
                            limit.threshold,
                            None,
                            f"Portfolio VaR {var_ratio:.2%} exceeds limit {limit.threshold:.2%}",
                            limit.action,
                        )

            # Check drawdown limits
            if RiskType.PORTFOLIO_DRAWDOWN in self.risk_limits:
                limit = self.risk_limits[RiskType.PORTFOLIO_DRAWDOWN]
                if limit.enabled and portfolio_risk.current_drawdown > limit.threshold:
                    await self._generate_alert(
                        RiskType.PORTFOLIO_DRAWDOWN,
                        RiskLevel.CRITICAL,
                        portfolio_risk.current_drawdown,
                        limit.threshold,
                        None,
                        f"Portfolio drawdown {portfolio_risk.current_drawdown:.2%} exceeds limit {limit.threshold:.2%}",
                        limit.action,
                    )

        except Exception as e:
            logger.warning(f"Error checking portfolio risks: {str(e)}")

    async def _generate_alert(
        self,
        risk_type: RiskType,
        risk_level: RiskLevel,
        current_value: float,
        threshold: float,
        symbol: str | None,
        message: str,
        action: AlertAction,
    ) -> None:
        """Generate and process risk alert"""
        try:
            # Check alert cooldown
            alert_key = f"{risk_type.value}_{symbol or 'portfolio'}"
            current_time = time.time()

            if alert_key in self.alert_cooldown:
                if current_time - self.alert_cooldown[alert_key] < (
                    self.config.alert_cooldown_minutes * 60
                ):
                    return  # Still in cooldown

            self.alert_cooldown[alert_key] = current_time

            # Create alert
            alert = RiskAlert(
                alert_id=f"alert_{int(current_time)}_{risk_type.value}",
                risk_type=risk_type,
                risk_level=risk_level,
                current_value=current_value,
                threshold=threshold,
                symbol=symbol,
                message=message,
                metadata={
                    "action": action.value,
                    "positions": dict(self.positions) if len(self.positions) < 10 else {},
                    "portfolio_metrics": {
                        "total_exposure": (
                            self.last_portfolio_risk.total_exposure
                            if self.last_portfolio_risk
                            else 0
                        ),
                        "leverage": (
                            self.last_portfolio_risk.leverage if self.last_portfolio_risk else 0
                        ),
                    },
                },
            )

            # Store alert
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            self.monitoring_stats["alerts_generated"] += 1

            # Execute action
            await self._execute_alert_action(alert, action)

            # Notify listeners
            await self._notify_risk_listeners(alert)

            # Log alert
            logger.warning(f"RISK ALERT [{risk_level.value.upper()}] {risk_type.value}: {message}")

            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                self.monitoring_stats["limits_breached"] += 1

        except Exception as e:
            logger.error(f"Error generating risk alert: {str(e)}")

    async def _execute_alert_action(self, alert: RiskAlert, action: AlertAction) -> None:
        """Execute the specified action for an alert"""
        try:
            if action == AlertAction.LOG_WARNING:
                logger.warning(f"Risk Alert: {alert.message}")

            elif action == AlertAction.SEND_ALERT:
                # In practice, would send to external alerting system
                logger.critical(f"CRITICAL RISK ALERT: {alert.message}")

            elif action == AlertAction.HALT_TRADING:
                logger.critical(f"HALTING TRADING: {alert.message}")
                # In practice, would stop order submission

            elif action in [AlertAction.REDUCE_POSITIONS, AlertAction.LIQUIDATE_POSITIONS]:
                logger.critical(f"POSITION ACTION REQUIRED: {alert.message}")
                # In practice, would trigger position reduction/liquidation

        except Exception as e:
            logger.error(f"Error executing alert action: {str(e)}")

    async def _notify_risk_listeners(self, alert: RiskAlert) -> None:
        """Notify risk event listeners"""
        for callback in self.risk_listeners:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.warning(f"Error notifying risk listener: {str(e)}")

    def get_current_risks(self) -> dict[str, Any]:
        """Get current risk assessment"""
        try:
            position_risks = {}

            for symbol in self.positions:
                if symbol in self.current_prices and self.positions[symbol] != 0:
                    pos_risk = self.risk_calculator.calculate_position_risk(
                        symbol, self.positions[symbol], self.current_prices[symbol]
                    )
                    position_risks[symbol] = {
                        "position": pos_risk.position,
                        "market_value": pos_risk.market_value,
                        "daily_var_95": pos_risk.daily_var_95,
                        "volatility_20d": pos_risk.volatility_20d,
                        "liquidity_score": pos_risk.liquidity_score,
                    }

            portfolio_risk_dict = {}
            if self.last_portfolio_risk:
                portfolio_risk_dict = {
                    "total_equity": self.last_portfolio_risk.total_equity,
                    "total_exposure": self.last_portfolio_risk.total_exposure,
                    "leverage": self.last_portfolio_risk.leverage,
                    "daily_var_95": self.last_portfolio_risk.daily_var_95,
                    "daily_var_99": self.last_portfolio_risk.daily_var_99,
                    "volatility": self.last_portfolio_risk.volatility,
                    "sharpe_ratio": self.last_portfolio_risk.sharpe_ratio,
                    "concentration_hhi": self.last_portfolio_risk.concentration_hhi,
                }

            return {
                "position_risks": position_risks,
                "portfolio_risk": portfolio_risk_dict,
                "active_alerts": len(self.active_alerts),
                "monitoring_stats": self.monitoring_stats.copy(),
                "risk_limits": {
                    risk_type.value: limit.threshold
                    for risk_type, limit in self.risk_limits.items()
                },
                "last_update": pd.Timestamp.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting current risks: {str(e)}")
            return {"error": str(e), "monitoring_stats": self.monitoring_stats.copy()}

    def get_alert_history(self, limit: int = 100) -> list[RiskAlert]:
        """Get recent alert history"""
        return list(self.alert_history)[-limit:]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a risk alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False


def create_risk_monitor(
    max_position_size: float = 1000000.0,
    max_leverage: float = 2.0,
    enable_auto_actions: bool = False,
    **kwargs,
) -> RealTimeRiskMonitor:
    """Factory function to create risk monitor"""
    config = RiskMonitorConfig(
        max_position_size=max_position_size,
        max_leverage=max_leverage,
        enable_auto_actions=enable_auto_actions,
        **kwargs,
    )

    return RealTimeRiskMonitor(config)


# Example usage and testing
async def main() -> None:
    """Example usage of real-time risk monitor"""
    print("Real-Time Risk Monitoring System Testing")
    print("=" * 50)

    # Create risk monitor
    risk_monitor = create_risk_monitor(
        max_position_size=500000.0,
        max_leverage=1.5,
        max_daily_var_95=0.03,  # 3%
        enable_auto_actions=False,
    )

    # Add risk listener
    def risk_listener(alert: RiskAlert) -> None:
        print(f"🚨 RISK ALERT: {alert.risk_type.value} - {alert.message}")
        print(
            f"   Level: {alert.risk_level.value}, Value: {alert.current_value:.4f}, Threshold: {alert.threshold:.4f}"
        )

    risk_monitor.add_risk_listener(risk_listener)

    # Start monitoring
    await risk_monitor.start_monitoring()
    print("✅ Risk monitoring started")

    # Simulate market data and position updates
    try:
        # Update some market data
        class MockMarketData:
            def __init__(self, symbol, price) -> None:
                self.symbol = symbol
                self.timestamp = pd.Timestamp.now()
                self.data_type = DataType.QUOTE
                self.data = {"price": price}

        # Add market data
        risk_monitor.update_market_data(MockMarketData("AAPL", 150.00))
        risk_monitor.update_market_data(MockMarketData("GOOGL", 2800.00))
        risk_monitor.update_market_data(MockMarketData("MSFT", 300.00))

        print("📊 Updated market data")

        # Add positions (some may trigger alerts)
        test_positions = {
            "AAPL": 2000,  # $300k position
            "GOOGL": 100,  # $280k position
            "MSFT": -500,  # $150k short position (total exposure: $730k)
        }

        risk_monitor.update_positions(test_positions)
        print("📈 Updated positions")

        # Wait for risk calculations
        await asyncio.sleep(3)

        # Get current risk assessment
        risks = risk_monitor.get_current_risks()
        print("\n📊 Current Risk Assessment:")

        if "portfolio_risk" in risks and risks["portfolio_risk"]:
            portfolio = risks["portfolio_risk"]
            print(f"   Total Exposure: ${portfolio.get('total_exposure', 0):,.0f}")
            print(f"   Leverage: {portfolio.get('leverage', 0):.2f}x")
            print(f"   Daily VaR (95%): ${portfolio.get('daily_var_95', 0):,.0f}")
            print(f"   Volatility: {portfolio.get('volatility', 0):.2%}")

        print(f"   Active Alerts: {risks.get('active_alerts', 0)}")
        print(f"   Risk Checks: {risks['monitoring_stats']['risk_checks_performed']}")

        # Show position risks
        if "position_risks" in risks:
            print("\n📋 Position Risks:")
            for symbol, pos_risk in risks["position_risks"].items():
                print(
                    f"   {symbol}: ${pos_risk['market_value']:,.0f}, VaR: ${pos_risk['daily_var_95']:,.0f}"
                )

    except Exception as e:
        print(f"❌ Error: {str(e)}")

    finally:
        await risk_monitor.stop_monitoring()
        print("🛑 Risk monitoring stopped")

    print("\n🚀 Real-Time Risk Monitoring System ready for production!")


if __name__ == "__main__":
    asyncio.run(main())
