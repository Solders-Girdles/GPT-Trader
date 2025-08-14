"""
Risk Metrics Calculation Engine
Phase 3, Week 3: RISK-002, RISK-003, RISK-004
Comprehensive risk metrics including VaR, CVaR, and exposure aggregation
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)


class RiskMetricType(Enum):
    """Types of risk metrics"""
    VAR = "value_at_risk"
    CVAR = "conditional_value_at_risk"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    EXPOSURE = "exposure"
    CONCENTRATION = "concentration"
    CORRELATION = "correlation"
    BETA = "beta"
    TRACKING_ERROR = "tracking_error"


class CalculationMethod(Enum):
    """VaR/CVaR calculation methods"""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"


class TimeHorizon(Enum):
    """Risk calculation time horizons"""
    DAILY = 1
    WEEKLY = 5
    MONTHLY = 21
    QUARTERLY = 63
    YEARLY = 252


@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    timestamp: datetime
    
    # Value at Risk
    var_95: Optional[float] = None
    var_99: Optional[float] = None
    var_995: Optional[float] = None
    
    # Conditional Value at Risk (Expected Shortfall)
    cvar_95: Optional[float] = None
    cvar_99: Optional[float] = None
    cvar_995: Optional[float] = None
    
    # Drawdown metrics
    current_drawdown: Optional[float] = None
    max_drawdown: Optional[float] = None
    drawdown_duration: Optional[int] = None
    
    # Risk-adjusted returns
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    information_ratio: Optional[float] = None
    
    # Exposure metrics
    gross_exposure: Optional[float] = None
    net_exposure: Optional[float] = None
    long_exposure: Optional[float] = None
    short_exposure: Optional[float] = None
    
    # Concentration metrics
    concentration_ratio: Optional[float] = None
    herfindahl_index: Optional[float] = None
    top_5_concentration: Optional[float] = None
    
    # Greeks (if applicable)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None
    
    # Metadata
    calculation_method: Optional[str] = None
    confidence_level: Optional[float] = None
    time_horizon: Optional[int] = None
    n_observations: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'cvar_99': self.cvar_99,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'gross_exposure': self.gross_exposure,
            'net_exposure': self.net_exposure
        }


@dataclass
class Position:
    """Single position for risk calculations"""
    symbol: str
    quantity: float
    current_price: float
    cost_basis: float
    position_type: str = "long"  # long or short
    asset_class: str = "equity"
    sector: Optional[str] = None
    
    @property
    def market_value(self) -> float:
        """Calculate current market value"""
        return self.quantity * self.current_price
    
    @property
    def pnl(self) -> float:
        """Calculate P&L"""
        if self.position_type == "long":
            return (self.current_price - self.cost_basis) * self.quantity
        else:  # short
            return (self.cost_basis - self.current_price) * abs(self.quantity)
    
    @property
    def pnl_percent(self) -> float:
        """Calculate P&L percentage"""
        if self.cost_basis == 0:
            return 0
        return self.pnl / (abs(self.cost_basis * self.quantity))


class VaRCalculator:
    """
    Value at Risk (VaR) Calculator
    
    Calculates the maximum expected loss at a given confidence level
    over a specified time horizon.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize VaR calculator.
        
        Args:
            confidence_level: Confidence level for VaR (e.g., 0.95 for 95%)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def calculate(self,
                 returns: np.ndarray,
                 method: CalculationMethod = CalculationMethod.HISTORICAL,
                 time_horizon: int = 1) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Array of returns
            method: Calculation method
            time_horizon: Time horizon in days
            
        Returns:
            VaR value (positive number representing loss)
        """
        if len(returns) == 0:
            return 0.0
        
        # Scale returns for time horizon
        if time_horizon > 1:
            returns = returns * np.sqrt(time_horizon)
        
        if method == CalculationMethod.HISTORICAL:
            return self._historical_var(returns)
        elif method == CalculationMethod.PARAMETRIC:
            return self._parametric_var(returns)
        elif method == CalculationMethod.MONTE_CARLO:
            return self._monte_carlo_var(returns)
        elif method == CalculationMethod.CORNISH_FISHER:
            return self._cornish_fisher_var(returns)
        else:
            return self._historical_var(returns)
    
    def _historical_var(self, returns: np.ndarray) -> float:
        """Calculate historical VaR"""
        return -np.percentile(returns, self.alpha * 100)
    
    def _parametric_var(self, returns: np.ndarray) -> float:
        """Calculate parametric VaR (assumes normal distribution)"""
        mean = np.mean(returns)
        std = np.std(returns)
        z_score = stats.norm.ppf(self.alpha)
        return -(mean + z_score * std)
    
    def _monte_carlo_var(self, returns: np.ndarray, n_simulations: int = 10000) -> float:
        """Calculate Monte Carlo VaR"""
        mean = np.mean(returns)
        std = np.std(returns)
        
        # Generate simulated returns
        simulated_returns = np.random.normal(mean, std, n_simulations)
        
        return -np.percentile(simulated_returns, self.alpha * 100)
    
    def _cornish_fisher_var(self, returns: np.ndarray) -> float:
        """
        Calculate Cornish-Fisher VaR (adjusts for skewness and kurtosis).
        
        This method provides better estimates for non-normal distributions.
        """
        mean = np.mean(returns)
        std = np.std(returns)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        z = stats.norm.ppf(self.alpha)
        
        # Cornish-Fisher expansion
        z_cf = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurt / 24 - (2*z**3 - 5*z) * skew**2 / 36
        
        return -(mean + z_cf * std)
    
    def calculate_marginal_var(self,
                              portfolio_returns: np.ndarray,
                              position_returns: np.ndarray,
                              position_weight: float) -> float:
        """
        Calculate marginal VaR for a position.
        
        Args:
            portfolio_returns: Portfolio returns
            position_returns: Individual position returns
            position_weight: Position weight in portfolio
            
        Returns:
            Marginal VaR
        """
        portfolio_var = self.calculate(portfolio_returns)
        
        # Calculate portfolio VaR without position
        portfolio_ex_position = portfolio_returns - position_weight * position_returns
        var_ex_position = self.calculate(portfolio_ex_position)
        
        # Marginal VaR is the difference
        return portfolio_var - var_ex_position
    
    def calculate_component_var(self,
                              portfolio_returns: np.ndarray,
                              position_returns: List[np.ndarray],
                              weights: List[float]) -> List[float]:
        """
        Calculate component VaR for each position.
        
        Args:
            portfolio_returns: Portfolio returns
            position_returns: List of position returns
            weights: Position weights
            
        Returns:
            List of component VaRs
        """
        component_vars = []
        
        for i, (pos_returns, weight) in enumerate(zip(position_returns, weights)):
            marginal_var = self.calculate_marginal_var(
                portfolio_returns, pos_returns, weight
            )
            component_var = marginal_var * weight
            component_vars.append(component_var)
        
        return component_vars


class CVaRCalculator:
    """
    Conditional Value at Risk (CVaR) Calculator
    Also known as Expected Shortfall (ES)
    
    Calculates the expected loss beyond VaR threshold.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize CVaR calculator.
        
        Args:
            confidence_level: Confidence level for CVaR
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.var_calculator = VaRCalculator(confidence_level)
    
    def calculate(self,
                 returns: np.ndarray,
                 method: CalculationMethod = CalculationMethod.HISTORICAL,
                 time_horizon: int = 1) -> float:
        """
        Calculate Conditional Value at Risk.
        
        Args:
            returns: Array of returns
            method: Calculation method
            time_horizon: Time horizon in days
            
        Returns:
            CVaR value (positive number representing expected loss beyond VaR)
        """
        if len(returns) == 0:
            return 0.0
        
        # Scale returns for time horizon
        if time_horizon > 1:
            returns = returns * np.sqrt(time_horizon)
        
        if method == CalculationMethod.HISTORICAL:
            return self._historical_cvar(returns)
        elif method == CalculationMethod.PARAMETRIC:
            return self._parametric_cvar(returns)
        elif method == CalculationMethod.MONTE_CARLO:
            return self._monte_carlo_cvar(returns)
        else:
            return self._historical_cvar(returns)
    
    def _historical_cvar(self, returns: np.ndarray) -> float:
        """Calculate historical CVaR"""
        var_threshold = self.var_calculator.calculate(returns)
        tail_losses = returns[returns <= -var_threshold]
        
        if len(tail_losses) == 0:
            return var_threshold
        
        return -np.mean(tail_losses)
    
    def _parametric_cvar(self, returns: np.ndarray) -> float:
        """Calculate parametric CVaR (assumes normal distribution)"""
        mean = np.mean(returns)
        std = np.std(returns)
        
        # For normal distribution, CVaR has closed-form solution
        z_alpha = stats.norm.ppf(self.alpha)
        phi_z = stats.norm.pdf(z_alpha)
        
        return -(mean - std * phi_z / self.alpha)
    
    def _monte_carlo_cvar(self, returns: np.ndarray, n_simulations: int = 10000) -> float:
        """Calculate Monte Carlo CVaR"""
        mean = np.mean(returns)
        std = np.std(returns)
        
        # Generate simulated returns
        simulated_returns = np.random.normal(mean, std, n_simulations)
        
        # Calculate VaR threshold
        var_threshold = np.percentile(simulated_returns, self.alpha * 100)
        
        # Calculate expected shortfall beyond VaR
        tail_losses = simulated_returns[simulated_returns <= var_threshold]
        
        if len(tail_losses) == 0:
            return -var_threshold
        
        return -np.mean(tail_losses)


class ExposureAggregator:
    """
    Exposure Aggregation System
    
    Aggregates and analyzes portfolio exposure across different dimensions.
    """
    
    def __init__(self):
        """Initialize exposure aggregator"""
        self.positions: List[Position] = []
        self.total_capital: float = 0
    
    def add_position(self, position: Position) -> None:
        """Add position to aggregator"""
        self.positions.append(position)
    
    def set_capital(self, capital: float) -> None:
        """Set total capital for exposure calculations"""
        self.total_capital = capital
    
    def calculate_exposures(self) -> Dict[str, float]:
        """
        Calculate various exposure metrics.
        
        Returns:
            Dictionary of exposure metrics
        """
        if not self.positions:
            return self._empty_exposures()
        
        long_exposure = sum(p.market_value for p in self.positions if p.position_type == "long")
        short_exposure = abs(sum(p.market_value for p in self.positions if p.position_type == "short"))
        
        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure
        
        # Calculate as percentage of capital if available
        if self.total_capital > 0:
            gross_pct = gross_exposure / self.total_capital
            net_pct = net_exposure / self.total_capital
            long_pct = long_exposure / self.total_capital
            short_pct = short_exposure / self.total_capital
        else:
            gross_pct = net_pct = long_pct = short_pct = 0
        
        return {
            'gross_exposure': gross_exposure,
            'net_exposure': net_exposure,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'gross_exposure_pct': gross_pct,
            'net_exposure_pct': net_pct,
            'long_exposure_pct': long_pct,
            'short_exposure_pct': short_pct,
            'n_positions': len(self.positions),
            'n_long': sum(1 for p in self.positions if p.position_type == "long"),
            'n_short': sum(1 for p in self.positions if p.position_type == "short")
        }
    
    def calculate_concentration(self) -> Dict[str, float]:
        """
        Calculate concentration metrics.
        
        Returns:
            Dictionary of concentration metrics
        """
        if not self.positions:
            return {'concentration_ratio': 0, 'herfindahl_index': 0, 'top_5_concentration': 0}
        
        # Calculate position weights
        total_value = sum(abs(p.market_value) for p in self.positions)
        if total_value == 0:
            return {'concentration_ratio': 0, 'herfindahl_index': 0, 'top_5_concentration': 0}
        
        weights = [abs(p.market_value) / total_value for p in self.positions]
        weights.sort(reverse=True)
        
        # Concentration ratio (largest position)
        concentration_ratio = weights[0] if weights else 0
        
        # Herfindahl-Hirschman Index
        herfindahl_index = sum(w**2 for w in weights)
        
        # Top 5 concentration
        top_5_concentration = sum(weights[:5])
        
        return {
            'concentration_ratio': concentration_ratio,
            'herfindahl_index': herfindahl_index,
            'top_5_concentration': top_5_concentration,
            'effective_n_positions': 1 / herfindahl_index if herfindahl_index > 0 else 0
        }
    
    def calculate_sector_exposure(self) -> Dict[str, float]:
        """
        Calculate exposure by sector.
        
        Returns:
            Dictionary of sector exposures
        """
        sector_exposure = defaultdict(float)
        
        for position in self.positions:
            if position.sector:
                sector_exposure[position.sector] += position.market_value
        
        # Calculate percentages
        total_value = sum(abs(p.market_value) for p in self.positions)
        if total_value > 0:
            sector_exposure_pct = {
                sector: value / total_value
                for sector, value in sector_exposure.items()
            }
        else:
            sector_exposure_pct = {}
        
        return {
            'sector_exposure': dict(sector_exposure),
            'sector_exposure_pct': sector_exposure_pct,
            'n_sectors': len(sector_exposure)
        }
    
    def _empty_exposures(self) -> Dict[str, float]:
        """Return empty exposure metrics"""
        return {
            'gross_exposure': 0,
            'net_exposure': 0,
            'long_exposure': 0,
            'short_exposure': 0,
            'gross_exposure_pct': 0,
            'net_exposure_pct': 0,
            'long_exposure_pct': 0,
            'short_exposure_pct': 0,
            'n_positions': 0,
            'n_long': 0,
            'n_short': 0
        }


class RiskMetricsEngine:
    """
    Comprehensive Risk Metrics Engine
    
    Integrates VaR, CVaR, and exposure calculations.
    """
    
    def __init__(self):
        """Initialize risk metrics engine"""
        self.var_calculator = VaRCalculator()
        self.cvar_calculator = CVaRCalculator()
        self.exposure_aggregator = ExposureAggregator()
        self.metrics_history: List[RiskMetrics] = []
    
    def calculate_risk_metrics(self,
                              returns: np.ndarray,
                              positions: Optional[List[Position]] = None,
                              capital: Optional[float] = None,
                              method: CalculationMethod = CalculationMethod.HISTORICAL) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            returns: Historical returns
            positions: Current positions
            capital: Total capital
            method: Calculation method
            
        Returns:
            Complete risk metrics
        """
        metrics = RiskMetrics(timestamp=datetime.now())
        
        # Calculate VaR at different confidence levels
        if len(returns) > 0:
            # 95% VaR
            self.var_calculator.confidence_level = 0.95
            metrics.var_95 = self.var_calculator.calculate(returns, method)
            
            # 99% VaR
            self.var_calculator.confidence_level = 0.99
            metrics.var_99 = self.var_calculator.calculate(returns, method)
            
            # 99.5% VaR
            self.var_calculator.confidence_level = 0.995
            metrics.var_995 = self.var_calculator.calculate(returns, method)
            
            # Calculate CVaR at different confidence levels
            # 95% CVaR
            self.cvar_calculator.confidence_level = 0.95
            metrics.cvar_95 = self.cvar_calculator.calculate(returns, method)
            
            # 99% CVaR
            self.cvar_calculator.confidence_level = 0.99
            metrics.cvar_99 = self.cvar_calculator.calculate(returns, method)
            
            # 99.5% CVaR
            self.cvar_calculator.confidence_level = 0.995
            metrics.cvar_995 = self.cvar_calculator.calculate(returns, method)
            
            # Calculate risk-adjusted returns
            metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns)
            metrics.sortino_ratio = self._calculate_sortino_ratio(returns)
            
            # Calculate drawdown
            cumulative_returns = np.cumprod(1 + returns) - 1
            metrics.max_drawdown = self._calculate_max_drawdown(cumulative_returns)
            metrics.current_drawdown = self._calculate_current_drawdown(cumulative_returns)
        
        # Calculate exposure metrics if positions provided
        if positions:
            self.exposure_aggregator.positions = positions
            if capital:
                self.exposure_aggregator.set_capital(capital)
            
            exposures = self.exposure_aggregator.calculate_exposures()
            metrics.gross_exposure = exposures['gross_exposure']
            metrics.net_exposure = exposures['net_exposure']
            metrics.long_exposure = exposures['long_exposure']
            metrics.short_exposure = exposures['short_exposure']
            
            concentration = self.exposure_aggregator.calculate_concentration()
            metrics.concentration_ratio = concentration['concentration_ratio']
            metrics.herfindahl_index = concentration['herfindahl_index']
            metrics.top_5_concentration = concentration['top_5_concentration']
        
        # Store metadata
        metrics.calculation_method = method.value
        metrics.n_observations = len(returns)
        
        # Add to history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (uses downside deviation)"""
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return 0
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0
        
        return np.sqrt(252) * np.mean(excess_returns) / downside_std
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(cumulative_returns) == 0:
            return 0
        
        running_max = np.maximum.accumulate(cumulative_returns + 1)
        drawdown = (cumulative_returns + 1) / running_max - 1
        return abs(np.min(drawdown))
    
    def _calculate_current_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate current drawdown from peak"""
        if len(cumulative_returns) == 0:
            return 0
        
        current_value = cumulative_returns[-1] + 1
        peak_value = np.max(cumulative_returns + 1)
        
        if peak_value == 0:
            return 0
        
        return abs((current_value / peak_value) - 1)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get summary of current risk metrics.
        
        Returns:
            Risk summary dictionary
        """
        if not self.metrics_history:
            return {'status': 'No risk metrics calculated'}
        
        latest = self.metrics_history[-1]
        
        return {
            'timestamp': latest.timestamp.isoformat(),
            'var': {
                '95%': latest.var_95,
                '99%': latest.var_99,
                '99.5%': latest.var_995
            },
            'cvar': {
                '95%': latest.cvar_95,
                '99%': latest.cvar_99,
                '99.5%': latest.cvar_995
            },
            'risk_adjusted_returns': {
                'sharpe_ratio': latest.sharpe_ratio,
                'sortino_ratio': latest.sortino_ratio
            },
            'drawdown': {
                'current': latest.current_drawdown,
                'maximum': latest.max_drawdown
            },
            'exposure': {
                'gross': latest.gross_exposure,
                'net': latest.net_exposure,
                'long': latest.long_exposure,
                'short': latest.short_exposure
            },
            'concentration': {
                'largest_position': latest.concentration_ratio,
                'herfindahl_index': latest.herfindahl_index,
                'top_5': latest.top_5_concentration
            }
        }


def demonstrate_risk_metrics():
    """Demonstrate risk metrics calculations"""
    print("Risk Metrics Engine Demonstration")
    print("=" * 60)
    
    # Create sample returns (simulate daily returns)
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns
    returns[10] = -0.05  # Add some tail events
    returns[50] = -0.04
    returns[100] = -0.06
    
    # Create sample positions
    positions = [
        Position("AAPL", 100, 150, 140),
        Position("GOOGL", 50, 2800, 2750),
        Position("MSFT", 75, 380, 370),
        Position("AMZN", -30, 3400, 3450, position_type="short"),  # Short position
        Position("TSLA", 40, 800, 850)
    ]
    
    capital = 100000
    
    # Initialize engine
    engine = RiskMetricsEngine()
    
    # Calculate metrics
    print("\nCalculating risk metrics...")
    metrics = engine.calculate_risk_metrics(returns, positions, capital)
    
    # Display results
    print("\n" + "=" * 40)
    print("RISK METRICS REPORT")
    print("=" * 40)
    
    print("\nValue at Risk (VaR):")
    print(f"  95% VaR: ${metrics.var_95*capital:.2f} ({metrics.var_95:.2%})")
    print(f"  99% VaR: ${metrics.var_99*capital:.2f} ({metrics.var_99:.2%})")
    print(f"  99.5% VaR: ${metrics.var_995*capital:.2f} ({metrics.var_995:.2%})")
    
    print("\nConditional VaR (CVaR/Expected Shortfall):")
    print(f"  95% CVaR: ${metrics.cvar_95*capital:.2f} ({metrics.cvar_95:.2%})")
    print(f"  99% CVaR: ${metrics.cvar_99*capital:.2f} ({metrics.cvar_99:.2%})")
    print(f"  99.5% CVaR: ${metrics.cvar_995*capital:.2f} ({metrics.cvar_995:.2%})")
    
    print("\nRisk-Adjusted Returns:")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
    
    print("\nDrawdown Metrics:")
    print(f"  Maximum Drawdown: {metrics.max_drawdown:.2%}")
    print(f"  Current Drawdown: {metrics.current_drawdown:.2%}")
    
    print("\nExposure Metrics:")
    print(f"  Gross Exposure: ${metrics.gross_exposure:,.2f} ({metrics.gross_exposure/capital:.1%})")
    print(f"  Net Exposure: ${metrics.net_exposure:,.2f} ({metrics.net_exposure/capital:.1%})")
    print(f"  Long Exposure: ${metrics.long_exposure:,.2f}")
    print(f"  Short Exposure: ${metrics.short_exposure:,.2f}")
    
    print("\nConcentration Metrics:")
    print(f"  Largest Position: {metrics.concentration_ratio:.1%}")
    print(f"  Herfindahl Index: {metrics.herfindahl_index:.3f}")
    print(f"  Top 5 Concentration: {metrics.top_5_concentration:.1%}")
    
    # Test different calculation methods
    print("\n" + "=" * 40)
    print("METHOD COMPARISON")
    print("=" * 40)
    
    methods = [
        CalculationMethod.HISTORICAL,
        CalculationMethod.PARAMETRIC,
        CalculationMethod.MONTE_CARLO,
        CalculationMethod.CORNISH_FISHER
    ]
    
    var_calc = VaRCalculator(0.95)
    
    print("\n95% VaR by Method:")
    for method in methods:
        var = var_calc.calculate(returns, method)
        print(f"  {method.value:20s}: {var:.4f} ({var*100:.2f}%)")
    
    print("\n✅ Risk Metrics Engine operational!")


if __name__ == "__main__":
    demonstrate_risk_metrics()