"""
Position Sizing Types - Complete Isolation

All data structures and enums needed for position sizing.
No external dependencies - everything local to this slice.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union
from enum import Enum
import numpy as np


class SizingMethod(Enum):
    """Position sizing methods available."""
    FIXED = "fixed"
    KELLY = "kelly"
    FRACTIONAL_KELLY = "fractional_kelly"
    CONFIDENCE_ADJUSTED = "confidence_adjusted"
    REGIME_ADJUSTED = "regime_adjusted"
    INTELLIGENT = "intelligent"  # Uses all available intelligence


class RiskLevel(Enum):
    """Risk level for position sizing."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class RiskParameters:
    """Risk parameters for position sizing calculations."""
    max_position_size: float = 0.1  # Maximum position as % of portfolio
    min_position_size: float = 0.01  # Minimum position as % of portfolio
    max_portfolio_risk: float = 0.02  # Maximum portfolio risk per trade
    kelly_fraction: float = 0.25  # Fraction of Kelly to use (conservative)
    confidence_threshold: float = 0.6  # Minimum confidence to trade
    risk_level: RiskLevel = RiskLevel.MODERATE


@dataclass
class PositionSizeRequest:
    """Request for position size calculation."""
    symbol: str
    current_price: float
    portfolio_value: float
    strategy_name: str
    method: SizingMethod = SizingMethod.INTELLIGENT
    
    # Optional intelligence inputs
    win_rate: Optional[float] = None
    avg_win: Optional[float] = None  
    avg_loss: Optional[float] = None
    confidence: Optional[float] = None
    market_regime: Optional[str] = None
    volatility: Optional[float] = None
    
    # Risk parameters
    risk_params: RiskParameters = field(default_factory=RiskParameters)
    
    # Strategy-specific overrides
    strategy_multiplier: float = 1.0


@dataclass
class PositionSizeResponse:
    """Response from position size calculation."""
    symbol: str
    recommended_shares: int
    recommended_value: float
    position_size_pct: float  # As percentage of portfolio
    risk_pct: float  # Estimated risk as percentage of portfolio
    
    # Calculation details
    method_used: SizingMethod
    kelly_fraction: Optional[float] = None
    confidence_adjustment: Optional[float] = None
    regime_adjustment: Optional[float] = None
    
    # Risk metrics
    max_loss_estimate: float = 0.0
    expected_return: float = 0.0
    
    # Reasoning
    calculation_notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class KellyParameters:
    """Parameters for Kelly Criterion calculation."""
    win_rate: float  # Probability of winning trade (0-1)
    avg_win: float   # Average winning trade return (positive)
    avg_loss: float  # Average losing trade return (negative)
    kelly_fraction: float = 0.25  # Fraction of full Kelly to use


@dataclass
class ConfidenceAdjustment:
    """Parameters for confidence-based position adjustments."""
    confidence: float  # Model confidence (0-1)
    min_confidence: float = 0.6  # Below this, reduce position size
    max_adjustment: float = 2.0  # Maximum multiplier for high confidence
    adjustment_curve: str = "linear"  # "linear", "exponential", "sigmoid"


@dataclass
class RegimeMultipliers:
    """Position size multipliers by market regime."""
    bull_quiet: float = 1.2      # Steady uptrend - increase positions
    bull_volatile: float = 0.9   # Volatile uptrend - slight reduction
    bear_quiet: float = 0.6      # Steady downtrend - reduce positions
    bear_volatile: float = 0.4   # Volatile downtrend - major reduction
    sideways_quiet: float = 1.0  # Range-bound - normal sizing
    sideways_volatile: float = 0.7  # Volatile range - reduce slightly
    crisis: float = 0.2          # Crisis mode - emergency reduction
    
    def get_multiplier(self, regime: str) -> float:
        """Get multiplier for a specific regime."""
        regime_map = {
            "bull_quiet": self.bull_quiet,
            "bull_volatile": self.bull_volatile,
            "bear_quiet": self.bear_quiet,
            "bear_volatile": self.bear_volatile,
            "sideways_quiet": self.sideways_quiet,
            "sideways_volatile": self.sideways_volatile,
            "crisis": self.crisis
        }
        return regime_map.get(regime.lower(), 1.0)


@dataclass
class PositionSizingResult:
    """Complete result from intelligent position sizing."""
    primary: PositionSizeResponse
    alternatives: List[PositionSizeResponse] = field(default_factory=list)
    portfolio_impact: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


# Local utility types for internal calculations
@dataclass
class TradeStatistics:
    """Statistics for Kelly Criterion calculation."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_return: float
    avg_win_return: float
    avg_loss_return: float
    win_rate: float
    profit_factor: float
    
    @classmethod
    def from_returns(cls, returns: List[float]) -> 'TradeStatistics':
        """Calculate statistics from list of trade returns."""
        if not returns:
            return cls(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
        total_trades = len(returns)
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        winning_trades = len(wins)
        losing_trades = len(losses)
        total_return = sum(returns)
        
        avg_win_return = np.mean(wins) if wins else 0.0
        avg_loss_return = np.mean(losses) if losses else 0.0
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        total_wins = sum(wins) if wins else 0.0
        total_losses = abs(sum(losses)) if losses else 1.0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        return cls(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_return=total_return,
            avg_win_return=avg_win_return,
            avg_loss_return=avg_loss_return,
            win_rate=win_rate,
            profit_factor=profit_factor
        )


@dataclass  
class VolatilityMetrics:
    """Volatility metrics for risk calculations."""
    daily_volatility: float
    annualized_volatility: float
    value_at_risk_95: float  # 95% VaR
    value_at_risk_99: float  # 99% VaR
    
    @classmethod
    def from_prices(cls, prices: List[float], confidence_95: float = 1.645, 
                   confidence_99: float = 2.326) -> 'VolatilityMetrics':
        """Calculate volatility metrics from price series."""
        if len(prices) < 2:
            return cls(0.0, 0.0, 0.0, 0.0)
            
        returns = np.diff(np.log(prices))
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)  # 252 trading days
        
        var_95 = daily_vol * confidence_95
        var_99 = daily_vol * confidence_99
        
        return cls(
            daily_volatility=daily_vol,
            annualized_volatility=annual_vol,
            value_at_risk_95=var_95,
            value_at_risk_99=var_99
        )