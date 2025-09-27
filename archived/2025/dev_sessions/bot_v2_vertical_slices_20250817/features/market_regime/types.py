"""
Type definitions for market regime detection - LOCAL to this slice.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum


class MarketRegime(Enum):
    """Primary market regime classifications."""
    BULL_QUIET = "bull_quiet"          # Steady uptrend, low volatility
    BULL_VOLATILE = "bull_volatile"    # Uptrend with high volatility
    BEAR_QUIET = "bear_quiet"          # Steady downtrend, low volatility
    BEAR_VOLATILE = "bear_volatile"    # Downtrend with high volatility
    SIDEWAYS_QUIET = "sideways_quiet"  # Range-bound, low volatility
    SIDEWAYS_VOLATILE = "sideways_volatile"  # Range-bound, high volatility
    CRISIS = "crisis"                  # Extreme market stress


class VolatilityRegime(Enum):
    """Volatility regime classifications."""
    LOW = "low"          # < 15% annualized
    MEDIUM = "medium"    # 15-25% annualized
    HIGH = "high"        # 25-40% annualized
    EXTREME = "extreme"  # > 40% annualized


class TrendRegime(Enum):
    """Trend regime classifications."""
    STRONG_UPTREND = "strong_uptrend"    # > 20% annualized
    UPTREND = "uptrend"                  # 5-20% annualized
    SIDEWAYS = "sideways"                # -5% to 5% annualized
    DOWNTREND = "downtrend"              # -20% to -5% annualized
    STRONG_DOWNTREND = "strong_downtrend"  # < -20% annualized


class RiskSentiment(Enum):
    """Market risk sentiment."""
    RISK_ON = "risk_on"    # Favoring risk assets
    NEUTRAL = "neutral"    # Mixed signals
    RISK_OFF = "risk_off"  # Flight to safety


@dataclass
class RegimeFeatures:
    """Features used for regime detection."""
    # Price features
    returns_1d: float
    returns_5d: float
    returns_20d: float
    returns_60d: float
    
    # Volatility features
    realized_vol_10d: float
    realized_vol_30d: float
    vol_of_vol: float  # Volatility of volatility
    
    # Trend features
    ma_5_20_spread: float  # 5-day MA vs 20-day MA
    ma_20_60_spread: float  # 20-day MA vs 60-day MA
    trend_strength: float  # ADX or similar
    
    # Market structure
    volume_ratio: float  # Current vs average volume
    high_low_ratio: float  # Daily range analysis
    correlation_market: float  # Correlation with broad market
    
    # Risk indicators
    vix_level: Optional[float]  # If available
    put_call_ratio: Optional[float]  # Options sentiment
    safe_haven_flow: Optional[float]  # Bond/gold performance


@dataclass
class RegimeAnalysis:
    """Complete regime analysis output."""
    # Current state
    current_regime: MarketRegime
    confidence: float  # 0-1 confidence in classification
    
    # Component regimes
    volatility_regime: VolatilityRegime
    trend_regime: TrendRegime
    risk_sentiment: RiskSentiment
    
    # Regime metrics
    regime_duration: int  # Days in current regime
    regime_strength: float  # How strongly expressed (0-1)
    stability_score: float  # Likelihood of staying (0-1)
    
    # Transition analysis
    transition_probability: Dict[MarketRegime, float]  # Next regime probabilities
    expected_transition_days: float  # Expected days until change
    
    # Supporting data
    features: RegimeFeatures
    supporting_indicators: Dict[str, float]
    timestamp: datetime


@dataclass
class RegimeChangePrediction:
    """Prediction of regime change."""
    current_regime: MarketRegime
    most_likely_next: MarketRegime
    change_probability: float  # Probability of change in next N days
    confidence: float
    
    # Detailed predictions
    regime_probabilities: Dict[MarketRegime, float]  # All possibilities
    timeframe_days: int  # Prediction horizon
    
    # Change indicators
    leading_indicators: List[str]  # What's signaling change
    confirming_indicators: List[str]  # What would confirm
    
    
@dataclass
class RegimeTransition:
    """Historical regime transition."""
    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_date: datetime
    duration_days: int  # How long the transition took
    trigger_events: List[str]  # What caused the transition
    
    
@dataclass
class RegimeHistory:
    """Historical regime analysis."""
    regimes: List[Tuple[MarketRegime, datetime, datetime]]  # (regime, start, end)
    transitions: List[RegimeTransition]
    
    # Statistics
    average_duration: Dict[MarketRegime, float]  # Days per regime
    transition_matrix: Dict[MarketRegime, Dict[MarketRegime, float]]  # Probabilities
    
    # Performance by regime
    returns_by_regime: Dict[MarketRegime, float]
    volatility_by_regime: Dict[MarketRegime, float]
    sharpe_by_regime: Dict[MarketRegime, float]


@dataclass
class RegimeMonitorState:
    """State for real-time regime monitoring."""
    symbols: List[str]
    current_regimes: Dict[str, MarketRegime]
    last_check: datetime
    check_interval_seconds: int
    alert_on_change: bool
    
    # Tracking
    regime_changes_today: int
    alerts_sent: int
    
    
@dataclass
class RegimeAlert:
    """Alert for regime change."""
    symbol: str
    old_regime: MarketRegime
    new_regime: MarketRegime
    confidence: float
    timestamp: datetime
    message: str
    severity: str  # 'info', 'warning', 'critical'