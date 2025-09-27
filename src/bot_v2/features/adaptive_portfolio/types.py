"""
Type definitions for adaptive portfolio management.

All types are local to this slice - no external dependencies.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum


class PortfolioTier(Enum):
    """Portfolio size tiers with different behaviors."""
    MICRO = "micro"
    SMALL = "small" 
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class RiskProfile:
    """Risk management parameters for a tier."""
    daily_limit_pct: float
    quarterly_limit_pct: float
    position_stop_loss_pct: float
    max_sector_exposure_pct: float


@dataclass
class PositionConstraints:
    """Position sizing constraints for a tier."""
    min_positions: int
    max_positions: int
    target_positions: int
    min_position_size: float


@dataclass
class TradingRules:
    """Trading frequency and account rules for a tier."""
    max_trades_per_week: int
    account_type: str  # "cash" or "margin"
    settlement_days: int
    pdt_compliant: bool


@dataclass
class TierConfig:
    """Complete configuration for a portfolio tier."""
    name: str
    range: Tuple[float, float]  # (min_capital, max_capital)
    positions: PositionConstraints
    min_position_size: float
    strategies: List[str]
    risk: RiskProfile
    trading: TradingRules


@dataclass
class MarketConstraints:
    """Market-wide constraints for all tiers."""
    min_share_price: float
    max_share_price: float
    min_daily_volume: int
    excluded_sectors: List[str]
    excluded_symbols: List[str]
    market_hours_only: bool


@dataclass
class CostStructure:
    """Trading cost assumptions."""
    commission_per_trade: float
    spread_estimate_pct: float
    slippage_pct: float
    financing_rate_annual_pct: float


@dataclass
class PortfolioConfig:
    """Complete adaptive portfolio configuration."""
    version: str
    last_updated: str
    description: str
    tiers: Dict[str, TierConfig]
    costs: CostStructure
    market_constraints: MarketConstraints
    validation: Dict[str, Any]
    rebalancing: Dict[str, Any]


@dataclass
class PositionInfo:
    """Information about a single position."""
    symbol: str
    shares: int
    entry_price: float
    current_price: float
    position_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    days_held: int
    stop_loss_price: Optional[float] = None


@dataclass
class PortfolioSnapshot:
    """Current state of the portfolio."""
    total_value: float
    cash: float
    positions: List[PositionInfo]
    daily_pnl: float
    daily_pnl_pct: float
    quarterly_pnl_pct: float
    current_tier: PortfolioTier
    positions_count: int
    largest_position_pct: float
    sector_exposures: Dict[str, float]


@dataclass
class TradingSignal:
    """A trading signal with tier-appropriate sizing."""
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0
    target_position_size: float  # Dollar amount
    stop_loss_pct: float
    strategy_source: str
    reasoning: str


@dataclass
class AdaptiveResult:
    """Result of adaptive portfolio analysis."""
    current_tier: PortfolioTier
    tier_config: TierConfig
    portfolio_snapshot: PortfolioSnapshot
    signals: List[TradingSignal]
    risk_metrics: Dict[str, float]
    tier_transition_needed: bool
    tier_transition_target: Optional[PortfolioTier]
    recommended_actions: List[str]
    warnings: List[str]
    timestamp: datetime


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


@dataclass
class BacktestMetrics:
    """Metrics from adaptive backtesting."""
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    avg_trade_return_pct: float
    total_trades: int
    tier_transitions: int
    final_tier: PortfolioTier
    tier_performance: Dict[str, Dict[str, float]]