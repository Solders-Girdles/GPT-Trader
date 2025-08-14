"""Risk Management Configuration Module.

This module provides comprehensive risk management configuration classes
and factories for creating risk limits and parameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RiskProfile(str, Enum):
    """Risk profile levels."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class RiskMetricType(str, Enum):
    """Types of risk metrics to monitor."""

    VAR = "var"  # Value at Risk
    CVAR = "cvar"  # Conditional VaR
    VOLATILITY = "volatility"
    DRAWDOWN = "drawdown"
    CORRELATION = "correlation"
    CONCENTRATION = "concentration"
    LIQUIDITY = "liquidity"
    LEVERAGE = "leverage"


@dataclass
class PositionLimits:
    """Position-level risk limits."""

    # Basic position limits
    max_position_size_pct: float = 0.10  # 10% max per position
    min_position_size_usd: float = 100.0  # $100 minimum
    max_position_size_usd: float = 1_000_000.0  # $1M maximum

    # Sector/industry limits
    max_sector_exposure_pct: float = 0.30  # 30% max per sector
    max_industry_exposure_pct: float = 0.20  # 20% max per industry

    # Correlation limits
    max_correlation_threshold: float = 0.70  # Max 70% correlation
    correlation_lookback_days: int = 60  # 60-day correlation window

    # Position holding limits
    max_holding_period_days: int = 365  # 1 year max hold
    position_review_frequency_days: int = 30  # Monthly review

    def validate(self) -> List[str]:
        """Validate position limits."""
        errors = []

        if not 0 < self.max_position_size_pct <= 1.0:
            errors.append("Max position size must be between 0% and 100%")

        if self.min_position_size_usd <= 0:
            errors.append("Minimum position size must be positive")

        if self.max_position_size_usd <= self.min_position_size_usd:
            errors.append("Maximum position size must be greater than minimum")

        if not 0 < self.max_correlation_threshold <= 1.0:
            errors.append("Correlation threshold must be between 0 and 1")

        return errors


@dataclass
class PortfolioLimits:
    """Portfolio-level risk limits."""

    # Exposure limits
    max_gross_exposure_pct: float = 0.95  # 95% max total exposure
    max_net_exposure_pct: float = 0.90  # 90% max net exposure (for long/short)
    min_cash_reserve_pct: float = 0.05  # 5% minimum cash

    # Risk limits
    max_portfolio_var_pct: float = 0.02  # 2% daily VaR
    max_portfolio_cvar_pct: float = 0.03  # 3% daily CVaR
    max_portfolio_volatility_pct: float = 0.25  # 25% annual volatility

    # Drawdown limits
    max_drawdown_pct: float = 0.15  # 15% max drawdown
    max_daily_loss_pct: float = 0.03  # 3% max daily loss
    max_weekly_loss_pct: float = 0.08  # 8% max weekly loss
    max_monthly_loss_pct: float = 0.15  # 15% max monthly loss

    # Position count limits
    max_positions: int = 20
    min_positions: int = 3
    target_positions: int = 10

    # Concentration limits
    max_concentration_ratio: float = 0.25  # Herfindahl index
    max_top_5_concentration_pct: float = 0.60  # Top 5 positions

    # Leverage limits
    max_leverage: float = 1.0  # 1x leverage (no leverage)
    leverage_check_frequency_minutes: int = 15  # Check every 15 minutes

    def validate(self) -> List[str]:
        """Validate portfolio limits."""
        errors = []

        if not 0 < self.max_gross_exposure_pct <= 1.0:
            errors.append("Max gross exposure must be between 0% and 100%")

        if self.max_net_exposure_pct > self.max_gross_exposure_pct:
            errors.append("Max net exposure cannot exceed max gross exposure")

        if self.min_cash_reserve_pct < 0 or self.min_cash_reserve_pct > 0.5:
            errors.append("Cash reserve must be between 0% and 50%")

        if self.max_positions <= self.min_positions:
            errors.append("Max positions must be greater than min positions")

        if not self.min_positions <= self.target_positions <= self.max_positions:
            errors.append("Target positions must be between min and max positions")

        return errors


@dataclass
class StopLossConfig:
    """Stop-loss configuration."""

    # Basic stop-loss settings
    default_stop_loss_pct: float = 0.05  # 5% stop loss
    max_stop_loss_pct: float = 0.15  # 15% maximum stop loss
    min_stop_loss_pct: float = 0.01  # 1% minimum stop loss

    # Trailing stop settings
    enable_trailing_stops: bool = True
    trailing_stop_pct: float = 0.03  # 3% trailing stop
    trailing_activation_pct: float = 0.02  # Activate after 2% profit

    # Time-based stops
    enable_time_stops: bool = False
    max_hold_days: int = 30  # 30-day time stop

    # Breakeven stops
    enable_breakeven_stops: bool = True
    breakeven_activation_pct: float = 0.02  # Move to breakeven after 2% profit

    # Volatility-based stops
    enable_atr_stops: bool = True
    atr_multiplier: float = 2.0  # 2x ATR stop distance
    atr_lookback_periods: int = 14  # 14-period ATR

    def calculate_stop_level(
        self, entry_price: float, current_price: float, atr_value: float = None
    ) -> Dict[str, float]:
        """Calculate stop-loss levels."""
        stops = {}

        # Basic stop loss
        basic_stop = entry_price * (1 - self.default_stop_loss_pct)
        stops["basic_stop"] = basic_stop

        # Trailing stop
        if self.enable_trailing_stops and current_price > entry_price * (
            1 + self.trailing_activation_pct
        ):
            trailing_stop = current_price * (1 - self.trailing_stop_pct)
            stops["trailing_stop"] = trailing_stop

        # ATR-based stop
        if self.enable_atr_stops and atr_value:
            atr_stop = entry_price - (self.atr_multiplier * atr_value)
            stops["atr_stop"] = atr_stop

        # Breakeven stop
        if self.enable_breakeven_stops and current_price > entry_price * (
            1 + self.breakeven_activation_pct
        ):
            stops["breakeven_stop"] = entry_price

        # Use the highest stop level (most conservative)
        if stops:
            stops["effective_stop"] = max(stops.values())

        return stops


@dataclass
class TakeProfitConfig:
    """Take-profit configuration."""

    # Basic take-profit settings
    enable_take_profits: bool = True
    default_take_profit_pct: float = 0.10  # 10% take profit
    max_take_profit_pct: float = 0.50  # 50% maximum take profit

    # Scaling out settings
    enable_scaling_out: bool = True
    first_target_pct: float = 0.05  # 5% first target
    first_target_size_pct: float = 0.30  # Take 30% profit at first target
    second_target_pct: float = 0.10  # 10% second target
    second_target_size_pct: float = 0.50  # Take 50% of remaining at second

    # Momentum-based exits
    enable_momentum_exits: bool = False
    momentum_threshold: float = 0.02  # 2% momentum threshold

    def calculate_targets(self, entry_price: float, position_size: int) -> List[Dict[str, Any]]:
        """Calculate take-profit targets."""
        targets = []

        if not self.enable_take_profits:
            return targets

        remaining_size = position_size

        # First target
        if self.enable_scaling_out and remaining_size > 0:
            target_price = entry_price * (1 + self.first_target_pct)
            target_size = int(position_size * self.first_target_size_pct)
            targets.append({"price": target_price, "size": target_size, "type": "first_target"})
            remaining_size -= target_size

        # Second target
        if self.enable_scaling_out and remaining_size > 0:
            target_price = entry_price * (1 + self.second_target_pct)
            target_size = int(remaining_size * self.second_target_size_pct)
            targets.append({"price": target_price, "size": target_size, "type": "second_target"})
            remaining_size -= target_size

        # Final target
        if remaining_size > 0:
            target_price = entry_price * (1 + self.default_take_profit_pct)
            targets.append({"price": target_price, "size": remaining_size, "type": "final_target"})

        return targets


@dataclass
class MonitoringConfig:
    """Risk monitoring configuration."""

    # Monitoring frequency
    real_time_monitoring: bool = True
    check_frequency_seconds: int = 60  # Check every minute

    # Alert settings
    enable_alerts: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["email", "log"])

    # Risk metric thresholds for alerts
    var_alert_threshold_pct: float = 0.015  # Alert at 1.5% VaR
    drawdown_alert_threshold_pct: float = 0.10  # Alert at 10% drawdown
    concentration_alert_threshold: float = 0.20  # Alert at 20% concentration

    # Performance monitoring
    track_performance: bool = True
    performance_benchmark: Optional[str] = "SPY"  # Benchmark symbol

    # Data requirements
    min_data_points: int = 30  # Minimum data points for calculations
    data_staleness_minutes: int = 5  # Alert if data older than 5 minutes

    # Historical analysis
    enable_backtesting: bool = True
    backtest_periods: List[str] = field(default_factory=lambda: ["1M", "3M", "6M", "1Y"])


@dataclass
class RiskManagementConfig:
    """Comprehensive risk management configuration."""

    # Configuration metadata
    profile: RiskProfile = RiskProfile.MODERATE
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"

    # Sub-configurations
    position_limits: PositionLimits = field(default_factory=PositionLimits)
    portfolio_limits: PortfolioLimits = field(default_factory=PortfolioLimits)
    stop_loss: StopLossConfig = field(default_factory=StopLossConfig)
    take_profit: TakeProfitConfig = field(default_factory=TakeProfitConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Risk calculation settings
    confidence_level: float = 0.95  # 95% confidence for VaR
    lookback_days: int = 252  # 1 year lookback for risk calculations

    # Emergency settings
    enable_circuit_breakers: bool = True
    emergency_stop_loss_pct: float = 0.20  # Emergency stop at 20% portfolio loss

    def validate(self) -> List[str]:
        """Validate entire risk configuration."""
        errors = []

        errors.extend(self.position_limits.validate())
        errors.extend(self.portfolio_limits.validate())

        # Cross-validation checks
        if self.stop_loss.default_stop_loss_pct > self.take_profit.default_take_profit_pct:
            errors.append("Default stop loss should be smaller than take profit")

        if self.portfolio_limits.max_daily_loss_pct > self.emergency_stop_loss_pct:
            errors.append("Daily loss limit should be less than emergency stop")

        return errors

    def update_timestamp(self) -> None:
        """Update the last updated timestamp."""
        self.last_updated = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskManagementConfig":
        """Create from dictionary."""
        # Convert string enums back to enum objects
        if "profile" in data:
            data["profile"] = RiskProfile(data["profile"])

        # Convert datetime strings back to datetime objects
        for field_name in ["created_at", "last_updated"]:
            if field_name in data and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])

        return cls(**data)


class RiskConfigFactory:
    """Factory for creating risk management configurations."""

    @staticmethod
    def create_conservative_config() -> RiskManagementConfig:
        """Create conservative risk configuration."""
        config = RiskManagementConfig(profile=RiskProfile.CONSERVATIVE)

        # Conservative position limits
        config.position_limits.max_position_size_pct = 0.05  # 5% max
        config.position_limits.max_sector_exposure_pct = 0.20  # 20% max
        config.position_limits.max_correlation_threshold = 0.50  # 50% max correlation

        # Conservative portfolio limits
        config.portfolio_limits.max_gross_exposure_pct = 0.80  # 80% max exposure
        config.portfolio_limits.max_drawdown_pct = 0.10  # 10% max drawdown
        config.portfolio_limits.max_daily_loss_pct = 0.02  # 2% daily loss
        config.portfolio_limits.min_cash_reserve_pct = 0.20  # 20% cash

        # Conservative stops
        config.stop_loss.default_stop_loss_pct = 0.03  # 3% stop loss
        config.take_profit.default_take_profit_pct = 0.06  # 6% take profit

        return config

    @staticmethod
    def create_moderate_config() -> RiskManagementConfig:
        """Create moderate risk configuration (default)."""
        return RiskManagementConfig(profile=RiskProfile.MODERATE)

    @staticmethod
    def create_aggressive_config() -> RiskManagementConfig:
        """Create aggressive risk configuration."""
        config = RiskManagementConfig(profile=RiskProfile.AGGRESSIVE)

        # Aggressive position limits
        config.position_limits.max_position_size_pct = 0.20  # 20% max
        config.position_limits.max_sector_exposure_pct = 0.50  # 50% max
        config.position_limits.max_correlation_threshold = 0.80  # 80% max correlation

        # Aggressive portfolio limits
        config.portfolio_limits.max_gross_exposure_pct = 0.98  # 98% max exposure
        config.portfolio_limits.max_drawdown_pct = 0.25  # 25% max drawdown
        config.portfolio_limits.max_daily_loss_pct = 0.05  # 5% daily loss
        config.portfolio_limits.min_cash_reserve_pct = 0.02  # 2% cash
        config.portfolio_limits.max_leverage = 2.0  # 2x leverage allowed

        # Aggressive stops
        config.stop_loss.default_stop_loss_pct = 0.08  # 8% stop loss
        config.take_profit.default_take_profit_pct = 0.20  # 20% take profit

        return config

    @staticmethod
    def create_from_profile(profile: RiskProfile) -> RiskManagementConfig:
        """Create configuration from risk profile."""
        if profile == RiskProfile.CONSERVATIVE:
            return RiskConfigFactory.create_conservative_config()
        elif profile == RiskProfile.MODERATE:
            return RiskConfigFactory.create_moderate_config()
        elif profile == RiskProfile.AGGRESSIVE:
            return RiskConfigFactory.create_aggressive_config()
        else:
            # Custom profile - return moderate as base
            config = RiskConfigFactory.create_moderate_config()
            config.profile = RiskProfile.CUSTOM
            return config
