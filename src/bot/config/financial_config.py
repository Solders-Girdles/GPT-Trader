"""Financial configuration for GPT-Trader.

This module centralizes all financial constants and trading parameters
to eliminate hardcoded values throughout the codebase.
"""

import decimal
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TradingLimits(BaseModel):
    """Trading position and risk limits."""

    max_position_size: Decimal = Field(
        default=Decimal("1000000.0"), description="Maximum size for a single position in USD", gt=0
    )

    max_order_value: Decimal = Field(
        default=Decimal("100000.0"), description="Maximum value for a single order in USD", gt=0
    )

    max_portfolio_positions: int = Field(
        default=20, description="Maximum number of concurrent positions", gt=0
    )

    max_leverage: float = Field(
        default=1.0, description="Maximum leverage multiplier", ge=1.0, le=4.0
    )

    min_position_size: Decimal = Field(
        default=Decimal("100.0"), description="Minimum position size in USD", gt=0
    )

    position_size_increment: Decimal = Field(
        default=Decimal("0.01"), description="Minimum increment for position sizing", gt=0
    )

    @field_validator("max_position_size", "max_order_value", "min_position_size")
    @classmethod
    def validate_positive_decimal(cls, v: Decimal) -> Decimal:
        """Ensure decimal values are positive."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v


class RiskParameters(BaseModel):
    """Risk management parameters."""

    max_portfolio_risk: float = Field(
        default=0.02, description="Maximum portfolio risk per trade (2%)", gt=0, le=0.1
    )

    max_daily_loss: Decimal = Field(
        default=Decimal("5000.0"), description="Maximum daily loss limit in USD", gt=0
    )

    max_drawdown_percent: float = Field(
        default=0.20, description="Maximum acceptable drawdown (20%)", gt=0, le=0.5
    )

    stop_loss_percent: float = Field(
        default=0.02, description="Default stop loss percentage (2%)", gt=0, le=0.1
    )

    take_profit_percent: float = Field(
        default=0.05, description="Default take profit percentage (5%)", gt=0, le=0.5
    )

    risk_free_rate: float = Field(
        default=0.05,
        description="Annual risk-free rate for Sharpe ratio calculations",
        ge=0,
        le=0.2,
    )

    confidence_level: float = Field(
        default=0.95, description="Confidence level for VaR calculations", gt=0.9, lt=1.0
    )


class TransactionCosts(BaseModel):
    """Transaction cost configuration."""

    commission_per_share: Decimal = Field(
        default=Decimal("0.005"), description="Commission per share in USD", ge=0
    )

    commission_minimum: Decimal = Field(
        default=Decimal("1.0"), description="Minimum commission per trade in USD", ge=0
    )

    commission_rate_bps: float = Field(
        default=10.0, description="Commission rate in basis points", ge=0, le=100
    )

    slippage_bps: float = Field(
        default=5.0, description="Expected slippage in basis points", ge=0, le=100
    )

    market_impact_bps: float = Field(
        default=2.0, description="Expected market impact in basis points", ge=0, le=100
    )

    @property
    def commission_rate_decimal(self) -> float:
        """Convert commission rate from basis points to decimal."""
        return self.commission_rate_bps / 10000.0

    @property
    def slippage_decimal(self) -> float:
        """Convert slippage from basis points to decimal."""
        return self.slippage_bps / 10000.0

    @property
    def total_cost_bps(self) -> float:
        """Total transaction cost in basis points."""
        return self.commission_rate_bps + self.slippage_bps + self.market_impact_bps


class CapitalAllocation(BaseModel):
    """Capital allocation configuration."""

    initial_capital: Decimal = Field(
        default=Decimal("100000.0"), description="Initial trading capital in USD", gt=0
    )

    paper_trading_capital: Decimal = Field(
        default=Decimal("100000.0"), description="Paper trading account capital in USD", gt=0
    )

    backtesting_capital: Decimal = Field(
        default=Decimal("100000.0"), description="Default capital for backtesting in USD", gt=0
    )

    deployment_budget: Decimal = Field(
        default=Decimal("10000.0"), description="Budget for strategy deployment in USD", gt=0
    )

    reserve_capital_percent: float = Field(
        default=0.20, description="Percentage of capital to keep in reserve (20%)", ge=0, le=0.5
    )

    max_capital_per_strategy: Decimal = Field(
        default=Decimal("50000.0"),
        description="Maximum capital allocation per strategy in USD",
        gt=0,
    )

    min_capital_per_strategy: Decimal = Field(
        default=Decimal("1000.0"),
        description="Minimum capital allocation per strategy in USD",
        gt=0,
    )

    @field_validator("initial_capital", "paper_trading_capital", "backtesting_capital")
    @classmethod
    def validate_capital_amounts(cls, v: Decimal) -> Decimal:
        """Validate capital amounts are reasonable."""
        if v < 1000:
            raise ValueError(f"Capital must be at least $1,000, got ${v}")
        if v > 100_000_000:
            raise ValueError(f"Capital exceeds maximum limit of $100M, got ${v}")
        return v

    @property
    def available_trading_capital(self) -> Decimal:
        """Calculate available trading capital after reserves."""
        return self.initial_capital * Decimal(str(1 - self.reserve_capital_percent))


class OptimizationParameters(BaseModel):
    """Optimization and backtesting parameters."""

    min_trades_for_validation: int = Field(
        default=30, description="Minimum trades required for strategy validation", gt=10
    )

    min_sharpe_ratio: float = Field(
        default=1.0, description="Minimum acceptable Sharpe ratio", gt=0
    )

    target_sharpe_ratio: float = Field(
        default=2.0, description="Target Sharpe ratio for optimization", gt=1.0
    )

    min_win_rate: float = Field(
        default=0.45, description="Minimum acceptable win rate", gt=0.3, le=1.0
    )

    min_profit_factor: float = Field(
        default=1.2, description="Minimum acceptable profit factor", gt=1.0
    )

    walk_forward_periods: int = Field(
        default=12, description="Number of walk-forward testing periods", gt=1
    )

    optimization_metric: str = Field(
        default="sharpe_ratio", description="Primary metric for optimization"
    )

    @field_validator("optimization_metric")
    @classmethod
    def validate_optimization_metric(cls, v: str) -> str:
        """Validate optimization metric is supported."""
        valid_metrics = {
            "sharpe_ratio",
            "total_return",
            "win_rate",
            "profit_factor",
            "calmar_ratio",
            "sortino_ratio",
        }
        if v not in valid_metrics:
            raise ValueError(f"Invalid metric: {v}. Must be one of {valid_metrics}")
        return v


class FinancialConfig(BaseModel):
    """Comprehensive financial configuration.

    This centralizes all financial constants and parameters,
    eliminating hardcoded values throughout the codebase.
    """

    # Sub-configurations
    capital: CapitalAllocation = Field(
        default_factory=CapitalAllocation, description="Capital allocation settings"
    )

    limits: TradingLimits = Field(
        default_factory=TradingLimits, description="Trading limits and constraints"
    )

    risk: RiskParameters = Field(
        default_factory=RiskParameters, description="Risk management parameters"
    )

    costs: TransactionCosts = Field(
        default_factory=TransactionCosts, description="Transaction cost configuration"
    )

    optimization: OptimizationParameters = Field(
        default_factory=OptimizationParameters,
        description="Optimization and backtesting parameters",
    )

    # Currency and locale settings
    base_currency: str = Field(default="USD", description="Base currency for all calculations")

    decimal_precision: int = Field(
        default=2, description="Decimal precision for financial calculations", ge=0, le=8
    )

    # Market hours (in exchange timezone)
    market_open_hour: int = Field(
        default=9, description="Market open hour (24-hour format)", ge=0, lt=24
    )

    market_open_minute: int = Field(default=30, description="Market open minute", ge=0, lt=60)

    market_close_hour: int = Field(
        default=16, description="Market close hour (24-hour format)", ge=0, lt=24
    )

    market_close_minute: int = Field(default=0, description="Market close minute", ge=0, lt=60)

    # Data retention
    max_historical_days: int = Field(
        default=730, description="Maximum days of historical data to retain (2 years)", gt=30
    )

    @property
    def initial_capital_float(self) -> float:
        """Get initial capital as float for compatibility."""
        return float(self.capital.initial_capital)

    @property
    def commission_rate(self) -> float:
        """Get commission rate as decimal for compatibility."""
        return self.costs.commission_rate_decimal

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return self.model_dump()

    @classmethod
    def from_env(cls) -> "FinancialConfig":
        """Create configuration from environment variables.

        Supports environment variable overrides for key parameters:
        - TRADING_INITIAL_CAPITAL
        - TRADING_MAX_POSITION_SIZE
        - TRADING_MAX_DAILY_LOSS
        - TRADING_COMMISSION_BPS
        """
        import os

        config = cls()

        # Override from environment if set
        if capital := os.getenv("TRADING_INITIAL_CAPITAL"):
            config.capital.initial_capital = Decimal(capital)

        if max_position := os.getenv("TRADING_MAX_POSITION_SIZE"):
            config.limits.max_position_size = Decimal(max_position)

        if max_loss := os.getenv("TRADING_MAX_DAILY_LOSS"):
            config.risk.max_daily_loss = Decimal(max_loss)

        if commission := os.getenv("TRADING_COMMISSION_BPS"):
            config.costs.commission_rate_bps = float(commission)

        return config

    def save_to_file(self, path: Path) -> None:
        """Save configuration to JSON file.

        Args:
            path: Path to save configuration file.
        """
        import json

        # Convert Decimal to string for JSON serialization
        def decimal_default(obj: Any) -> str:
            if isinstance(obj, Decimal):
                return str(obj)
            raise TypeError

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2, default=decimal_default)

    @classmethod
    def load_from_file(cls, path: Path) -> "FinancialConfig":
        """Load configuration from JSON file.

        Args:
            path: Path to configuration file.

        Returns:
            FinancialConfig instance.
        """
        import json

        with open(path) as f:
            data = json.load(f)

        # Convert string decimals back to Decimal
        def convert_decimals(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: convert_decimals(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_decimals(item) for item in obj]
            elif isinstance(obj, str) and "." in obj:
                try:
                    return Decimal(obj)
                except (ValueError, TypeError, decimal.InvalidOperation):
                    # String cannot be converted to Decimal - return as is
                    return obj
            return obj

        data = convert_decimals(data)
        return cls(**data)
