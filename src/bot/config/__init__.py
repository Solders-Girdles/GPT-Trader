"""Configuration module for GPT-Trader.

This module provides a unified configuration system that:
- Centralizes all configuration in one place
- Eliminates hardcoded values
- Provides type safety and validation
- Supports environment overrides
- Enables configuration profiles
"""

from .financial_config import (
    CapitalAllocation,
    FinancialConfig,
    OptimizationParameters,
    RiskParameters,
    TradingLimits,
    TransactionCosts,
)
from .unified_config import (
    AlpacaConfig,
    DataConfig,
    Environment,
    LoggingConfig,
    OptimizationConfig,
    TradingConfig,
    get_config,
    set_config,
)

__all__ = [
    # Financial configuration
    "FinancialConfig",
    "CapitalAllocation",
    "TradingLimits",
    "RiskParameters",
    "TransactionCosts",
    "OptimizationParameters",
    # Unified configuration
    "TradingConfig",
    "Environment",
    "DataConfig",
    "LoggingConfig",
    "AlpacaConfig",
    "OptimizationConfig",
    "get_config",
    "set_config",
]
