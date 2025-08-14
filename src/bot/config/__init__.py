"""Configuration module for GPT-Trader.

This module provides a unified configuration system that:
- Centralizes all configuration in one place
- Eliminates hardcoded values
- Provides type safety and validation
- Supports environment overrides
- Enables configuration profiles

NOTE: The consolidated configuration is now in the parent config.py module.
This __init__.py maintains backward compatibility by re-exporting the
original separate config modules.
"""

# Re-export from the existing separate modules for backward compatibility
from .financial_config import (
    CapitalAllocation,
    FinancialConfig,
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
    # Main configuration
    "TradingConfig",
    "get_config",
    "set_config",
    # Enums
    "Environment",
    # Financial configuration
    "FinancialConfig",
    "CapitalAllocation",
    "TradingLimits",
    "RiskParameters",
    "TransactionCosts",
    # Component configurations
    "DataConfig",
    "LoggingConfig",
    "AlpacaConfig",
    "OptimizationConfig",
]

__all__ = [
    # Main configuration
    "TradingConfig",
    "get_config",
    "set_config",
    "reload_config",
    # Enums
    "Environment",
    "LogLevel",
    "SecurityLevel",
    # Financial configuration
    "FinancialConfig",
    "CapitalAllocation",
    "TradingLimits",
    "RiskParameters",
    "TransactionCosts",
    # Component configurations
    "DataConfig",
    "LoggingConfig",
    "AlpacaConfig",
    "DatabaseConfig",
    "SecurityConfig",
    "OptimizationConfig",
    # Utilities
    "ConfigManager",
]
