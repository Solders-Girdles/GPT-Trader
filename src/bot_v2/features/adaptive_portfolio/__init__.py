"""
Adaptive Portfolio Management Slice

A configuration-first, self-contained slice for managing portfolios that adapt
their behavior based on portfolio size - from $500 micro accounts to $50,000+
large accounts.

Key Features:
- Configuration-driven parameter management
- Hot-swappable settings without code changes
- Tier-based risk and strategy allocation
- PDT rule compliance for small accounts
- Realistic position sizing and cost modeling
"""

from .adaptive_portfolio import get_current_tier, load_portfolio_config, run_adaptive_strategy
from .config_manager import validate_portfolio_config
from .types import AdaptiveResult, PortfolioTier, RiskProfile, TierConfig

__all__ = [
    "run_adaptive_strategy",
    "get_current_tier",
    "validate_portfolio_config",
    "load_portfolio_config",
    "PortfolioTier",
    "AdaptiveResult",
    "TierConfig",
    "RiskProfile",
]
