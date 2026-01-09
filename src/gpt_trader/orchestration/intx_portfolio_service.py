"""
DEPRECATED: This module has moved to gpt_trader.features.brokerages.coinbase.intx_portfolio_service

This shim re-exports all symbols for backwards compatibility.
Please update your imports to use the new location:

    # Old (deprecated)
    from gpt_trader.orchestration.intx_portfolio_service import IntxPortfolioService

    # New (preferred)
    from gpt_trader.features.brokerages.coinbase.intx_portfolio_service import IntxPortfolioService
"""

from __future__ import annotations

import warnings

# Re-export all symbols from canonical location
from gpt_trader.features.brokerages.coinbase.intx_portfolio_service import IntxPortfolioService

__all__ = ["IntxPortfolioService"]

# Emit deprecation warning on import
warnings.warn(
    "gpt_trader.orchestration.intx_portfolio_service is deprecated. "
    "Import from gpt_trader.features.brokerages.coinbase.intx_portfolio_service instead.",
    DeprecationWarning,
    stacklevel=2,
)
