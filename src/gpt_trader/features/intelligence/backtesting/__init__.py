"""
Backtesting integration for intelligence module.

Provides batch processing and backtest-specific adapters for:
- Historical regime detection with batch mode
- Ensemble orchestration during backtests
- Regime/decision history tracking
"""

from gpt_trader.features.intelligence.backtesting.backtest_adapter import (
    EnsembleBacktestAdapter,
    EnsembleBacktestResult,
)
from gpt_trader.features.intelligence.backtesting.batch_regime import (
    BatchRegimeDetector,
    RegimeHistory,
    RegimeSnapshot,
)

__all__ = [
    # Batch regime detection
    "BatchRegimeDetector",
    "RegimeHistory",
    "RegimeSnapshot",
    # Backtest adapter
    "EnsembleBacktestAdapter",
    "EnsembleBacktestResult",
]
