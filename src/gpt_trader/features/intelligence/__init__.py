"""
Intelligence feature slice for market regime detection and multi-signal ensemble.

This module provides:
- Market regime detection (trend/volatility classification)
- Multi-signal ensemble orchestration with dynamic weighting
- Confidence-based voting mechanisms
- Regime-aware position sizing
- Backtesting integration for historical analysis

Usage:
    from gpt_trader.features.intelligence import (
        MarketRegimeDetector,
        EnsembleOrchestrator,
        PositionSizer,
        RegimeType,
    )

    # For backtesting
    from gpt_trader.features.intelligence import (
        BatchRegimeDetector,
        EnsembleBacktestAdapter,
    )
"""

from gpt_trader.features.intelligence.backtesting import (
    BatchRegimeDetector,
    EnsembleBacktestAdapter,
    EnsembleBacktestResult,
    RegimeHistory,
    RegimeSnapshot,
)
from gpt_trader.features.intelligence.ensemble import (
    BayesianWeightConfig,
    BayesianWeightUpdater,
    ConfidenceLeaderVoting,
    DynamicWeightCalculator,
    EnsembleConfig,
    EnsembleOrchestrator,
    StrategyPerformanceRecord,
    StrategyVote,
    VotingMechanism,
    WeightedMajorityVoting,
)
from gpt_trader.features.intelligence.regime import (
    MarketRegimeDetector,
    RegimeConfig,
    RegimeState,
    RegimeType,
)
from gpt_trader.features.intelligence.sizing import (
    PositionSizer,
    PositionSizingConfig,
    SizingResult,
)

__all__ = [
    # Regime detection
    "MarketRegimeDetector",
    "RegimeConfig",
    "RegimeState",
    "RegimeType",
    # Ensemble
    "BayesianWeightConfig",
    "BayesianWeightUpdater",
    "ConfidenceLeaderVoting",
    "DynamicWeightCalculator",
    "EnsembleConfig",
    "EnsembleOrchestrator",
    "StrategyPerformanceRecord",
    "StrategyVote",
    "VotingMechanism",
    "WeightedMajorityVoting",
    # Position sizing
    "PositionSizer",
    "PositionSizingConfig",
    "SizingResult",
    # Backtesting
    "BatchRegimeDetector",
    "EnsembleBacktestAdapter",
    "EnsembleBacktestResult",
    "RegimeHistory",
    "RegimeSnapshot",
]
