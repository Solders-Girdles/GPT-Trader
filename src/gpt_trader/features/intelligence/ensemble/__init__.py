"""
Multi-signal ensemble module.

Provides ensemble orchestration for combining multiple trading strategies:
- Dynamic weight calculation based on market regime
- Confidence-based voting for decision aggregation
- Bayesian adaptive learning for weight adjustment
- Strategy performance tracking with Beta distributions
"""

from gpt_trader.features.intelligence.ensemble.adaptive import (
    BayesianWeightConfig,
    BayesianWeightUpdater,
    StrategyPerformanceRecord,
)
from gpt_trader.features.intelligence.ensemble.models import (
    EnsembleConfig,
    StrategyVote,
)
from gpt_trader.features.intelligence.ensemble.orchestrator import EnsembleOrchestrator
from gpt_trader.features.intelligence.ensemble.voting import (
    ConfidenceLeaderVoting,
    VotingMechanism,
    WeightedMajorityVoting,
)
from gpt_trader.features.intelligence.ensemble.weighting import DynamicWeightCalculator

__all__ = [
    # Adaptive learning
    "BayesianWeightConfig",
    "BayesianWeightUpdater",
    "StrategyPerformanceRecord",
    # Orchestrator
    "EnsembleOrchestrator",
    # Models
    "EnsembleConfig",
    "StrategyVote",
    # Voting
    "ConfidenceLeaderVoting",
    "VotingMechanism",
    "WeightedMajorityVoting",
    # Weighting
    "DynamicWeightCalculator",
]
