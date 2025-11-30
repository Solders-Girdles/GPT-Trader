"""
Type definitions for the Signal Ensemble architecture.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SignalType(Enum):
    """Category of the trading signal."""

    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    SENTIMENT = "sentiment"
    OTHER = "other"


@dataclass
class SignalOutput:
    """Standardized output from a SignalGenerator."""

    name: str
    type: SignalType
    strength: float  # -1.0 (Strong Sell) to +1.0 (Strong Buy)
    confidence: float  # 0.0 (No confidence) to 1.0 (High confidence)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate ranges."""
        self.strength = max(-1.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))
