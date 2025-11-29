"""Core strategy types used across all slices."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Action(Enum):
    """Strategy action to take."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class Decision:
    """Strategy decision output."""

    action: Action
    reason: str
    confidence: float = 0.0
    indicators: dict[str, Any] = field(default_factory=dict)
