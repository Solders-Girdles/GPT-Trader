"""
Stress Testing Types and Data Structures

Defines core types, enums, and data classes used across the stress testing system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class StressTestType(Enum):
    """Types of stress tests"""

    MONTE_CARLO = "monte_carlo"
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    SENSITIVITY = "sensitivity"
    REVERSE = "reverse"
    PARAMETRIC = "parametric"


class ScenarioType(Enum):
    """Types of stress scenarios"""

    MARKET_CRASH = "market_crash"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    FLASH_CRASH = "flash_crash"
    BLACK_SWAN = "black_swan"
    REGIME_CHANGE = "regime_change"
    CUSTOM = "custom"


@dataclass
class StressScenario:
    """Definition of a stress scenario"""

    name: str
    scenario_type: ScenarioType
    description: str

    # Market parameters
    market_shock: float = 0.0  # Percentage market move
    volatility_multiplier: float = 1.0
    correlation_adjustment: float = 0.0
    liquidity_factor: float = 1.0

    # Time parameters
    duration_days: int = 1
    shock_speed: str = "instant"  # instant, gradual, accelerating

    # Asset-specific shocks
    asset_shocks: dict[str, float] = field(default_factory=dict)
    sector_shocks: dict[str, float] = field(default_factory=dict)

    # Additional parameters
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StressTestResult:
    """Results from a stress test"""

    scenario: StressScenario
    test_type: StressTestType

    # Portfolio impacts
    portfolio_loss: float
    max_drawdown: float
    var_impact: float
    expected_shortfall: float

    # Risk metrics
    new_var: float
    new_cvar: float
    new_sharpe: float

    # Position impacts
    position_losses: dict[str, float]
    worst_positions: list[tuple[str, float]]

    # Liquidity impacts
    liquidation_cost: float
    days_to_liquidate: float

    # Recovery metrics
    recovery_time: int | None = None
    permanent_loss: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    computation_time: float = 0.0
    confidence_level: float = 0.95
