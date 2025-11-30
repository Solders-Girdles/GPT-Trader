"""Strategy profile definitions.

Provides:
- StrategyProfile: Complete configuration for a trading strategy
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SignalConfig:
    """Configuration for a signal source."""

    name: str
    weight: float = 1.0
    enabled: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "weight": self.weight,
            "enabled": self.enabled,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SignalConfig":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            weight=data.get("weight", 1.0),
            enabled=data.get("enabled", True),
            parameters=data.get("parameters", {}),
        )


@dataclass
class RiskConfig:
    """Risk management configuration."""

    max_position_size: float = 0.10
    max_portfolio_heat: float = 0.06
    stop_loss_percent: float = 0.02
    take_profit_percent: float | None = None
    trailing_stop_percent: float | None = None
    max_daily_loss: float = 0.05
    max_drawdown_pause: float = 0.15

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_position_size": self.max_position_size,
            "max_portfolio_heat": self.max_portfolio_heat,
            "stop_loss_percent": self.stop_loss_percent,
            "take_profit_percent": self.take_profit_percent,
            "trailing_stop_percent": self.trailing_stop_percent,
            "max_daily_loss": self.max_daily_loss,
            "max_drawdown_pause": self.max_drawdown_pause,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RiskConfig":
        """Create from dictionary."""
        return cls(
            max_position_size=data.get("max_position_size", 0.10),
            max_portfolio_heat=data.get("max_portfolio_heat", 0.06),
            stop_loss_percent=data.get("stop_loss_percent", 0.02),
            take_profit_percent=data.get("take_profit_percent"),
            trailing_stop_percent=data.get("trailing_stop_percent"),
            max_daily_loss=data.get("max_daily_loss", 0.05),
            max_drawdown_pause=data.get("max_drawdown_pause", 0.15),
        )


@dataclass
class RegimeConfig:
    """Regime-specific configuration."""

    enabled: bool = True
    scale_factors: dict[str, float] = field(
        default_factory=lambda: {
            "BULL_QUIET": 1.2,
            "BULL_VOLATILE": 0.8,
            "BEAR_QUIET": 0.6,
            "BEAR_VOLATILE": 0.4,
            "CRISIS": 0.2,
            "UNKNOWN": 0.5,
        }
    )
    pause_in_crisis: bool = True
    min_confidence: float = 0.6

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "scale_factors": self.scale_factors,
            "pause_in_crisis": self.pause_in_crisis,
            "min_confidence": self.min_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegimeConfig":
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            scale_factors=data.get("scale_factors", {}),
            pause_in_crisis=data.get("pause_in_crisis", True),
            min_confidence=data.get("min_confidence", 0.6),
        )


@dataclass
class ExecutionConfig:
    """Trade execution configuration."""

    order_type: str = "limit"  # limit, market
    limit_offset_bps: float = 10.0  # basis points from mid
    timeout_seconds: int = 30
    retry_count: int = 3
    slippage_tolerance: float = 0.001

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_type": self.order_type,
            "limit_offset_bps": self.limit_offset_bps,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "slippage_tolerance": self.slippage_tolerance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionConfig":
        """Create from dictionary."""
        return cls(
            order_type=data.get("order_type", "limit"),
            limit_offset_bps=data.get("limit_offset_bps", 10.0),
            timeout_seconds=data.get("timeout_seconds", 30),
            retry_count=data.get("retry_count", 3),
            slippage_tolerance=data.get("slippage_tolerance", 0.001),
        )


@dataclass
class StrategyProfile:
    """Complete configuration profile for a trading strategy.

    Combines all configuration aspects:
    - Strategy identification
    - Signal sources and weights
    - Risk management
    - Regime adaptation
    - Execution settings
    """

    # Identity
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    # Trading settings
    symbols: list[str] = field(default_factory=lambda: ["BTC-USD"])
    timeframe: str = "1h"
    min_confidence_threshold: float = 0.6

    # Signal configuration
    signals: list[SignalConfig] = field(default_factory=list)

    # Risk configuration
    risk: RiskConfig = field(default_factory=RiskConfig)

    # Regime configuration
    regime: RegimeConfig = field(default_factory=RegimeConfig)

    # Execution configuration
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Additional parameters
    parameters: dict[str, Any] = field(default_factory=dict)

    # Tags for organization
    tags: list[str] = field(default_factory=list)

    # Environment (paper, live)
    environment: str = "paper"

    def __post_init__(self) -> None:
        """Compute config hash."""
        self._config_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of configuration for versioning."""
        config_str = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]

    @property
    def config_hash(self) -> str:
        """Get configuration hash."""
        return self._config_hash

    def get_signal_weights(self) -> dict[str, float]:
        """Get enabled signal weights as dictionary."""
        return {signal.name: signal.weight for signal in self.signals if signal.enabled}

    def get_signal_parameters(self, signal_name: str) -> dict[str, Any]:
        """Get parameters for a specific signal."""
        for signal in self.signals:
            if signal.name == signal_name:
                return signal.parameters
        return {}

    def get_regime_scale(self, regime: str) -> float:
        """Get scale factor for a regime."""
        if not self.regime.enabled:
            return 1.0
        return self.regime.scale_factors.get(regime, 0.5)

    def should_trade_in_regime(self, regime: str, confidence: float) -> bool:
        """Check if trading is allowed in current regime."""
        if not self.regime.enabled:
            return True

        # Check confidence threshold
        if confidence < self.regime.min_confidence:
            return False

        # Check crisis pause
        if self.regime.pause_in_crisis and regime == "CRISIS":
            return False

        return True

    def clone(self, new_name: str | None = None) -> "StrategyProfile":
        """Create a copy of this profile.

        Args:
            new_name: Name for the cloned profile

        Returns:
            New StrategyProfile instance
        """
        data = self.to_dict()
        data["name"] = new_name or f"{self.name}_copy"
        data["created_at"] = datetime.now().isoformat()
        return StrategyProfile.from_dict(data)

    def validate(self) -> list[str]:
        """Validate profile configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check name
        if not self.name:
            errors.append("Profile name is required")

        # Check symbols
        if not self.symbols:
            errors.append("At least one symbol is required")

        # Check signal weights
        total_weight = sum(s.weight for s in self.signals if s.enabled)
        if total_weight <= 0 and self.signals:
            errors.append("Total signal weight must be positive")

        # Check risk limits
        if self.risk.max_position_size > 1.0:
            errors.append("max_position_size cannot exceed 1.0")

        if self.risk.stop_loss_percent >= 1.0:
            errors.append("stop_loss_percent must be less than 1.0")

        # Check confidence threshold
        if not 0 <= self.min_confidence_threshold <= 1:
            errors.append("min_confidence_threshold must be between 0 and 1")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "min_confidence_threshold": self.min_confidence_threshold,
            "signals": [s.to_dict() for s in self.signals],
            "risk": self.risk.to_dict(),
            "regime": self.regime.to_dict(),
            "execution": self.execution.to_dict(),
            "parameters": self.parameters,
            "tags": self.tags,
            "environment": self.environment,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyProfile":
        """Create from dictionary."""
        signals = [SignalConfig.from_dict(s) for s in data.get("signals", [])]

        risk = RiskConfig.from_dict(data.get("risk", {}))
        regime = RegimeConfig.from_dict(data.get("regime", {}))
        execution = ExecutionConfig.from_dict(data.get("execution", {}))

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            created_at=created_at,
            symbols=data.get("symbols", ["BTC-USD"]),
            timeframe=data.get("timeframe", "1h"),
            min_confidence_threshold=data.get("min_confidence_threshold", 0.6),
            signals=signals,
            risk=risk,
            regime=regime,
            execution=execution,
            parameters=data.get("parameters", {}),
            tags=data.get("tags", []),
            environment=data.get("environment", "paper"),
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StrategyProfile(name={self.name}, version={self.version}, "
            f"signals={len(self.signals)}, environment={self.environment})"
        )
