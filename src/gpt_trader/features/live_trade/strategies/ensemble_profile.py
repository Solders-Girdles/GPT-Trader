"""
Ensemble strategy profile loading and configuration.

Provides YAML-based configuration for EnsembleStrategy with:
- Signal selection and parameters
- Combiner configuration
- Decision thresholds
- Multiple pre-defined profiles
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gpt_trader.features.live_trade.combiners.regime import (
    RegimeAwareCombiner,
    RegimeCombinerConfig,
)
from gpt_trader.features.live_trade.signals.protocol import SignalCombiner, SignalGenerator
from gpt_trader.features.live_trade.signals.registry import create_signal
from gpt_trader.features.live_trade.signals.types import SignalType
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="ensemble_profile")


@dataclass
class SignalProfileConfig:
    """Configuration for a single signal in a profile."""

    name: str
    enabled: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SignalProfileConfig":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            enabled=data.get("enabled", True),
            parameters=data.get("parameters", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "parameters": self.parameters,
        }


@dataclass
class CombinerProfileConfig:
    """Configuration for the signal combiner."""

    adx_period: int = 14
    trending_threshold: int = 25
    ranging_threshold: int = 20

    # Signal type weights by regime
    trending_weights: dict[str, float] = field(default_factory=dict)
    ranging_weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CombinerProfileConfig":
        """Create from dictionary."""
        return cls(
            adx_period=data.get("adx_period", 14),
            trending_threshold=data.get("trending_threshold", 25),
            ranging_threshold=data.get("ranging_threshold", 20),
            trending_weights=data.get("trending_weights", {}),
            ranging_weights=data.get("ranging_weights", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "adx_period": self.adx_period,
            "trending_threshold": self.trending_threshold,
            "ranging_threshold": self.ranging_threshold,
            "trending_weights": self.trending_weights,
            "ranging_weights": self.ranging_weights,
        }

    def to_regime_config(self) -> RegimeCombinerConfig:
        """Convert to RegimeCombinerConfig."""
        config = RegimeCombinerConfig(
            adx_period=self.adx_period,
            trending_threshold=self.trending_threshold,
            ranging_threshold=self.ranging_threshold,
        )

        # Override weights if specified
        if self.trending_weights:
            for signal_type_name, weight in self.trending_weights.items():
                signal_type = _parse_signal_type(signal_type_name)
                if signal_type:
                    config.trending_weights[signal_type] = weight

        if self.ranging_weights:
            for signal_type_name, weight in self.ranging_weights.items():
                signal_type = _parse_signal_type(signal_type_name)
                if signal_type:
                    config.ranging_weights[signal_type] = weight

        return config


@dataclass
class DecisionProfileConfig:
    """Configuration for decision thresholds."""

    buy_threshold: float = 0.2
    sell_threshold: float = -0.2
    close_threshold: float = 0.1
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionProfileConfig":
        """Create from dictionary."""
        return cls(
            buy_threshold=data.get("buy_threshold", 0.2),
            sell_threshold=data.get("sell_threshold", -0.2),
            close_threshold=data.get("close_threshold", 0.1),
            stop_loss_pct=data.get("stop_loss_pct", 0.02),
            take_profit_pct=data.get("take_profit_pct", 0.05),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "close_threshold": self.close_threshold,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
        }


@dataclass
class EnsembleProfile:
    """Complete configuration profile for an EnsembleStrategy.

    Defines:
    - Which signals to use and their parameters
    - How signals are combined (regime weights)
    - Decision thresholds for trading actions
    """

    name: str
    version: str = "1.0"
    description: str = ""

    # Signal configuration
    signals: list[SignalProfileConfig] = field(default_factory=list)

    # Combiner configuration
    combiner: CombinerProfileConfig = field(default_factory=CombinerProfileConfig)

    # Decision thresholds
    decision: DecisionProfileConfig = field(default_factory=DecisionProfileConfig)

    # Tags for organization
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnsembleProfile":
        """Create from dictionary."""
        signals = [
            SignalProfileConfig.from_dict(s) for s in data.get("signals", [])
        ]

        return cls(
            name=data.get("name", "unnamed"),
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            signals=signals,
            combiner=CombinerProfileConfig.from_dict(data.get("combiner", {})),
            decision=DecisionProfileConfig.from_dict(data.get("decision", {})),
            tags=data.get("tags", []),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "EnsembleProfile":
        """Load profile from YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "signals": [s.to_dict() for s in self.signals],
            "combiner": self.combiner.to_dict(),
            "decision": self.decision.to_dict(),
            "tags": self.tags,
        }

    def to_yaml(self, path: Path | str) -> None:
        """Save profile to YAML file."""
        path = Path(path)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def build_signals(self) -> list[SignalGenerator]:
        """Build signal instances from profile configuration.

        Returns:
            List of configured SignalGenerator instances.
        """
        signals = []
        for signal_config in self.signals:
            if not signal_config.enabled:
                continue

            try:
                signal = create_signal(signal_config.name, signal_config.parameters)
                signals.append(signal)
            except ValueError as e:
                logger.warning(
                    f"Failed to create signal '{signal_config.name}': {e}",
                    signal_name=signal_config.name,
                )

        return signals

    def build_combiner(self) -> SignalCombiner:
        """Build combiner from profile configuration.

        Returns:
            Configured SignalCombiner instance.
        """
        config = self.combiner.to_regime_config()
        return RegimeAwareCombiner(config)

    def validate(self) -> list[str]:
        """Validate profile configuration.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        if not self.name:
            errors.append("Profile name is required")

        if not self.signals:
            errors.append("At least one signal is required")

        enabled_signals = [s for s in self.signals if s.enabled]
        if not enabled_signals:
            errors.append("At least one signal must be enabled")

        if self.decision.buy_threshold <= 0:
            errors.append("buy_threshold must be positive")

        if self.decision.sell_threshold >= 0:
            errors.append("sell_threshold must be negative")

        if self.decision.stop_loss_pct <= 0 or self.decision.stop_loss_pct >= 1:
            errors.append("stop_loss_pct must be between 0 and 1")

        return errors


def _parse_signal_type(name: str) -> SignalType | None:
    """Parse signal type from string name."""
    name_upper = name.upper()
    try:
        return SignalType[name_upper]
    except KeyError:
        logger.warning(f"Unknown signal type: {name}")
        return None


def load_ensemble_profile(name: str) -> EnsembleProfile:
    """Load a named ensemble profile from config directory.

    Args:
        name: Profile name (without .yaml extension).

    Returns:
        Loaded EnsembleProfile.

    Raises:
        FileNotFoundError: If profile doesn't exist.
    """
    from gpt_trader.config.path_registry import PROJECT_ROOT

    profiles_dir = PROJECT_ROOT / "config" / "intelligence" / "ensemble_strategies"
    profile_path = profiles_dir / f"{name}.yaml"

    if not profile_path.exists():
        raise FileNotFoundError(f"Ensemble profile not found: {profile_path}")

    return EnsembleProfile.from_yaml(profile_path)


def list_ensemble_profiles() -> list[str]:
    """List available ensemble profiles.

    Returns:
        List of profile names (without .yaml extension).
    """
    from gpt_trader.config.path_registry import PROJECT_ROOT

    profiles_dir = PROJECT_ROOT / "config" / "intelligence" / "ensemble_strategies"
    if not profiles_dir.exists():
        return []

    return [p.stem for p in profiles_dir.glob("*.yaml")]


# Pre-defined profile templates
def get_default_profile() -> EnsembleProfile:
    """Get the default ensemble profile."""
    return EnsembleProfile(
        name="default",
        version="1.0",
        description="Balanced ensemble with trend and mean reversion signals",
        signals=[
            SignalProfileConfig(name="trend"),
            SignalProfileConfig(name="mean_reversion"),
            SignalProfileConfig(name="momentum"),
        ],
        combiner=CombinerProfileConfig(),
        decision=DecisionProfileConfig(),
    )


def get_microstructure_profile() -> EnsembleProfile:
    """Get a profile focused on market microstructure signals."""
    return EnsembleProfile(
        name="microstructure",
        version="1.0",
        description="Microstructure-enhanced ensemble using orderbook and trade flow",
        signals=[
            SignalProfileConfig(name="trend"),
            SignalProfileConfig(
                name="order_flow",
                parameters={"aggressor_threshold_bullish": 0.65, "min_trades": 15},
            ),
            SignalProfileConfig(
                name="orderbook_imbalance",
                parameters={"levels": 10, "imbalance_threshold": 0.25},
            ),
            SignalProfileConfig(name="spread"),
            SignalProfileConfig(
                name="vwap",
                parameters={"deviation_threshold": 0.008, "min_trades": 25},
            ),
        ],
        combiner=CombinerProfileConfig(
            trending_weights={
                "TREND": 0.8,
                "ORDER_FLOW": 1.0,
                "MICROSTRUCTURE": 0.6,
                "MEAN_REVERSION": 0.2,
            },
            ranging_weights={
                "TREND": 0.2,
                "ORDER_FLOW": 0.7,
                "MICROSTRUCTURE": 0.8,
                "MEAN_REVERSION": 1.0,
            },
        ),
        decision=DecisionProfileConfig(
            buy_threshold=0.25,
            sell_threshold=-0.25,
        ),
        tags=["microstructure", "advanced"],
    )


def get_conservative_profile() -> EnsembleProfile:
    """Get a conservative profile with tighter thresholds."""
    return EnsembleProfile(
        name="conservative",
        version="1.0",
        description="Conservative ensemble with higher confidence requirements",
        signals=[
            SignalProfileConfig(name="trend"),
            SignalProfileConfig(name="mean_reversion"),
            SignalProfileConfig(name="spread"),
        ],
        combiner=CombinerProfileConfig(
            trending_threshold=30,  # Require stronger trend
            ranging_threshold=18,
        ),
        decision=DecisionProfileConfig(
            buy_threshold=0.35,
            sell_threshold=-0.35,
            stop_loss_pct=0.015,
            take_profit_pct=0.03,
        ),
        tags=["conservative", "low-risk"],
    )


def get_aggressive_profile() -> EnsembleProfile:
    """Get an aggressive profile with looser thresholds."""
    return EnsembleProfile(
        name="aggressive",
        version="1.0",
        description="Aggressive ensemble for higher frequency trading",
        signals=[
            SignalProfileConfig(name="trend"),
            SignalProfileConfig(name="momentum"),
            SignalProfileConfig(name="order_flow"),
            SignalProfileConfig(name="orderbook_imbalance"),
        ],
        combiner=CombinerProfileConfig(
            trending_threshold=20,
            ranging_threshold=15,
        ),
        decision=DecisionProfileConfig(
            buy_threshold=0.15,
            sell_threshold=-0.15,
            close_threshold=0.08,
            stop_loss_pct=0.025,
            take_profit_pct=0.04,
        ),
        tags=["aggressive", "high-frequency"],
    )
