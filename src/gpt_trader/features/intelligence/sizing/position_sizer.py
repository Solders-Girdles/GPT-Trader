"""
Regime-aware position sizing module.

Provides intelligent position sizing that adapts to market conditions:
- Regime-based scaling (reduce in crisis, adjust for volatility)
- ATR-based volatility targeting
- Optional Kelly criterion for optimal sizing
- Risk budgeting across strategies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from ..regime.models import RegimeState, RegimeType

if TYPE_CHECKING:
    from ..regime.detector import MarketRegimeDetector


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing behavior.

    All parameters can be tuned via YAML profiles.
    """

    # Base position size as fraction of equity (default 2%)
    base_position_fraction: float = 0.02

    # Maximum position size as fraction of equity (default 10%)
    max_position_fraction: float = 0.10

    # Minimum position size as fraction of equity (default 0.5%)
    min_position_fraction: float = 0.005

    # Regime-specific scaling multipliers
    regime_scale_factors: dict[str, float] = field(
        default_factory=lambda: {
            RegimeType.BULL_QUIET.name: 1.2,  # Favorable conditions
            RegimeType.BULL_VOLATILE.name: 0.8,  # Reduce for volatility
            RegimeType.BEAR_QUIET.name: 0.7,  # Careful in bear
            RegimeType.BEAR_VOLATILE.name: 0.5,  # Very careful
            RegimeType.SIDEWAYS_QUIET.name: 1.0,  # Normal
            RegimeType.SIDEWAYS_VOLATILE.name: 0.6,  # Reduce for chop
            RegimeType.CRISIS.name: 0.2,  # Minimal in crisis
            RegimeType.UNKNOWN.name: 0.5,  # Conservative when uncertain
        }
    )

    # ATR-based volatility targeting
    enable_volatility_scaling: bool = True
    target_volatility: float = 0.02  # 2% daily target volatility
    volatility_lookback: int = 20  # ATR lookback period

    # Kelly criterion
    enable_kelly_sizing: bool = False
    kelly_fraction: float = 0.25  # Use 25% of full Kelly (quarter Kelly)
    min_win_rate_for_kelly: float = 0.5  # Need >50% win rate

    # Confidence scaling
    enable_confidence_scaling: bool = True
    confidence_exponent: float = 1.5  # How strongly confidence affects size

    # Risk limits
    max_portfolio_heat: float = 0.06  # Max total risk across all positions
    max_single_loss: float = 0.02  # Max loss per trade (2% of equity)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "base_position_fraction": self.base_position_fraction,
            "max_position_fraction": self.max_position_fraction,
            "min_position_fraction": self.min_position_fraction,
            "regime_scale_factors": self.regime_scale_factors,
            "enable_volatility_scaling": self.enable_volatility_scaling,
            "target_volatility": self.target_volatility,
            "volatility_lookback": self.volatility_lookback,
            "enable_kelly_sizing": self.enable_kelly_sizing,
            "kelly_fraction": self.kelly_fraction,
            "min_win_rate_for_kelly": self.min_win_rate_for_kelly,
            "enable_confidence_scaling": self.enable_confidence_scaling,
            "confidence_exponent": self.confidence_exponent,
            "max_portfolio_heat": self.max_portfolio_heat,
            "max_single_loss": self.max_single_loss,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PositionSizingConfig:
        """Create config from dictionary."""
        return cls(
            base_position_fraction=data.get("base_position_fraction", 0.02),
            max_position_fraction=data.get("max_position_fraction", 0.10),
            min_position_fraction=data.get("min_position_fraction", 0.005),
            regime_scale_factors=data.get("regime_scale_factors", {}),
            enable_volatility_scaling=data.get("enable_volatility_scaling", True),
            target_volatility=data.get("target_volatility", 0.02),
            volatility_lookback=data.get("volatility_lookback", 20),
            enable_kelly_sizing=data.get("enable_kelly_sizing", False),
            kelly_fraction=data.get("kelly_fraction", 0.25),
            min_win_rate_for_kelly=data.get("min_win_rate_for_kelly", 0.5),
            enable_confidence_scaling=data.get("enable_confidence_scaling", True),
            confidence_exponent=data.get("confidence_exponent", 1.5),
            max_portfolio_heat=data.get("max_portfolio_heat", 0.06),
            max_single_loss=data.get("max_single_loss", 0.02),
        )


@dataclass
class SizingResult:
    """Result of position sizing calculation.

    Contains the recommended position size along with all factors
    that influenced the calculation for transparency.
    """

    # Final recommended position size (as fraction of equity)
    position_fraction: float

    # Position size in quote currency (e.g., USD)
    position_value: Decimal

    # Position size in base currency (e.g., BTC)
    position_quantity: Decimal

    # Component factors (for transparency/debugging)
    base_size: float
    regime_factor: float
    volatility_factor: float
    confidence_factor: float
    kelly_factor: float

    # Risk metrics
    estimated_risk: float  # Expected max loss as fraction of equity
    risk_reward_ratio: float  # If available

    # Metadata
    regime: str
    regime_confidence: float
    atr_value: float | None
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "position_fraction": round(self.position_fraction, 6),
            "position_value": str(self.position_value),
            "position_quantity": str(self.position_quantity),
            "factors": {
                "base_size": round(self.base_size, 4),
                "regime_factor": round(self.regime_factor, 4),
                "volatility_factor": round(self.volatility_factor, 4),
                "confidence_factor": round(self.confidence_factor, 4),
                "kelly_factor": round(self.kelly_factor, 4),
            },
            "risk": {
                "estimated_risk": round(self.estimated_risk, 6),
                "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            },
            "context": {
                "regime": self.regime,
                "regime_confidence": round(self.regime_confidence, 4),
                "atr_value": round(self.atr_value, 6) if self.atr_value else None,
            },
            "reasoning": self.reasoning,
        }


class PositionSizer:
    """Regime-aware position sizing calculator.

    Integrates with MarketRegimeDetector to provide intelligent position
    sizing that adapts to market conditions.

    Features:
    - Regime-based scaling (crisis mode = 0.2x)
    - ATR-based volatility targeting
    - Confidence-weighted sizing
    - Optional Kelly criterion
    - Risk limit enforcement

    Example:
        from gpt_trader.features.intelligence.regime import MarketRegimeDetector
        from gpt_trader.features.intelligence.sizing import PositionSizer

        detector = MarketRegimeDetector()
        sizer = PositionSizer(regime_detector=detector)

        # Update regime with price data
        detector.update("BTC-USD", Decimal("50000"))

        # Calculate position size
        result = sizer.calculate_size(
            symbol="BTC-USD",
            current_price=Decimal("50000"),
            equity=Decimal("10000"),
            decision_confidence=0.75,
        )

        print(f"Position size: {result.position_quantity} BTC")
        print(f"Position value: ${result.position_value}")
    """

    def __init__(
        self,
        regime_detector: MarketRegimeDetector | None = None,
        config: PositionSizingConfig | None = None,
    ):
        """Initialize position sizer.

        Args:
            regime_detector: Regime detector for market state
            config: Sizing configuration
        """
        self.regime_detector = regime_detector
        self.config = config or PositionSizingConfig()

        # Track historical win rates for Kelly (per symbol)
        self._win_rates: dict[str, tuple[int, int]] = {}  # symbol -> (wins, losses)

        # Track current portfolio heat
        self._portfolio_positions: dict[str, float] = {}  # symbol -> risk amount

    def calculate_size(
        self,
        symbol: str,
        current_price: Decimal,
        equity: Decimal,
        decision_confidence: float = 1.0,
        stop_loss_distance: Decimal | None = None,
        take_profit_distance: Decimal | None = None,
        existing_positions: dict[str, float] | None = None,
    ) -> SizingResult:
        """Calculate regime-aware position size.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            equity: Account equity
            decision_confidence: Confidence from trading decision (0-1)
            stop_loss_distance: Distance to stop loss (for risk calculation)
            take_profit_distance: Distance to take profit (for R:R)
            existing_positions: Current position risk amounts

        Returns:
            SizingResult with recommended position size
        """
        reasoning_parts: list[str] = []

        # 1. Get regime state
        regime_state = self._get_regime_state(symbol)
        regime_factor = self._get_regime_factor(regime_state)
        reasoning_parts.append(f"Regime={regime_state.regime.name} (factor={regime_factor:.2f})")

        # 2. Calculate volatility factor
        volatility_factor = self._get_volatility_factor(symbol, current_price)
        if self.config.enable_volatility_scaling:
            reasoning_parts.append(f"Vol factor={volatility_factor:.2f}")

        # 3. Calculate confidence factor
        confidence_factor = self._get_confidence_factor(decision_confidence)
        if self.config.enable_confidence_scaling:
            reasoning_parts.append(f"Confidence factor={confidence_factor:.2f}")

        # 4. Calculate Kelly factor
        kelly_factor = self._get_kelly_factor(symbol)
        if self.config.enable_kelly_sizing and kelly_factor != 1.0:
            reasoning_parts.append(f"Kelly factor={kelly_factor:.2f}")

        # 5. Combine all factors
        base_size = self.config.base_position_fraction
        combined_fraction = (
            base_size * regime_factor * volatility_factor * confidence_factor * kelly_factor
        )

        # 6. Apply position limits
        clamped_fraction = max(
            self.config.min_position_fraction,
            min(self.config.max_position_fraction, combined_fraction),
        )

        # 7. Check portfolio heat limit
        portfolio_heat = self._calculate_portfolio_heat(existing_positions)
        available_heat = self.config.max_portfolio_heat - portfolio_heat
        if available_heat < clamped_fraction:
            clamped_fraction = max(0.0, available_heat)
            reasoning_parts.append(f"Limited by portfolio heat ({portfolio_heat:.2%})")

        # 8. Calculate actual position values
        position_value = Decimal(str(float(equity) * clamped_fraction))
        position_quantity = position_value / current_price if current_price > 0 else Decimal("0")

        # 9. Calculate risk metrics
        estimated_risk = self._estimate_risk(clamped_fraction, stop_loss_distance, current_price)
        risk_reward = self._calculate_risk_reward(stop_loss_distance, take_profit_distance)

        # 10. Get ATR value for metadata
        atr_value = self._get_atr_value(symbol)

        return SizingResult(
            position_fraction=clamped_fraction,
            position_value=position_value,
            position_quantity=position_quantity,
            base_size=base_size,
            regime_factor=regime_factor,
            volatility_factor=volatility_factor,
            confidence_factor=confidence_factor,
            kelly_factor=kelly_factor,
            estimated_risk=estimated_risk,
            risk_reward_ratio=risk_reward,
            regime=regime_state.regime.name,
            regime_confidence=regime_state.confidence,
            atr_value=atr_value,
            reasoning="; ".join(reasoning_parts),
        )

    def _get_regime_state(self, symbol: str) -> RegimeState:
        """Get current regime state for symbol."""
        if self.regime_detector is None:
            return RegimeState.unknown()

        return self.regime_detector.get_regime(symbol)

    def _get_regime_factor(self, regime_state: RegimeState) -> float:
        """Get position scaling factor based on regime."""
        regime_name = regime_state.regime.name
        base_factor = self.config.regime_scale_factors.get(regime_name, 1.0)

        # Scale by regime confidence (uncertain regime = more conservative)
        confidence_adjustment = 0.5 + (0.5 * regime_state.confidence)
        return base_factor * confidence_adjustment

    def _get_volatility_factor(self, symbol: str, current_price: Decimal) -> float:
        """Calculate volatility-based sizing factor.

        Uses ATR to target a specific volatility level.
        Higher volatility = smaller position size.
        """
        if not self.config.enable_volatility_scaling:
            return 1.0

        atr_value = self._get_atr_value(symbol)
        if atr_value is None or float(current_price) <= 0:
            return 1.0

        # Calculate ATR as percentage of price
        atr_percent = atr_value / float(current_price)

        if atr_percent <= 0:
            return 1.0

        # Target volatility scaling
        # If ATR% > target, reduce size proportionally
        volatility_factor = self.config.target_volatility / atr_percent

        # Clamp to reasonable range (0.25x to 2x)
        return max(0.25, min(2.0, volatility_factor))

    def _get_confidence_factor(self, decision_confidence: float) -> float:
        """Calculate confidence-based sizing factor.

        Higher confidence = larger position size (up to a point).
        """
        if not self.config.enable_confidence_scaling:
            return 1.0

        # Clamp confidence to 0-1
        confidence = max(0.0, min(1.0, decision_confidence))

        # Apply exponent to make scaling non-linear
        # With exponent=1.5: 0.5 confidence -> 0.35x, 0.8 -> 0.72x, 1.0 -> 1.0x
        return confidence**self.config.confidence_exponent

    def _get_kelly_factor(self, symbol: str) -> float:
        """Calculate Kelly criterion sizing factor.

        Kelly fraction = (p * b - q) / b
        where:
            p = probability of winning
            q = probability of losing (1 - p)
            b = win/loss ratio

        We use quarter-Kelly for safety.
        """
        if not self.config.enable_kelly_sizing:
            return 1.0

        wins, losses = self._win_rates.get(symbol, (0, 0))
        total = wins + losses

        if total < 10:  # Need minimum sample size
            return 1.0

        win_rate = wins / total

        if win_rate < self.config.min_win_rate_for_kelly:
            return 0.5  # Below threshold, reduce size

        # Assume 1:1 risk/reward for simplicity
        # Full Kelly = 2 * win_rate - 1
        full_kelly = max(0.0, 2 * win_rate - 1)

        # Apply Kelly fraction (quarter Kelly = 0.25)
        kelly_adjusted = full_kelly * self.config.kelly_fraction

        # Convert to multiplier (base 1.0)
        return 0.5 + kelly_adjusted  # Range: 0.5 to ~0.75 for typical win rates

    def _calculate_portfolio_heat(self, existing_positions: dict[str, float] | None) -> float:
        """Calculate total portfolio risk (heat)."""
        if existing_positions is None:
            existing_positions = self._portfolio_positions

        return sum(existing_positions.values())

    def _estimate_risk(
        self,
        position_fraction: float,
        stop_loss_distance: Decimal | None,
        current_price: Decimal,
    ) -> float:
        """Estimate risk for this position."""
        if stop_loss_distance is not None and current_price > 0:
            # Risk = position size * (stop distance / price)
            stop_percent = float(stop_loss_distance) / float(current_price)
            return position_fraction * stop_percent

        # Default: assume 2x ATR stop = ~4% typical move
        return position_fraction * 0.04

    def _calculate_risk_reward(
        self,
        stop_loss_distance: Decimal | None,
        take_profit_distance: Decimal | None,
    ) -> float:
        """Calculate risk/reward ratio."""
        if stop_loss_distance is None or take_profit_distance is None:
            return 0.0

        if stop_loss_distance <= 0:
            return 0.0

        return float(take_profit_distance) / float(stop_loss_distance)

    def _get_atr_value(self, symbol: str) -> float | None:
        """Get current ATR value for symbol."""
        if self.regime_detector is None:
            return None

        indicators = self.regime_detector.get_indicator_values(symbol)
        return indicators.get("atr")

    def record_trade_result(self, symbol: str, is_win: bool, risk_amount: float = 0.0) -> None:
        """Record trade result for Kelly calculation.

        Args:
            symbol: Trading symbol
            is_win: Whether trade was profitable
            risk_amount: Risk amount to remove from portfolio heat
        """
        wins, losses = self._win_rates.get(symbol, (0, 0))
        if is_win:
            wins += 1
        else:
            losses += 1
        self._win_rates[symbol] = (wins, losses)

        # Update portfolio heat
        if symbol in self._portfolio_positions:
            self._portfolio_positions[symbol] -= risk_amount
            if self._portfolio_positions[symbol] <= 0:
                del self._portfolio_positions[symbol]

    def add_position_risk(self, symbol: str, risk_amount: float) -> None:
        """Track position risk for portfolio heat.

        Args:
            symbol: Trading symbol
            risk_amount: Risk amount as fraction of equity
        """
        current = self._portfolio_positions.get(symbol, 0.0)
        self._portfolio_positions[symbol] = current + risk_amount

    def get_portfolio_heat(self) -> dict[str, Any]:
        """Get current portfolio heat summary."""
        total_heat = sum(self._portfolio_positions.values())
        return {
            "total_heat": round(total_heat, 4),
            "max_heat": self.config.max_portfolio_heat,
            "available_heat": round(self.config.max_portfolio_heat - total_heat, 4),
            "positions": {
                symbol: round(risk, 4) for symbol, risk in self._portfolio_positions.items()
            },
        }

    def serialize_state(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "win_rates": self._win_rates,
            "portfolio_positions": self._portfolio_positions,
        }

    def deserialize_state(self, state: dict[str, Any]) -> None:
        """Restore state from serialized data."""
        self._win_rates = {k: tuple(v) for k, v in state.get("win_rates", {}).items()}
        self._portfolio_positions = state.get("portfolio_positions", {})


__all__ = [
    "PositionSizer",
    "PositionSizingConfig",
    "SizingResult",
]
