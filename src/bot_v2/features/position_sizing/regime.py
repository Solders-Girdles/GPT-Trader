"""
Market Regime-Based Position Sizing - Complete Isolation

Adjusts position sizes based on current market regime.
No external dependencies - all logic local to this slice.
"""

import math

import numpy as np

from bot_v2.errors import ValidationError, log_error
from bot_v2.features.position_sizing.types import RegimeMultipliers
from bot_v2.utilities.logging_patterns import get_logger
from bot_v2.validation import ChoiceValidator, RangeValidator

logger = get_logger(__name__, component="position_sizing")


def regime_adjusted_size(
    base_size: float, market_regime: str, multipliers: RegimeMultipliers
) -> tuple[float, str]:
    """
    Adjust position size based on current market regime with validation.

    Different market regimes require different position sizing approaches:
    - Bull markets: Can afford larger positions
    - Bear markets: Need smaller, defensive positions
    - Crisis periods: Emergency position reduction

    Args:
        base_size: Base position size (0-1 as fraction of portfolio)
        market_regime: Current market regime identifier
        multipliers: Regime-specific position multipliers

    Returns:
        Tuple of (adjusted_size, explanation)

    Raises:
        ValidationError: If inputs are invalid
    """
    try:
        # Validate inputs
        RangeValidator(min_value=0.0)(base_size, "base_size")

        if not market_regime:
            logger.warning("No regime data provided, using base size")
            return base_size, "No regime data available, using base size"

        # Validate regime
        valid_regimes = [
            "bull_quiet",
            "bull_volatile",
            "bear_quiet",
            "bear_volatile",
            "sideways_quiet",
            "sideways_volatile",
            "crisis",
        ]
        ChoiceValidator(valid_regimes)(market_regime, "market_regime")

        if not multipliers:
            raise ValidationError("Regime multipliers required", field="multipliers")

        # Get multiplier with validation
        multiplier = multipliers.get_multiplier(market_regime)

        # Validate multiplier bounds
        if multiplier < 0 or multiplier > 5.0:
            logger.warning(f"Extreme multiplier {multiplier:.2f} for regime {market_regime}")
            multiplier = max(0.1, min(3.0, multiplier))  # Clamp to reasonable bounds

        adjusted_size = base_size * multiplier

        # Ensure adjusted size doesn't become negative or extreme
        adjusted_size = max(0.0, min(1.0, adjusted_size))

        if base_size >= 0.5 and adjusted_size > base_size:
            adjusted_size = base_size

        explanation = f"Regime '{market_regime}' → {multiplier:.2f}x multiplier"

        return adjusted_size, explanation

    except ValidationError:
        raise
    except Exception as e:
        error = ValidationError(f"Regime adjustment failed: {e}", field="regime_adjustment")
        log_error(error)
        raise error


def dynamic_regime_multipliers(
    regime_history: list[tuple[str, float]], volatility_adjustment: bool = True
) -> RegimeMultipliers:
    """
    Create dynamic regime multipliers based on recent regime performance.

    Args:
        regime_history: List of (regime, performance) tuples from recent trades
        volatility_adjustment: Whether to adjust for regime volatility

    Returns:
        Dynamically adjusted regime multipliers
    """
    # Start with default multipliers
    multipliers = RegimeMultipliers()

    if not regime_history:
        return multipliers

    # Calculate performance by regime
    regime_performance: dict[str, list[float]] = {}
    regime_counts: dict[str, int] = {}

    for regime, performance in regime_history:
        if regime not in regime_performance:
            regime_performance[regime] = []
            regime_counts[regime] = 0

        regime_performance[regime].append(performance)
        regime_counts[regime] += 1

    # Adjust multipliers based on performance
    adjustments: dict[str, float] = {}
    for regime, performances in regime_performance.items():
        if len(performances) >= 2:
            avg_performance = float(np.mean(performances))
            volatility = float(np.std(performances))

            # Positive performance increases multiplier, negative decreases
            performance_adjustment = avg_performance * 2  # Scale factor

            # High volatility decreases multiplier (more risk)
            if volatility_adjustment:
                volatility_penalty = min(0.5, volatility * 0.5)
                performance_adjustment -= volatility_penalty

            adjustments[regime] = performance_adjustment

    # Apply adjustments to base multipliers
    base_values = {
        "bull_quiet": multipliers.bull_quiet,
        "bull_volatile": multipliers.bull_volatile,
        "bear_quiet": multipliers.bear_quiet,
        "bear_volatile": multipliers.bear_volatile,
        "sideways_quiet": multipliers.sideways_quiet,
        "sideways_volatile": multipliers.sideways_volatile,
        "crisis": multipliers.crisis,
    }

    for regime, base_value in base_values.items():
        if regime in adjustments:
            new_value = base_value + adjustments[regime]
            # Keep within reasonable bounds
            new_value = max(0.1, min(2.0, new_value))
            setattr(multipliers, regime, new_value)

    return multipliers


def regime_transition_adjustment(
    current_regime: str, previous_regime: str, transition_confidence: float, base_multiplier: float
) -> float:
    """
    Adjust position size during regime transitions.

    During uncertain regime transitions, reduce position sizes to manage risk.

    Args:
        current_regime: Current detected regime
        previous_regime: Previous regime
        transition_confidence: Confidence in the regime detection (0-1)
        base_multiplier: Base regime multiplier

    Returns:
        Transition-adjusted multiplier
    """
    if current_regime == previous_regime:
        # No transition, use full multiplier
        return base_multiplier

    if transition_confidence < 0.7:
        conservative_factor = 0.5 + (transition_confidence - 0.5) * 2.5
        conservative_factor = max(0.5, min(1.0, conservative_factor))
        return base_multiplier * conservative_factor

    # High confidence transition, use full multiplier
    return base_multiplier


def regime_momentum_factor(regime_duration_days: int, regime: str) -> float:
    """
    Calculate momentum factor based on how long regime has persisted.

    Longer-lasting regimes may have more momentum, justifying larger positions.

    Args:
        regime_duration_days: Days the current regime has persisted
        regime: Current regime type

    Returns:
        Momentum factor (0.8 - 1.2)
    """
    # Crisis regimes don't get momentum bonuses
    if regime == "crisis":
        return 0.8

    # Bull regimes can benefit more from momentum
    if "bull" in regime:
        if regime_duration_days > 20:
            return min(1.2, 1.0 + (regime_duration_days - 20) * 0.02)
        return 1.0

    # Bear regimes get diminishing momentum (mean reversion expectation)
    if "bear" in regime:
        if regime_duration_days > 15:
            return max(0.8, 1.0 - (regime_duration_days - 15) * 0.01)
        return 1.0

    # Sideways regimes get slight momentum bonus
    if "sideways" in regime:
        if regime_duration_days > 30:
            return min(1.1, 1.0 + (regime_duration_days - 30) * 0.005)
        else:
            return 1.0

    return 1.0


def portfolio_regime_allocation(
    current_regimes: dict[str, str], regime_confidences: dict[str, float], total_risk_budget: float
) -> dict[str, float]:
    """
    Allocate portfolio risk budget across assets based on their regimes.

    Args:
        current_regimes: Dict of {symbol: regime}
        regime_confidences: Dict of {symbol: confidence}
        total_risk_budget: Total risk budget to allocate (0-1)

    Returns:
        Dict of {symbol: risk_allocation}
    """
    if not current_regimes:
        return {}

    # Calculate base allocations by regime
    regime_weights = {
        "bull_quiet": 1.5,
        "bull_volatile": 1.2,
        "sideways_quiet": 1.0,
        "sideways_volatile": 0.8,
        "bear_quiet": 0.6,
        "bear_volatile": 0.4,
        "crisis": 0.2,
    }

    # Calculate weighted scores for each symbol
    symbol_scores: dict[str, float] = {}
    for symbol, regime in current_regimes.items():
        base_weight = regime_weights.get(regime, 1.0)
        confidence = regime_confidences.get(symbol, 0.5)

        # Combine regime weight with confidence
        symbol_scores[symbol] = base_weight * confidence

    # Normalize to total risk budget
    total_score = float(sum(symbol_scores.values()))
    if total_score == 0:
        return {symbol: 0.0 for symbol in current_regimes}

    allocations: dict[str, float] = {}
    for symbol, score in symbol_scores.items():
        allocations[symbol] = (score / total_score) * total_risk_budget

    return allocations


def regime_correlation_adjustment(
    symbol_regimes: dict[str, str], correlation_matrix: dict[tuple[str, str], float] | None = None
) -> dict[str, float]:
    """
    Adjust position sizes based on regime correlations across portfolio.

    If multiple assets are in similar regimes, reduce individual allocations
    to avoid over-concentration.

    Args:
        symbol_regimes: Dict of {symbol: regime}
        correlation_matrix: Optional correlation data between symbols

    Returns:
        Dict of {symbol: correlation_adjustment_factor}
    """
    if not symbol_regimes or len(symbol_regimes) == 1:
        return {symbol: 1.0 for symbol in symbol_regimes}

    # Count symbols in each regime
    regime_counts: dict[str, int] = {}
    for regime in symbol_regimes.values():
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    # Calculate concentration penalty for each regime
    total_symbols = len(symbol_regimes)
    adjustments = {}

    for symbol, regime in symbol_regimes.items():
        regime_concentration = regime_counts[regime] / total_symbols

        # Penalize high concentration
        if regime_concentration > 0.5:
            # More than half the portfolio in same regime
            concentration_penalty = 0.7 + (0.5 - regime_concentration) * 0.6
        else:
            concentration_penalty = 1.0

        adjustments[symbol] = concentration_penalty

    return adjustments


def regime_volatility_scaling(
    regime: str, realized_volatility: float, expected_volatility: float
) -> float:
    """
    Scale position size based on regime-specific volatility expectations.

    Args:
        regime: Current market regime
        realized_volatility: Recent realized volatility
        expected_volatility: Expected volatility for this regime

    Returns:
        Volatility-based scaling factor
    """
    if expected_volatility <= 0:
        return 1.0

    volatility_ratio = realized_volatility / expected_volatility
    sqrt_ratio = math.sqrt(volatility_ratio)

    # Different regimes handle volatility differently
    if regime == "crisis":
        return min(0.5, 1.0 / max(1.0, volatility_ratio * volatility_ratio))
    elif "volatile" in regime:
        return max(0.9, min(1.1, 1.0 / max(1.0, volatility_ratio)))
    elif "quiet" in regime:
        return max(0.8, min(1.0, 1.0 / max(1.0, volatility_ratio)))
    else:
        return max(0.85, min(0.95, 1.0 / max(0.9, volatility_ratio)))


def validate_regime_inputs(regime: str, multipliers: RegimeMultipliers) -> list[str]:
    """
    Validate regime-based position sizing inputs using new validation framework.

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    valid_regimes = [
        "bull_quiet",
        "bull_volatile",
        "bear_quiet",
        "bear_volatile",
        "sideways_quiet",
        "sideways_volatile",
        "crisis",
    ]

    try:
        if regime:  # Only validate if regime is provided
            ChoiceValidator(valid_regimes)(regime, "regime")
    except ValidationError as e:
        errors.append(e.message)

    # Check multiplier ranges using validation framework
    multiplier_checks = [
        ("bull_quiet", multipliers.bull_quiet),
        ("bull_volatile", multipliers.bull_volatile),
        ("bear_quiet", multipliers.bear_quiet),
        ("bear_volatile", multipliers.bear_volatile),
        ("sideways_quiet", multipliers.sideways_quiet),
        ("sideways_volatile", multipliers.sideways_volatile),
        ("crisis", multipliers.crisis),
    ]

    for regime_name, multiplier in multiplier_checks:
        try:
            RangeValidator(min_value=0.0, max_value=5.0)(multiplier, f"{regime_name}_multiplier")
        except ValidationError as e:
            errors.append(e.message)

    return errors


def safe_regime_calculation(regime: str, base_multiplier: float, confidence: float = 1.0) -> float:
    """
    Safely calculate regime multiplier with bounds checking.

    Args:
        regime: Market regime identifier
        base_multiplier: Base multiplier for the regime
        confidence: Confidence in regime detection (0-1)

    Returns:
        Safe regime multiplier
    """
    try:
        # Validate inputs
        valid_regimes = [
            "bull_quiet",
            "bull_volatile",
            "bear_quiet",
            "bear_volatile",
            "sideways_quiet",
            "sideways_volatile",
            "crisis",
        ]

        if regime not in valid_regimes:
            logger.warning(f"Unknown regime {regime}, using neutral multiplier")
            return 1.0

        RangeValidator(min_value=0.0, max_value=1.0)(confidence, "confidence")

        clamped_multiplier = max(0.1, min(3.0, base_multiplier))
        adjusted_multiplier = clamped_multiplier * confidence + (1.0 - confidence)

        return max(0.1, min(3.0, adjusted_multiplier))

    except ValidationError as e:
        log_error(e)
        logger.warning("Regime calculation failed, using conservative multiplier")
        return 0.8
