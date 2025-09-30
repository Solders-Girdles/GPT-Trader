"""
Confidence-Based Position Sizing - Complete Isolation

Adjusts position sizes based on model confidence scores.
No external dependencies - all calculations local to this slice.
"""

import logging
import math

import numpy as np
from bot_v2.errors import ValidationError, log_error
from bot_v2.features.position_sizing.types import ConfidenceAdjustment
from bot_v2.validation import PercentageValidator, RangeValidator

logger = logging.getLogger(__name__)


def confidence_adjusted_size(
    base_size: float, confidence: float, adjustment: ConfidenceAdjustment
) -> tuple[float, str]:
    """
    Adjust position size based on model confidence.

    Higher confidence = larger positions (up to max_adjustment)
    Lower confidence = smaller positions (down to min threshold)

    Args:
        base_size: Base position size (0-1 as fraction of portfolio)
        confidence: Model confidence score (0-1)
        adjustment: Confidence adjustment parameters

    Returns:
        Tuple of (adjusted_size, explanation)
    """
    try:
        # Validate inputs
        PercentageValidator(as_decimal=True)(confidence, "confidence")
        RangeValidator(min_value=0.0)(base_size, "base_size")

        if not isinstance(adjustment, ConfidenceAdjustment):
            raise ValidationError("Invalid adjustment parameters", field="adjustment")

    except ValidationError as e:
        log_error(e)
        return 0.0, f"Validation error: {e.message}"

    # Below minimum confidence threshold
    if confidence < adjustment.min_confidence:
        reduction_factor = confidence / adjustment.min_confidence
        adjusted_size = base_size * reduction_factor * 0.5  # Extra conservative
        explanation = f"Low confidence ({confidence:.2f} < {adjustment.min_confidence:.2f}), reduced to {reduction_factor*0.5:.2f}x"
        return adjusted_size, explanation

    # Calculate adjustment multiplier based on curve type
    if adjustment.adjustment_curve == "linear":
        multiplier = _linear_confidence_curve(confidence, adjustment)
    elif adjustment.adjustment_curve == "exponential":
        multiplier = _exponential_confidence_curve(confidence, adjustment)
    elif adjustment.adjustment_curve == "sigmoid":
        multiplier = _sigmoid_confidence_curve(confidence, adjustment)
    else:
        multiplier = _linear_confidence_curve(confidence, adjustment)

    adjusted_size = base_size * multiplier
    explanation = f"Confidence {confidence:.2f} → {multiplier:.2f}x multiplier ({adjustment.adjustment_curve} curve)"

    return adjusted_size, explanation


def _linear_confidence_curve(confidence: float, adjustment: ConfidenceAdjustment) -> float:
    """Linear confidence adjustment curve."""
    if confidence <= adjustment.min_confidence:
        return 0.5  # Conservative below threshold

    # Linear interpolation from min_confidence to 1.0
    confidence_range = 1.0 - adjustment.min_confidence
    confidence_excess = confidence - adjustment.min_confidence

    # Scale from 1.0 at min_confidence to max_adjustment at 1.0
    multiplier = 1.0 + (adjustment.max_adjustment - 1.0) * (confidence_excess / confidence_range)
    return min(adjustment.max_adjustment, multiplier)


def _exponential_confidence_curve(confidence: float, adjustment: ConfidenceAdjustment) -> float:
    """Exponential confidence adjustment curve - more aggressive scaling."""
    if confidence <= adjustment.min_confidence:
        return 0.5

    # Exponential curve: multiplier = 1 + (max_adj - 1) * ((conf - min_conf) / (1 - min_conf))^2
    confidence_range = 1.0 - adjustment.min_confidence
    confidence_excess = confidence - adjustment.min_confidence

    normalized_confidence = confidence_excess / confidence_range
    multiplier = 1.0 + (adjustment.max_adjustment - 1.0) * (normalized_confidence**2)

    return min(adjustment.max_adjustment, multiplier)


def _sigmoid_confidence_curve(confidence: float, adjustment: ConfidenceAdjustment) -> float:
    """Sigmoid confidence adjustment curve - smooth S-curve."""
    if confidence <= adjustment.min_confidence:
        return 0.5

    # Sigmoid transformation centered at 0.8 confidence
    center = 0.8
    steepness = 10  # Controls curve steepness

    # Transform confidence to sigmoid input
    x = steepness * (confidence - center)
    sigmoid_value = 1 / (1 + math.exp(-x))

    # Scale sigmoid output to adjustment range
    multiplier = 1.0 + (adjustment.max_adjustment - 1.0) * sigmoid_value
    return min(adjustment.max_adjustment, multiplier)


def multi_model_confidence(confidences: list[float], weights: list[float] | None = None) -> float:
    """
    Combine confidence scores from multiple models.

    Args:
        confidences: List of confidence scores from different models
        weights: Optional weights for each model (defaults to equal weighting)

    Returns:
        Combined confidence score (0-1)
    """
    if not confidences:
        return 0.0

    if weights is None:
        weights = [1.0] * len(confidences)

    if len(weights) != len(confidences):
        weights = [1.0] * len(confidences)

    # Weighted average
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(conf * weight for conf, weight in zip(confidences, weights, strict=False))
    return weighted_sum / total_weight


def confidence_decay(
    initial_confidence: float, time_since_prediction: float, half_life_hours: float = 24.0
) -> float:
    """
    Apply time decay to confidence scores.

    Confidence naturally decays over time as market conditions change.

    Args:
        initial_confidence: Original confidence score
        time_since_prediction: Hours since prediction was made
        half_life_hours: Hours for confidence to decay to half (default 24)

    Returns:
        Time-decayed confidence score
    """
    if time_since_prediction <= 0:
        return initial_confidence

    # Exponential decay: conf(t) = conf(0) * 0.5^(t/half_life)
    decay_factor = 0.5 ** (time_since_prediction / half_life_hours)
    return initial_confidence * decay_factor


def confidence_from_backtest_metrics(
    sharpe_ratio: float, win_rate: float, profit_factor: float, max_drawdown: float
) -> float:
    """
    Generate confidence score from backtest performance metrics.

    Args:
        sharpe_ratio: Risk-adjusted return measure
        win_rate: Percentage of winning trades (0-1)
        profit_factor: Total gains / total losses
        max_drawdown: Maximum drawdown experienced (0-1)

    Returns:
        Confidence score (0-1)
    """
    # Component confidence scores
    sharpe_conf = min(1.0, max(0.0, sharpe_ratio / 2.0))  # Normalized to 0-1
    win_rate_conf = win_rate  # Already 0-1
    profit_conf = min(1.0, max(0.0, (profit_factor - 1.0) / 2.0))  # >1 is good
    drawdown_conf = max(0.0, 1.0 - max_drawdown * 5.0)  # Penalize high drawdown

    # Weighted combination
    weights = [0.3, 0.25, 0.25, 0.2]  # Sharpe gets highest weight
    components = [sharpe_conf, win_rate_conf, profit_conf, drawdown_conf]

    confidence = sum(comp * weight for comp, weight in zip(components, weights, strict=False))
    return max(0.0, min(1.0, confidence))


def adaptive_confidence_threshold(
    recent_performance: list[float], base_threshold: float = 0.6
) -> float:
    """
    Adapt confidence threshold based on recent strategy performance.

    If strategy has been performing well, lower the threshold (more trades).
    If strategy has been performing poorly, raise the threshold (fewer trades).

    Args:
        recent_performance: List of recent trade returns
        base_threshold: Base confidence threshold

    Returns:
        Adapted confidence threshold
    """
    if not recent_performance or len(recent_performance) < 5:
        return base_threshold

    # Calculate recent performance metrics
    avg_return = np.mean(recent_performance)
    win_rate = len([r for r in recent_performance if r > 0]) / len(recent_performance)

    # Positive performance lowers threshold, negative raises it
    performance_adjustment = avg_return * 2  # Scale factor
    win_rate_adjustment = (win_rate - 0.5) * 0.2  # Adjustment based on win rate

    adapted_threshold = base_threshold - performance_adjustment - win_rate_adjustment

    # Keep within reasonable bounds
    return max(0.3, min(0.9, adapted_threshold))


def confidence_position_limits(confidence: float, base_max_position: float) -> tuple[float, float]:
    """
    Calculate position size limits based on confidence.

    Args:
        confidence: Model confidence (0-1)
        base_max_position: Base maximum position size (0-1)

    Returns:
        Tuple of (min_position, max_position) sizes
    """
    if confidence < 0.5:
        # Very low confidence - severely limit position size
        max_pos = base_max_position * confidence * 0.5
        min_pos = 0.001  # Almost no minimum
    elif confidence < 0.7:
        # Moderate confidence - normal limits but conservative max
        max_pos = base_max_position * 0.8
        min_pos = base_max_position * 0.1
    else:
        # High confidence - allow larger positions
        max_pos = base_max_position * min(2.0, 1.0 + confidence)
        min_pos = base_max_position * 0.05

    return min_pos, max_pos


def confidence_risk_budget(confidence: float, total_risk_budget: float) -> float:
    """
    Allocate risk budget based on confidence score.

    Higher confidence gets larger share of total risk budget.

    Args:
        confidence: Model confidence (0-1)
        total_risk_budget: Total risk budget available (0-1)

    Returns:
        Risk budget allocation for this position
    """
    if confidence < 0.5:
        # Low confidence gets minimal risk budget
        return total_risk_budget * 0.1 * confidence
    elif confidence < 0.7:
        # Moderate confidence gets proportional allocation
        return total_risk_budget * 0.5 * confidence
    else:
        # High confidence gets larger allocation
        return total_risk_budget * min(1.0, confidence * 1.2)


def validate_confidence_inputs(confidence: float, adjustment: ConfidenceAdjustment) -> list[str]:
    """
    Validate confidence adjustment inputs using new validation framework.

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    try:
        PercentageValidator(as_decimal=True)(confidence, "confidence")
    except ValidationError as e:
        errors.append(e.message)

    try:
        RangeValidator(min_value=0.0, max_value=1.0, inclusive=False)(
            adjustment.min_confidence, "min_confidence"
        )
    except ValidationError as e:
        errors.append(e.message)

    try:
        RangeValidator(min_value=0.0, inclusive=False)(adjustment.max_adjustment, "max_adjustment")
    except ValidationError as e:
        errors.append(e.message)

    if adjustment.adjustment_curve not in ["linear", "exponential", "sigmoid"]:
        errors.append(f"Invalid adjustment curve: {adjustment.adjustment_curve}")

    return errors


def safe_confidence_calculation(
    confidences: list[float], weights: list[float] | None = None
) -> float:
    """
    Safely combine confidence scores with input validation.

    Args:
        confidences: List of confidence scores from different models
        weights: Optional weights for each model

    Returns:
        Combined confidence score (0-1)
    """
    try:
        if not confidences:
            return 0.0

        # Validate all confidence scores
        for i, conf in enumerate(confidences):
            PercentageValidator(as_decimal=True)(conf, f"confidence[{i}]")

        if weights is not None:
            if len(weights) != len(confidences):
                logger.warning("Weight count mismatch, using equal weights")
                weights = None
            else:
                # Validate weights are positive
                for i, weight in enumerate(weights):
                    RangeValidator(min_value=0.0)(weight, f"weight[{i}]")

        return multi_model_confidence(confidences, weights)

    except ValidationError as e:
        log_error(e)
        logger.warning("Confidence calculation failed, using conservative default")
        return 0.5  # Conservative default
