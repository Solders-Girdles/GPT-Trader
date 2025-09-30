"""
Kelly Criterion Implementation - Complete Isolation

Pure Kelly Criterion calculations with no external dependencies.
All math and utilities implemented locally within this slice.
"""

import logging

from bot_v2.errors import RiskLimitExceeded, ValidationError, log_error
from bot_v2.features.position_sizing.types import RiskParameters, TradeStatistics
from bot_v2.validation import PositiveNumberValidator, RangeValidator

logger = logging.getLogger(__name__)


def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate Kelly Criterion optimal fraction with comprehensive validation.

    Formula: f* = (bp - q) / b
    Where:
    - f* = Kelly fraction
    - b = odds received on the wager (avg_win / avg_loss)
    - p = probability of winning (win_rate)
    - q = probability of losing (1 - win_rate)

    Args:
        win_rate: Probability of winning trade (0-1)
        avg_win: Average winning trade return (positive decimal, e.g., 0.05 for 5%)
        avg_loss: Average losing trade return (negative decimal, e.g., -0.03 for -3%)

    Returns:
        Kelly fraction (0-1), 0 if negative expected value

    Raises:
        ValidationError: If inputs are invalid
    """
    try:
        # Validate inputs
        RangeValidator(min_value=0.01, max_value=0.99)(win_rate, "win_rate")
        PositiveNumberValidator()(avg_win, "avg_win")

        if avg_loss >= 0:
            raise ValidationError("Average loss must be negative", field="avg_loss", value=avg_loss)

        # Check for division by zero risk
        if abs(avg_loss) < 1e-10:
            raise ValidationError(
                "Average loss too close to zero", field="avg_loss", value=avg_loss
            )

        # Calculate odds ratio (b) safely
        odds_ratio = abs(avg_win / avg_loss)

        # Check for reasonable odds ratio
        if odds_ratio > 100:  # Extreme odds ratio
            logger.warning(f"Extreme odds ratio {odds_ratio:.2f}, capping Kelly calculation")
            odds_ratio = 100

        # Calculate Kelly fraction: f* = (bp - q) / b
        # Simplified: f* = p - q/b = p - (1-p)/b
        kelly_fraction = win_rate - (1 - win_rate) / odds_ratio

        # Kelly fraction should not be negative (negative expected value)
        kelly_fraction = max(0.0, kelly_fraction)

        # Cap extreme Kelly values for safety
        if kelly_fraction > 0.5:
            logger.warning(
                f"High Kelly fraction {kelly_fraction:.4f}, consider using fractional Kelly"
            )

        return kelly_fraction

    except ValidationError:
        raise
    except Exception as e:
        error = ValidationError(f"Kelly calculation failed: {e}", field="kelly_calculation")
        log_error(error)
        raise error


def fractional_kelly(
    win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.25
) -> float:
    """
    Calculate fractional Kelly for more conservative sizing.

    Most practitioners use 1/4 or 1/2 Kelly to reduce volatility
    while maintaining good growth characteristics.

    Args:
        win_rate: Probability of winning trade (0-1)
        avg_win: Average winning trade return (positive)
        avg_loss: Average losing trade return (negative)
        fraction: Fraction of full Kelly to use (default 0.25 = quarter Kelly)

    Returns:
        Fractional Kelly position size
    """
    full_kelly = kelly_criterion(win_rate, avg_win, avg_loss)
    return full_kelly * fraction


def kelly_from_statistics(stats: TradeStatistics, fraction: float = 0.25) -> float:
    """
    Calculate Kelly fraction from trade statistics.

    Args:
        stats: Historical trade statistics
        fraction: Fraction of full Kelly to use

    Returns:
        Fractional Kelly position size
    """
    if stats.total_trades < 10:  # Need minimum sample size
        return 0.0

    return fractional_kelly(
        win_rate=stats.win_rate,
        avg_win=stats.avg_win_return,
        avg_loss=stats.avg_loss_return,
        fraction=fraction,
    )


def kelly_with_drawdown_protection(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    current_drawdown: float = 0.0,
    max_drawdown_threshold: float = 0.1,
) -> float:
    """
    Kelly calculation with drawdown protection.

    Reduces position size during drawdown periods to preserve capital.

    Args:
        win_rate: Probability of winning trade
        avg_win: Average winning trade return
        avg_loss: Average losing trade return
        current_drawdown: Current portfolio drawdown (0-1)
        max_drawdown_threshold: Threshold to start reducing size

    Returns:
        Drawdown-adjusted Kelly fraction
    """
    base_kelly = fractional_kelly(win_rate, avg_win, avg_loss)

    if current_drawdown <= max_drawdown_threshold:
        return base_kelly

    # Linear reduction based on drawdown severity
    drawdown_multiplier = max(0.1, 1.0 - (current_drawdown - max_drawdown_threshold) * 2)
    return base_kelly * drawdown_multiplier


def optimal_kelly_fraction(
    returns: list[float], test_fractions: list[float] | None = None
) -> tuple[float, float]:
    """
    Find optimal Kelly fraction by testing different fractions on historical data.

    Args:
        returns: List of historical trade returns
        test_fractions: Fractions to test (default: [0.1, 0.25, 0.5, 0.75, 1.0])

    Returns:
        Tuple of (optimal_fraction, final_wealth)
    """
    if not returns or len(returns) < 10:
        return 0.0, 1.0

    if test_fractions is None:
        test_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]

    stats = TradeStatistics.from_returns(returns)
    full_kelly = kelly_criterion(stats.win_rate, stats.avg_win_return, stats.avg_loss_return)

    if full_kelly <= 0:
        return 0.0, 1.0

    best_fraction = 0.0
    best_wealth = 1.0

    for fraction in test_fractions:
        kelly_size = full_kelly * fraction
        wealth = simulate_kelly_growth(returns, kelly_size)

        if wealth > best_wealth:
            best_wealth = wealth
            best_fraction = fraction

    return best_fraction, best_wealth


def simulate_kelly_growth(
    returns: list[float], kelly_fraction: float, initial_wealth: float = 1.0
) -> float:
    """
    Simulate wealth growth using Kelly sizing on historical returns.

    Args:
        returns: Historical trade returns
        kelly_fraction: Kelly fraction to use for position sizing
        initial_wealth: Starting wealth (default 1.0)

    Returns:
        Final wealth after applying Kelly sizing
    """
    wealth = initial_wealth

    for trade_return in returns:
        # Position size as fraction of current wealth
        position_size = kelly_fraction * wealth

        # Wealth change from this trade
        wealth_change = position_size * trade_return
        wealth += wealth_change

        # Prevent wealth from going negative (bankruptcy protection)
        wealth = max(0.01, wealth)

    return wealth


def kelly_position_value(
    portfolio_value: float,
    kelly_fraction: float,
    price_per_share: float,
    risk_params: RiskParameters,
) -> tuple[float, int]:
    """
    Convert Kelly fraction to actual position value and share count with safety checks.

    Args:
        portfolio_value: Total portfolio value
        kelly_fraction: Kelly fraction (0-1)
        price_per_share: Current share price
        risk_params: Risk parameters for position limits

    Returns:
        Tuple of (position_value, share_count)

    Raises:
        ValidationError: If inputs are invalid
        RiskLimitExceeded: If position exceeds risk limits
    """
    try:
        # Validate inputs
        PositiveNumberValidator()(portfolio_value, "portfolio_value")
        PositiveNumberValidator()(price_per_share, "price_per_share")
        RangeValidator(min_value=0.0, max_value=1.0)(kelly_fraction, "kelly_fraction")

        if not risk_params:
            raise ValidationError("Risk parameters required", field="risk_params")

        # Calculate raw Kelly position value
        raw_position_value = portfolio_value * kelly_fraction

        # Apply position size limits
        max_position_value = portfolio_value * risk_params.max_position_size
        min_position_value = portfolio_value * risk_params.min_position_size

        # Validate risk parameters
        if max_position_value <= 0:
            raise ValidationError("Max position value must be positive", field="max_position_size")

        if min_position_value < 0:
            raise ValidationError(
                "Min position value cannot be negative", field="min_position_size"
            )

        if min_position_value > max_position_value:
            raise ValidationError(
                "Min position size exceeds max position size", field="position_limits"
            )

        # Check if raw position exceeds max limit
        if raw_position_value > max_position_value:
            logger.warning(
                f"Kelly position ${raw_position_value:.2f} exceeds max ${max_position_value:.2f}"
            )

        # Clamp to risk limits
        position_value = max(min_position_value, min(max_position_value, raw_position_value))

        # Ensure we can afford at least one share
        if position_value < price_per_share:
            logger.warning(
                f"Position value ${position_value:.2f} less than share price ${price_per_share:.2f}"
            )
            return 0.0, 0

        # Convert to share count
        share_count = int(position_value / price_per_share)

        # Recalculate actual position value based on whole shares
        actual_position_value = share_count * price_per_share

        # Final safety check
        if actual_position_value > portfolio_value:
            raise RiskLimitExceeded(
                "Position value exceeds portfolio value",
                limit_type="portfolio_value",
                limit_value=portfolio_value,
                current_value=actual_position_value,
            )

        return actual_position_value, share_count

    except (ValidationError, RiskLimitExceeded):
        raise
    except Exception as e:
        error = ValidationError(
            f"Position value calculation failed: {e}", field="position_calculation"
        )
        log_error(error)
        raise error


def kelly_risk_metrics(kelly_fraction: float, avg_loss: float, portfolio_value: float) -> dict:
    """
    Calculate risk metrics for a Kelly position.

    Args:
        kelly_fraction: Kelly fraction being used
        avg_loss: Average losing trade return (negative)
        portfolio_value: Total portfolio value

    Returns:
        Dictionary of risk metrics
    """
    position_value = portfolio_value * kelly_fraction

    # Maximum expected loss on this position
    max_expected_loss = position_value * abs(avg_loss)

    # Maximum loss as percentage of portfolio
    max_loss_pct = max_expected_loss / portfolio_value

    # Risk-adjusted return expectation
    # This is simplified - in practice would use more sophisticated models
    risk_adjusted_return = kelly_fraction * 0.1  # Placeholder

    return {
        "position_value": position_value,
        "max_expected_loss": max_expected_loss,
        "max_loss_pct": max_loss_pct,
        "risk_adjusted_return": risk_adjusted_return,
        "kelly_fraction": kelly_fraction,
    }


def validate_kelly_inputs(win_rate: float, avg_win: float, avg_loss: float) -> list[str]:
    """
    Validate inputs for Kelly Criterion calculation.

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    if not (0 < win_rate < 1):
        errors.append(f"Win rate must be between 0 and 1, got {win_rate}")

    if avg_win <= 0:
        errors.append(f"Average win must be positive, got {avg_win}")

    if avg_loss >= 0:
        errors.append(f"Average loss must be negative, got {avg_loss}")

    if win_rate > 0 and avg_win > 0 and avg_loss < 0:
        # Check if expected value is positive
        expected_value = win_rate * avg_win + (1 - win_rate) * avg_loss
        if expected_value <= 0:
            errors.append(f"Strategy has negative expected value: {expected_value:.4f}")

    return errors
