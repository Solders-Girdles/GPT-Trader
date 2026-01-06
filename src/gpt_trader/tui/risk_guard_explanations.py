"""Risk guard threshold explanations.

Generates human-readable explanations for why risk guards trigger,
using current risk state values and threshold logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpt_trader.tui.types import RiskState


def get_guard_explanation(guard_name: str, state: RiskState) -> str:
    """Generate a human-readable explanation for a risk guard.

    Args:
        guard_name: Name of the guard (e.g., "DailyLossGuard").
        state: Current RiskState with threshold and measured values.

    Returns:
        Explanation string describing when the guard triggers and current value.
    """
    # Normalize guard name for matching
    name_lower = guard_name.lower().replace("_", "").replace("-", "")

    # Daily loss guard
    if "dailyloss" in name_lower or "losslimit" in name_lower:
        return _explain_daily_loss_guard(state)

    # Max leverage guard
    if "leverage" in name_lower or "maxlev" in name_lower:
        return _explain_leverage_guard(state)

    # Position limit guard
    if "position" in name_lower and "limit" in name_lower:
        return _explain_position_limit_guard(state)

    # Reduce-only mode guard
    if "reduceonly" in name_lower or "reduce_only" in name_lower:
        return _explain_reduce_only_guard(state)

    # Drawdown guard
    if "drawdown" in name_lower:
        return _explain_drawdown_guard(state)

    # Volatility guard
    if "volatility" in name_lower or "vol" in name_lower:
        return _explain_volatility_guard(state)

    # Rate limit guard
    if "ratelimit" in name_lower or "rate_limit" in name_lower:
        return _explain_rate_limit_guard()

    # Fallback for unknown guards
    return "Guard active (threshold details not available)"


def _explain_daily_loss_guard(state: RiskState) -> str:
    """Explain daily loss guard threshold."""
    if state.daily_loss_limit_pct <= 0:
        return "triggers when daily loss limit is breached (limit not configured)"

    limit_pct = state.daily_loss_limit_pct * 100
    current_pct = abs(state.current_daily_loss_pct) * 100
    usage_pct = (current_pct / limit_pct) * 100 if limit_pct > 0 else 0

    # Standard threshold is 75% of limit
    threshold_pct = 75
    return f"triggers at >= {threshold_pct}% of daily loss limit " f"(current {usage_pct:.0f}%)"


def _explain_leverage_guard(state: RiskState) -> str:
    """Explain max leverage guard threshold."""
    if state.max_leverage <= 0:
        return "triggers when leverage exceeds maximum (limit not configured)"

    # Assume current leverage would be tracked separately
    # For now, show the threshold
    return f"triggers at >= {state.max_leverage:.1f}x leverage"


def _explain_position_limit_guard(state: RiskState) -> str:
    """Explain position limit guard threshold."""
    # Position count would come from portfolio state
    # Show generic explanation
    return "triggers when position count exceeds configured limit"


def _explain_reduce_only_guard(state: RiskState) -> str:
    """Explain reduce-only mode guard."""
    if state.reduce_only_mode:
        reason = state.reduce_only_reason or "manually enabled"
        return f"active - reduce-only mode enabled ({reason})"
    return "triggers when reduce-only mode is enabled"


def _explain_drawdown_guard(state: RiskState) -> str:
    """Explain drawdown guard threshold."""
    return "triggers when portfolio drawdown exceeds threshold"


def _explain_volatility_guard(state: RiskState) -> str:
    """Explain volatility guard threshold."""
    return "triggers when market volatility exceeds threshold"


def _explain_rate_limit_guard() -> str:
    """Explain rate limit guard threshold."""
    return "triggers when API rate limits are approached"


def get_guard_explanations(guard_names: list[str], state: RiskState) -> list[tuple[str, str]]:
    """Generate explanations for multiple guards.

    Args:
        guard_names: List of guard names to explain.
        state: Current RiskState with threshold and measured values.

    Returns:
        List of (guard_name, explanation) tuples.
    """
    return [(name, get_guard_explanation(name, state)) for name in guard_names]
