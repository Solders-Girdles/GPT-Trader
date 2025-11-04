"""Kelly sizing adjustments for drawdowns and volatility regimes."""

from __future__ import annotations

from bot_v2.utilities.logging_patterns import get_logger

from .calculations import fractional_kelly

logger = get_logger(__name__, component="position_sizing")


def kelly_with_drawdown_protection(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    current_drawdown: float = 0.0,
    max_drawdown_threshold: float = 0.1,
) -> float:
    """Reduce Kelly sizing during drawdowns to preserve capital."""
    base_kelly = fractional_kelly(win_rate, avg_win, avg_loss)

    if current_drawdown <= max_drawdown_threshold:
        return base_kelly

    drawdown_multiplier = max(0.1, 1.0 - (current_drawdown - max_drawdown_threshold) * 2)
    return base_kelly * drawdown_multiplier


def kelly_with_volatility_scaling(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    recent_prices: list[float],
    fraction: float = 0.25,
    lookback_window: int = 100,
    high_vol_percentile: float = 0.75,
    high_vol_scaling: float = 0.5,
) -> tuple[float, dict]:
    """Scale fractional Kelly when realized volatility is elevated."""
    base_kelly = fractional_kelly(win_rate, avg_win, avg_loss, fraction)

    if len(recent_prices) < max(20, lookback_window):
        logger.warning(
            "Insufficient price history (%d) for volatility scaling, using base Kelly",
            len(recent_prices),
        )
        return base_kelly, {
            "base_kelly": base_kelly,
            "scaled_kelly": base_kelly,
            "regime": "insufficient_data",
            "scaling_factor": 1.0,
        }

    returns: list[float] = []
    for i in range(1, len(recent_prices)):
        if recent_prices[i - 1] != 0:
            ret = (recent_prices[i] - recent_prices[i - 1]) / recent_prices[i - 1]
            returns.append(ret)

    if len(returns) < 20:
        return base_kelly, {
            "base_kelly": base_kelly,
            "scaled_kelly": base_kelly,
            "regime": "insufficient_returns",
            "scaling_factor": 1.0,
        }

    mean_return = sum(returns) / len(returns)
    squared_diffs = [(r - mean_return) ** 2 for r in returns]
    variance = sum(squared_diffs) / len(squared_diffs)
    current_volatility = variance**0.5

    lookback_returns = returns[-min(lookback_window, len(returns)) :]
    volatilities: list[float] = []
    vol_window = 20
    for i in range(vol_window, len(lookback_returns)):
        window_returns = lookback_returns[i - vol_window : i]
        window_mean = sum(window_returns) / len(window_returns)
        window_squared_diffs = [(r - window_mean) ** 2 for r in window_returns]
        window_variance = sum(window_squared_diffs) / len(window_squared_diffs)
        volatilities.append(window_variance**0.5)

    if not volatilities:
        return base_kelly, {
            "base_kelly": base_kelly,
            "scaled_kelly": base_kelly,
            "current_vol": current_volatility,
            "regime": "insufficient_vol_history",
            "scaling_factor": 1.0,
        }

    sorted_vols = sorted(volatilities)
    percentile_idx = int(len(sorted_vols) * high_vol_percentile)
    vol_threshold = sorted_vols[min(percentile_idx, len(sorted_vols) - 1)]

    if current_volatility < 1e-4 or vol_threshold <= 1e-6:
        return base_kelly, {
            "base_kelly": base_kelly,
            "scaled_kelly": base_kelly,
            "current_vol": current_volatility,
            "vol_threshold": vol_threshold,
            "regime": "normal_volatility",
            "scaling_factor": 1.0,
        }

    ratio = current_volatility / vol_threshold if vol_threshold > 0 else float("inf")
    if ratio <= 1.5:
        scaling_factor = 1.0
        regime = "normal_volatility"
    elif current_volatility > vol_threshold:
        scaling_factor = high_vol_scaling
        regime = "high_volatility"
    else:
        scaling_factor = 1.0
        regime = "normal_volatility"

    scaled_kelly = base_kelly * scaling_factor
    metrics = {
        "base_kelly": base_kelly,
        "scaled_kelly": scaled_kelly,
        "current_vol": current_volatility,
        "vol_threshold": vol_threshold,
        "vol_percentile": high_vol_percentile,
        "regime": regime,
        "scaling_factor": scaling_factor,
        "annualized_vol": current_volatility * (252**0.5),
    }

    logger.info(
        "Kelly volatility scaling: regime=%s, vol=%.4f, threshold=%.4f, base=%.4f, scaled=%.4f",
        regime,
        current_volatility,
        vol_threshold,
        base_kelly,
        scaled_kelly,
    )

    return scaled_kelly, metrics


__all__ = ["kelly_with_drawdown_protection", "kelly_with_volatility_scaling"]
