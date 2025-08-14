"""Risk Management Utilities and Helper Functions.

This module provides utility functions for risk calculations,
risk limit configurations, and risk reporting across the system.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_position_risk(
    shares: int, current_price: float, entry_price: float, stop_loss: float, portfolio_value: float
) -> Dict[str, float]:
    """Calculate comprehensive risk metrics for a position.

    Args:
        shares: Number of shares
        current_price: Current market price
        entry_price: Entry price of position
        stop_loss: Stop-loss price level
        portfolio_value: Total portfolio value

    Returns:
        Dictionary of risk metrics
    """
    position_value = shares * current_price
    position_size_pct = position_value / portfolio_value if portfolio_value > 0 else 0

    # Risk per share
    risk_per_share = max(0, entry_price - stop_loss)
    total_risk = shares * risk_per_share
    risk_pct = total_risk / portfolio_value if portfolio_value > 0 else 0

    # P&L calculations
    unrealized_pnl = shares * (current_price - entry_price)
    unrealized_pnl_pct = unrealized_pnl / (shares * entry_price) if entry_price > 0 else 0

    # Distance to stop
    stop_distance = (current_price - stop_loss) / current_price if current_price > 0 else 0

    return {
        "position_value": position_value,
        "position_size_pct": position_size_pct,
        "risk_per_share": risk_per_share,
        "total_risk": total_risk,
        "risk_pct": risk_pct,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pnl_pct": unrealized_pnl_pct,
        "stop_distance": stop_distance,
        "shares": shares,
        "current_price": current_price,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
    }


def calculate_portfolio_metrics(
    positions: Dict[str, Dict[str, float]], portfolio_value: float
) -> Dict[str, float]:
    """Calculate portfolio-level risk metrics.

    Args:
        positions: Dictionary of position data {symbol: position_info}
        portfolio_value: Total portfolio value

    Returns:
        Portfolio risk metrics
    """
    if not positions or portfolio_value <= 0:
        return {
            "total_exposure": 0.0,
            "total_risk": 0.0,
            "largest_position": 0.0,
            "concentration_ratio": 0.0,
            "number_of_positions": 0,
            "avg_position_size": 0.0,
        }

    position_values = []
    total_risk = 0.0

    for symbol, pos_data in positions.items():
        if "position_value" in pos_data:
            position_values.append(pos_data["position_value"])
        if "total_risk" in pos_data:
            total_risk += pos_data["total_risk"]

    if not position_values:
        return {
            "total_exposure": 0.0,
            "total_risk": 0.0,
            "largest_position": 0.0,
            "concentration_ratio": 0.0,
            "number_of_positions": 0,
            "avg_position_size": 0.0,
        }

    total_exposure = sum(position_values)
    largest_position = max(position_values)

    # Herfindahl concentration index
    weights = [pv / total_exposure for pv in position_values]
    concentration_ratio = sum(w**2 for w in weights)

    return {
        "total_exposure": total_exposure / portfolio_value,
        "total_risk": total_risk / portfolio_value,
        "largest_position": largest_position / portfolio_value,
        "concentration_ratio": concentration_ratio,
        "number_of_positions": len(position_values),
        "avg_position_size": (total_exposure / len(position_values)) / portfolio_value,
    }


def calculate_var(
    returns: pd.Series, confidence_level: float = 0.95, method: str = "historical"
) -> float:
    """Calculate Value at Risk (VaR).

    Args:
        returns: Series of returns
        confidence_level: Confidence level (0.95 = 95%)
        method: Calculation method ('historical', 'parametric')

    Returns:
        VaR value (negative indicates loss)
    """
    if len(returns) == 0:
        return 0.0

    if method == "historical":
        # Historical simulation method
        alpha = 1 - confidence_level
        return float(np.percentile(returns, alpha * 100))

    elif method == "parametric":
        # Parametric method assuming normal distribution
        from scipy import stats

        alpha = 1 - confidence_level
        mean_return = returns.mean()
        std_return = returns.std()
        z_score = stats.norm.ppf(alpha)
        return float(mean_return + z_score * std_return)

    else:
        raise ValueError(f"Unknown VaR method: {method}")


def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.

    Args:
        returns: Series of returns
        confidence_level: Confidence level (0.95 = 95%)

    Returns:
        CVaR value (negative indicates loss)
    """
    if len(returns) == 0:
        return 0.0

    var = calculate_var(returns, confidence_level)
    tail_losses = returns[returns <= var]

    return float(tail_losses.mean()) if len(tail_losses) > 0 else var


def calculate_max_drawdown(equity_curve: pd.Series) -> Dict[str, float]:
    """Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: Series of portfolio values over time

    Returns:
        Dictionary with drawdown metrics
    """
    if len(equity_curve) == 0:
        return {
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "drawdown_duration": 0,
            "current_drawdown": 0.0,
        }

    # Calculate running maximum (peak)
    running_max = equity_curve.expanding().max()

    # Calculate drawdown
    drawdown = equity_curve - running_max
    drawdown_pct = drawdown / running_max

    max_drawdown = drawdown.min()
    max_drawdown_pct = drawdown_pct.min()

    # Calculate drawdown duration
    in_drawdown = drawdown < 0
    if in_drawdown.any():
        # Find longest consecutive period of drawdown
        groups = (in_drawdown != in_drawdown.shift()).cumsum()
        drawdown_periods = in_drawdown.groupby(groups).sum()
        max_duration = drawdown_periods.max() if len(drawdown_periods) > 0 else 0
    else:
        max_duration = 0

    # Current drawdown
    current_drawdown = drawdown.iloc[-1]

    return {
        "max_drawdown": float(max_drawdown),
        "max_drawdown_pct": float(max_drawdown_pct),
        "drawdown_duration": int(max_duration),
        "current_drawdown": float(current_drawdown),
    }


def calculate_correlation_matrix(
    returns_data: Dict[str, pd.Series], min_periods: int = 30
) -> pd.DataFrame:
    """Calculate correlation matrix from returns data.

    Args:
        returns_data: Dictionary of {symbol: returns_series}
        min_periods: Minimum periods required for correlation

    Returns:
        Correlation matrix DataFrame
    """
    if not returns_data:
        return pd.DataFrame()

    # Align all series to same index
    aligned_data = {}
    all_indices = set()

    for symbol, returns in returns_data.items():
        if len(returns) >= min_periods:
            aligned_data[symbol] = returns
            all_indices.update(returns.index)

    if not aligned_data:
        return pd.DataFrame()

    # Create aligned DataFrame
    common_index = sorted(all_indices)
    returns_df = pd.DataFrame(index=common_index)

    for symbol, returns in aligned_data.items():
        returns_df[symbol] = returns.reindex(common_index)

    # Calculate correlation matrix
    return returns_df.corr(min_periods=min_periods)


def detect_high_correlations(
    correlation_matrix: pd.DataFrame, threshold: float = 0.7
) -> List[Tuple[str, str, float]]:
    """Detect pairs of assets with high correlation.

    Args:
        correlation_matrix: Correlation matrix
        threshold: Correlation threshold to flag

    Returns:
        List of (symbol1, symbol2, correlation) tuples
    """
    high_corr_pairs = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            symbol1 = correlation_matrix.columns[i]
            symbol2 = correlation_matrix.columns[j]
            corr = correlation_matrix.iloc[i, j]

            if not pd.isna(corr) and abs(corr) > threshold:
                high_corr_pairs.append((symbol1, symbol2, float(corr)))

    return sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)


def calculate_risk_adjusted_returns(
    returns: pd.Series, risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """Calculate risk-adjusted return metrics.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary of risk-adjusted metrics
    """
    if len(returns) == 0:
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "total_return": 0.0,
            "volatility": 0.0,
        }

    # Annualized metrics
    periods_per_year = 252  # Assuming daily returns
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    volatility = returns.std() * np.sqrt(periods_per_year)

    # Sharpe ratio
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0

    # Sortino ratio (using downside deviation)
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(periods_per_year)
    sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0

    # Calmar ratio (return / max drawdown)
    equity_curve = (1 + returns).cumprod()
    dd_metrics = calculate_max_drawdown(equity_curve)
    max_dd_pct = abs(dd_metrics["max_drawdown_pct"])
    calmar_ratio = annualized_return / max_dd_pct if max_dd_pct > 0 else 0

    return {
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "calmar_ratio": float(calmar_ratio),
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "volatility": float(volatility),
        "downside_volatility": float(downside_volatility),
        "max_drawdown": max_dd_pct,
    }


def create_risk_limits_from_config(config: Any) -> Dict[str, float]:
    """Create risk limits dictionary from configuration.

    Args:
        config: Configuration object with risk parameters

    Returns:
        Dictionary of risk limits
    """
    try:
        financial_config = config.financial

        return {
            "max_position_size": float(financial_config.limits.max_position_size)
            / float(financial_config.capital.initial_capital),
            "max_portfolio_var": financial_config.risk.max_portfolio_risk,
            "max_daily_loss": float(financial_config.risk.max_daily_loss)
            / float(financial_config.capital.initial_capital),
            "max_drawdown": financial_config.risk.max_drawdown_percent,
            "stop_loss_pct": financial_config.risk.stop_loss_percent,
            "take_profit_pct": financial_config.risk.take_profit_percent,
            "max_leverage": financial_config.limits.max_leverage,
            "risk_free_rate": financial_config.risk.risk_free_rate,
        }
    except AttributeError as e:
        logger.warning(f"Could not extract risk limits from config: {e}")
        return {
            "max_position_size": 0.10,
            "max_portfolio_var": 0.02,
            "max_daily_loss": 0.03,
            "max_drawdown": 0.20,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.10,
            "max_leverage": 1.0,
            "risk_free_rate": 0.02,
        }


def validate_risk_parameters(risk_params: Dict[str, float]) -> List[str]:
    """Validate risk parameters for reasonableness.

    Args:
        risk_params: Dictionary of risk parameters

    Returns:
        List of validation error messages
    """
    errors = []

    # Position size limits
    if risk_params.get("max_position_size", 0) <= 0:
        errors.append("Max position size must be positive")
    elif risk_params.get("max_position_size", 0) > 0.5:
        errors.append("Max position size should not exceed 50%")

    # Portfolio limits
    if risk_params.get("max_portfolio_var", 0) <= 0:
        errors.append("Max portfolio VaR must be positive")
    elif risk_params.get("max_portfolio_var", 0) > 0.1:
        errors.append("Max portfolio VaR seems high (>10%)")

    # Stop loss
    if risk_params.get("stop_loss_pct", 0) <= 0:
        errors.append("Stop loss percentage must be positive")
    elif risk_params.get("stop_loss_pct", 0) > 0.3:
        errors.append("Stop loss percentage seems high (>30%)")

    # Take profit
    take_profit = risk_params.get("take_profit_pct", 0)
    stop_loss = risk_params.get("stop_loss_pct", 0)
    if take_profit > 0 and stop_loss > 0 and take_profit < stop_loss:
        errors.append("Take profit should be larger than stop loss")

    # Daily loss limit
    if risk_params.get("max_daily_loss", 0) <= 0:
        errors.append("Max daily loss must be positive")
    elif risk_params.get("max_daily_loss", 0) > 0.2:
        errors.append("Max daily loss seems high (>20%)")

    # Drawdown limit
    if risk_params.get("max_drawdown", 0) <= 0:
        errors.append("Max drawdown must be positive")
    elif risk_params.get("max_drawdown", 0) > 0.5:
        errors.append("Max drawdown seems high (>50%)")

    return errors


def format_risk_report(
    portfolio_metrics: Dict[str, float],
    position_risks: Dict[str, Dict[str, float]],
    risk_limits: Dict[str, float],
) -> str:
    """Format a comprehensive risk report.

    Args:
        portfolio_metrics: Portfolio-level risk metrics
        position_risks: Position-level risk data
        risk_limits: Risk limits configuration

    Returns:
        Formatted risk report string
    """
    report = []
    report.append("=" * 60)
    report.append("RISK MANAGEMENT REPORT")
    report.append("=" * 60)

    # Portfolio metrics
    report.append("\nPORTFOLIO METRICS:")
    report.append("-" * 30)
    report.append(f"Total Exposure: {portfolio_metrics.get('total_exposure', 0):.1%}")
    report.append(f"Total Risk: {portfolio_metrics.get('total_risk', 0):.1%}")
    report.append(f"Largest Position: {portfolio_metrics.get('largest_position', 0):.1%}")
    report.append(f"Number of Positions: {portfolio_metrics.get('number_of_positions', 0)}")
    report.append(f"Concentration Ratio: {portfolio_metrics.get('concentration_ratio', 0):.3f}")

    # Position details
    if position_risks:
        report.append("\nPOSITION RISKS:")
        report.append("-" * 30)
        for symbol, risk_data in position_risks.items():
            report.append(
                f"{symbol:6s}: Size={risk_data.get('position_size_pct', 0):.1%}, "
                f"Risk={risk_data.get('risk_pct', 0):.2%}, "
                f"P&L={risk_data.get('unrealized_pnl_pct', 0):+.1%}"
            )

    # Risk limits
    report.append("\nRISK LIMITS:")
    report.append("-" * 30)
    report.append(f"Max Position Size: {risk_limits.get('max_position_size', 0):.1%}")
    report.append(f"Max Daily Loss: {risk_limits.get('max_daily_loss', 0):.1%}")
    report.append(f"Stop Loss: {risk_limits.get('stop_loss_pct', 0):.1%}")
    report.append(f"Max Drawdown: {risk_limits.get('max_drawdown', 0):.1%}")

    # Limit checks
    report.append("\nLIMIT CHECKS:")
    report.append("-" * 30)

    # Check portfolio exposure
    exposure = portfolio_metrics.get("total_exposure", 0)
    if exposure > 0.95:
        report.append(f"⚠️  High portfolio exposure: {exposure:.1%}")

    # Check largest position
    largest = portfolio_metrics.get("largest_position", 0)
    max_pos = risk_limits.get("max_position_size", 0.1)
    if largest > max_pos:
        report.append(f"⚠️  Position size limit exceeded: {largest:.1%} > {max_pos:.1%}")

    # Check concentration
    concentration = portfolio_metrics.get("concentration_ratio", 0)
    if concentration > 0.3:
        report.append(f"⚠️  High concentration: {concentration:.3f}")

    if not any("⚠️" in line for line in report[-10:]):
        report.append("✅ All risk limits within acceptable ranges")

    report.append("=" * 60)

    return "\n".join(report)
