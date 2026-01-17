"""Risk metrics calculation for backtesting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpt_trader.backtesting.simulation.broker import SimulatedBroker


@dataclass
class RiskMetrics:
    """Risk-adjusted performance metrics from a backtest."""

    # Drawdown metrics
    max_drawdown_pct: Decimal
    max_drawdown_usd: Decimal
    avg_drawdown_pct: Decimal
    drawdown_duration_days: int  # Longest time in drawdown

    # Return metrics
    total_return_pct: Decimal
    annualized_return_pct: Decimal
    daily_return_avg: Decimal
    daily_return_std: Decimal

    # Risk-adjusted returns
    sharpe_ratio: Decimal | None  # (return - risk_free) / std
    sortino_ratio: Decimal | None  # (return - risk_free) / downside_std
    calmar_ratio: Decimal | None  # annual_return / max_drawdown

    # Volatility metrics
    volatility_annualized: Decimal
    downside_volatility: Decimal

    # Risk exposure
    max_leverage_used: Decimal
    avg_leverage_used: Decimal
    time_in_market_pct: Decimal  # % of time with open positions

    # Value at Risk
    var_95_daily: Decimal  # 95% VaR
    var_99_daily: Decimal  # 99% VaR


def calculate_risk_metrics(
    broker: SimulatedBroker,
    risk_free_rate: Decimal = Decimal("0.05"),  # 5% annual risk-free rate
    trading_days_per_year: int = 365,  # Crypto trades 365 days
) -> RiskMetrics:
    """
    Calculate comprehensive risk metrics from a SimulatedBroker.

    Args:
        broker: SimulatedBroker with equity curve data
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        trading_days_per_year: Number of trading days per year

    Returns:
        RiskMetrics with all computed values
    """
    equity_curve = broker.get_equity_curve()

    if len(equity_curve) < 2:
        return _empty_risk_metrics()

    # Extract equity values and timestamps
    timestamps = [t for t, _ in equity_curve]
    equities = [float(e) for _, e in equity_curve]

    # Duration calculations
    start_time = timestamps[0]
    end_time = timestamps[-1]
    duration_days = max(1, (end_time - start_time).days)

    # Aggregate to daily closes and calculate daily returns
    daily_equities = _daily_equity_series(equity_curve)
    daily_returns = _calculate_returns(daily_equities)

    period_returns = daily_returns
    periods_per_year = float(trading_days_per_year)
    if not period_returns:
        period_returns = _calculate_returns(equities)
        if not period_returns:
            return _empty_risk_metrics()
        periods_per_year = (
            float(len(period_returns))
            / max(1.0, float(duration_days))
            * float(trading_days_per_year)
        )

    # Return calculations
    initial_equity = float(broker._initial_equity)
    final_equity = float(broker.get_equity())
    total_return = (final_equity - initial_equity) / initial_equity

    # Annualized return
    years = duration_days / 365
    if years > 0:
        annualized_return = (1 + total_return) ** (1 / years) - 1
    else:
        annualized_return = total_return

    # Period statistics
    period_avg = sum(period_returns) / len(period_returns)
    period_std = _std_dev(period_returns)

    # Downside deviation (only negative returns)
    negative_returns = [r for r in period_returns if r < 0]
    downside_std = _std_dev(negative_returns) if negative_returns else 0.0

    # Annualized volatility
    volatility_annual = period_std * math.sqrt(periods_per_year)
    downside_vol = downside_std * math.sqrt(periods_per_year)

    # Sharpe ratio (period risk-free, annualized)
    period_risk_free = float(risk_free_rate) / periods_per_year
    excess_return = period_avg - period_risk_free
    sharpe = (
        Decimal(str(excess_return / period_std * math.sqrt(periods_per_year)))
        if period_std > 0
        else None
    )

    # Sortino ratio (period risk-free, annualized)
    sortino = (
        Decimal(str(excess_return / downside_std * math.sqrt(periods_per_year)))
        if downside_std > 0
        else None
    )

    # Calmar ratio
    max_dd = float(broker._max_drawdown)
    calmar = Decimal(str(annualized_return * 100 / max_dd)) if max_dd > 0 else None

    # Drawdown metrics
    drawdown_data = _calculate_drawdown_metrics(equities, timestamps)

    # VaR calculations (parametric, assuming normal distribution)
    var_95 = _calculate_var(period_returns, 0.95)
    var_99 = _calculate_var(period_returns, 0.99)

    # Leverage and time in market
    max_leverage = Decimal("1")  # Default
    avg_leverage = Decimal("1")
    time_in_market = _calculate_time_in_market(broker)

    avg_dd = drawdown_data["avg_drawdown"]
    max_dur = drawdown_data["max_duration"]
    return RiskMetrics(
        max_drawdown_pct=broker._max_drawdown,
        max_drawdown_usd=broker._max_drawdown_usd,
        avg_drawdown_pct=Decimal(str(avg_dd)),
        drawdown_duration_days=int(max_dur),
        total_return_pct=Decimal(str(total_return * 100)),
        annualized_return_pct=Decimal(str(annualized_return * 100)),
        daily_return_avg=Decimal(str(period_avg * 100)),
        daily_return_std=Decimal(str(period_std * 100)),
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        volatility_annualized=Decimal(str(volatility_annual * 100)),
        downside_volatility=Decimal(str(downside_vol * 100)),
        max_leverage_used=max_leverage,
        avg_leverage_used=avg_leverage,
        time_in_market_pct=time_in_market,
        var_95_daily=Decimal(str(var_95 * 100)),
        var_99_daily=Decimal(str(var_99 * 100)),
    )


def _empty_risk_metrics() -> RiskMetrics:
    """Return empty risk metrics when insufficient data."""
    return RiskMetrics(
        max_drawdown_pct=Decimal("0"),
        max_drawdown_usd=Decimal("0"),
        avg_drawdown_pct=Decimal("0"),
        drawdown_duration_days=0,
        total_return_pct=Decimal("0"),
        annualized_return_pct=Decimal("0"),
        daily_return_avg=Decimal("0"),
        daily_return_std=Decimal("0"),
        sharpe_ratio=None,
        sortino_ratio=None,
        calmar_ratio=None,
        volatility_annualized=Decimal("0"),
        downside_volatility=Decimal("0"),
        max_leverage_used=Decimal("1"),
        avg_leverage_used=Decimal("1"),
        time_in_market_pct=Decimal("0"),
        var_95_daily=Decimal("0"),
        var_99_daily=Decimal("0"),
    )


def _calculate_returns(equities: list[float]) -> list[float]:
    """Calculate period-over-period returns."""
    if len(equities) < 2:
        return []

    returns = []
    for i in range(1, len(equities)):
        if equities[i - 1] != 0:
            ret = (equities[i] - equities[i - 1]) / equities[i - 1]
            returns.append(ret)

    return returns


def _daily_equity_series(equity_curve: list[tuple[datetime, Decimal]]) -> list[float]:
    """Aggregate intraday equity curve to daily closing values."""
    if not equity_curve:
        return []

    daily_closes: dict[date, float] = {}  # date -> equity close
    for timestamp, equity in equity_curve:
        daily_closes[timestamp.date()] = float(equity)

    return [daily_closes[day] for day in sorted(daily_closes.keys())]


def _std_dev(values: list[float]) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0

    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _calculate_drawdown_metrics(
    equities: list[float], timestamps: list[datetime]
) -> dict[str, Decimal | int]:
    """Calculate drawdown statistics."""
    if not equities:
        return {"avg_drawdown": Decimal("0"), "max_duration": 0}

    peak = equities[0]
    drawdowns: list[float] = []
    drawdown_start: datetime | None = None
    max_duration = 0

    for i, equity in enumerate(equities):
        if equity > peak:
            peak = equity
            if drawdown_start is not None:
                # End of drawdown period
                duration = (timestamps[i] - drawdown_start).days
                max_duration = max(max_duration, duration)
                drawdown_start = None
        else:
            drawdown_pct = (peak - equity) / peak if peak > 0 else 0
            drawdowns.append(drawdown_pct)
            if drawdown_start is None:
                drawdown_start = timestamps[i]

    # Check if still in drawdown
    if drawdown_start is not None:
        duration = (timestamps[-1] - drawdown_start).days
        max_duration = max(max_duration, duration)

    avg_drawdown = (
        Decimal(str(sum(drawdowns) / len(drawdowns) * 100)) if drawdowns else Decimal("0")
    )

    return {"avg_drawdown": avg_drawdown, "max_duration": max_duration}


def _calculate_var(returns: list[float], confidence: float) -> float:
    """
    Calculate Value at Risk using historical simulation.

    Args:
        returns: List of daily returns
        confidence: Confidence level (e.g., 0.95 for 95% VaR)

    Returns:
        VaR as a positive number (loss)
    """
    if not returns:
        return 0.0

    sorted_returns = sorted(returns)
    index = int((1 - confidence) * len(sorted_returns))
    index = max(0, min(index, len(sorted_returns) - 1))

    return -sorted_returns[index]  # Return as positive loss


def _calculate_time_in_market(broker: SimulatedBroker) -> Decimal:
    """Calculate percentage of time with open positions."""
    equity_curve = broker.get_equity_curve()
    if len(equity_curve) < 2:
        return Decimal("0")

    # Count bars where we had positions
    # This is an approximation based on order history
    total_bars = len(equity_curve)
    bars_with_positions = 0

    # If we have positions at any point, count that bar
    # For accurate tracking, we'd need position history per bar
    if broker.positions:
        bars_with_positions = total_bars // 2  # Rough estimate

    return (
        Decimal(bars_with_positions) / Decimal(total_bars) * 100 if total_bars > 0 else Decimal("0")
    )
