"""
Main market analysis orchestration - entry point for the slice.

Complete isolation - everything needed is local.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from bot_v2.data_providers import get_data_provider
from bot_v2.features.analyze.indicators import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_obv,
    calculate_rsi,
    calculate_sma,
    calculate_stochastic,
    calculate_volatility,
    detect_trend,
    identify_support_resistance,
)
from bot_v2.features.analyze.patterns import detect_patterns
from bot_v2.features.analyze.strategies import analyze_with_strategies
from bot_v2.features.analyze.types import (
    AnalysisResult,
    MarketRegime,
    PortfolioAnalysis,
    PricePattern,
    StrategyComparison,
    StrategySignals,
    SupportResistance,
    TechnicalIndicators,
)
from bot_v2.utilities.logging_patterns import get_logger

Recommendation = Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
TrendLiteral = Literal["bullish", "bearish", "neutral"]
VolatilityLiteral = Literal["low", "medium", "high"]
MomentumLiteral = Literal["strong_up", "weak_up", "neutral", "weak_down", "strong_down"]
VolumeProfileLiteral = Literal["accumulation", "distribution", "neutral"]

logger = get_logger(__name__, component="analysis")


def analyze_symbol(
    symbol: str,
    lookback_days: int = 90,
    include_patterns: bool = True,
    include_strategies: bool = True,
) -> AnalysisResult:
    """
    Perform complete analysis on a symbol.

    Args:
        symbol: Stock symbol to analyze
        lookback_days: Days of historical data to analyze
        include_patterns: Whether to detect patterns
        include_strategies: Whether to run strategy analysis

    Returns:
        Complete analysis result
    """
    # Fetch data
    data = fetch_data(symbol, lookback_days)

    if data.empty:
        raise ValueError(f"No data available for {symbol}")

    # Calculate indicators
    indicators = calculate_indicators(data)

    # Determine market regime
    regime = determine_regime(data, indicators)

    # Identify support/resistance
    levels = identify_levels(data)

    # Detect patterns
    patterns: list[PricePattern] = detect_patterns(data) if include_patterns else []

    # Get strategy signals
    strategy_signals: list[StrategySignals] = (
        analyze_with_strategies(data) if include_strategies else []
    )

    # Generate recommendation
    recommendation, confidence = generate_recommendation(
        indicators, regime, patterns, strategy_signals
    )

    return AnalysisResult(
        symbol=symbol,
        timestamp=datetime.now(),
        current_price=float(data["close"].iloc[-1]),
        indicators=indicators,
        regime=regime,
        patterns=patterns,
        levels=levels,
        strategy_signals=strategy_signals,
        recommendation=recommendation,
        confidence=confidence,
    )


def analyze_portfolio(
    symbols: list[str], weights: dict[str, float] | None = None, lookback_days: int = 90
) -> PortfolioAnalysis:
    """
    Analyze a portfolio of symbols.

    Args:
        symbols: List of symbols in portfolio
        weights: Symbol weights (equal weight if None)
        lookback_days: Days of historical data

    Returns:
        Portfolio analysis
    """
    if not symbols:
        raise ValueError("Portfolio analysis requires at least one symbol")

    if weights is None:
        weights = {s: 1.0 / len(symbols) for s in symbols}

    # Analyze each symbol
    symbol_analyses = {}
    for symbol in symbols:
        try:
            symbol_analyses[symbol] = analyze_symbol(symbol, lookback_days)
        except Exception as e:
            logger.warning("Could not analyze %s: %s", symbol, e)

    # Calculate correlation matrix
    correlation_matrix = calculate_correlations(symbols, lookback_days)

    # Calculate risk metrics
    risk_metrics = calculate_portfolio_risk(symbols, weights, correlation_matrix)

    # Generate rebalance suggestions
    rebalance_suggestions = generate_rebalance_suggestions(symbol_analyses, weights)

    # Calculate total value (simplified)
    total_value = 100000  # Placeholder

    # Sector allocation (simplified)
    sector_allocation = {"Technology": 0.4, "Finance": 0.3, "Healthcare": 0.3}

    return PortfolioAnalysis(
        timestamp=datetime.now(),
        total_value=total_value,
        symbol_analyses=symbol_analyses,
        correlation_matrix=correlation_matrix,
        sector_allocation=sector_allocation,
        risk_metrics=risk_metrics,
        rebalance_suggestions=rebalance_suggestions,
    )


def compare_strategies(
    symbol: str, strategies: list[str] | None = None, lookback_days: int = 90
) -> StrategyComparison:
    """
    Compare performance of different strategies.

    Args:
        symbol: Symbol to test strategies on
        strategies: List of strategies to compare (None for all)
        lookback_days: Period to test over

    Returns:
        Strategy comparison results
    """
    if strategies is None:
        strategies = ["SimpleMA", "Momentum", "MeanReversion", "Volatility", "Breakout"]

    # Fetch data
    data = fetch_data(symbol, lookback_days)

    # Run each strategy and collect metrics
    metrics: dict[str, dict[str, float]] = {}
    for strategy in strategies:
        strategy_metrics = backtest_strategy(strategy, data)
        metrics[strategy] = strategy_metrics

    # Rank strategies
    rankings = rank_strategies(metrics)

    # Find best strategy
    best_strategy = min(rankings.keys(), key=lambda strategy_: rankings[strategy_])

    # Generate recommendation
    recommendation = (
        f"Best strategy for {symbol}: {best_strategy} "
        f"with {metrics[best_strategy]['return']:.2%} return"
    )

    return StrategyComparison(
        strategies=strategies,
        period=f"{lookback_days} days",
        metrics=metrics,
        rankings=rankings,
        best_strategy=best_strategy,
        recommendation=recommendation,
    )


# Helper functions


def fetch_data(symbol: str, lookback_days: int) -> pd.DataFrame:
    """Fetch historical data for analysis."""
    provider = get_data_provider()
    data = cast(pd.DataFrame, provider.get_historical_data(symbol, period=f"{lookback_days}d"))

    # Standardize columns
    data.columns = data.columns.str.lower()

    return data


def calculate_indicators(data: pd.DataFrame) -> TechnicalIndicators:
    """Calculate all technical indicators."""
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    volume_series = data.get("volume")
    if volume_series is None:
        volume = pd.Series(0.0, index=data.index)
    else:
        volume = volume_series.astype(float)

    # Moving averages
    sma_20 = calculate_sma(close, 20).iloc[-1]
    sma_50 = calculate_sma(close, 50).iloc[-1] if len(close) >= 50 else sma_20
    sma_200 = calculate_sma(close, 200).iloc[-1] if len(close) >= 200 else sma_50
    ema_12 = calculate_ema(close, 12).iloc[-1]
    ema_26 = calculate_ema(close, 26).iloc[-1]

    # RSI
    rsi = calculate_rsi(close).iloc[-1]

    # MACD
    macd, signal, histogram = calculate_macd(close)

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)

    # ATR
    atr = calculate_atr(high, low, close).iloc[-1]

    # Volume indicators
    volume_sma_series = calculate_sma(volume, 20)
    volume_sma = volume_sma_series.iloc[-1] if not volume_sma_series.empty else 0.0
    obv_series = calculate_obv(close, volume)
    obv = obv_series.iloc[-1] if not obv_series.empty else 0.0

    # Stochastic
    stoch_k, stoch_d = calculate_stochastic(high, low, close)

    return TechnicalIndicators(
        sma_20=float(sma_20),
        sma_50=float(sma_50),
        sma_200=float(sma_200),
        ema_12=float(ema_12),
        ema_26=float(ema_26),
        rsi=float(rsi) if not pd.isna(rsi) else 50.0,
        macd=float(macd.iloc[-1]) if not macd.empty else 0.0,
        macd_signal=float(signal.iloc[-1]) if not signal.empty else 0.0,
        macd_histogram=float(histogram.iloc[-1]) if not histogram.empty else 0.0,
        bollinger_upper=float(bb_upper.iloc[-1]) if not bb_upper.empty else close.iloc[-1],
        bollinger_middle=float(bb_middle.iloc[-1]) if not bb_middle.empty else close.iloc[-1],
        bollinger_lower=float(bb_lower.iloc[-1]) if not bb_lower.empty else close.iloc[-1],
        atr=float(atr) if not pd.isna(atr) else 0.0,
        volume_sma=float(volume_sma),
        obv=float(obv),
        stochastic_k=float(stoch_k.iloc[-1]) if not stoch_k.empty else 50.0,
        stochastic_d=float(stoch_d.iloc[-1]) if not stoch_d.empty else 50.0,
    )


def determine_regime(data: pd.DataFrame, indicators: TechnicalIndicators) -> MarketRegime:
    """Determine current market regime."""
    close = data["close"]
    returns = close.pct_change()

    # Trend
    raw_trend = detect_trend(close)
    if raw_trend in {"bullish", "bearish", "neutral"}:
        trend = cast(TrendLiteral, raw_trend)
    else:
        trend = "neutral"

    # Volatility
    raw_volatility = calculate_volatility(returns)
    if raw_volatility in {"low", "medium", "high"}:
        volatility = cast(VolatilityLiteral, raw_volatility)
    else:
        volatility = "medium"

    # Momentum
    if indicators.rsi > 70:
        momentum: MomentumLiteral = "strong_up"
    elif indicators.rsi > 55:
        momentum = "weak_up"
    elif indicators.rsi < 30:
        momentum = "strong_down"
    elif indicators.rsi < 45:
        momentum = "weak_down"
    else:
        momentum = "neutral"

    # Volume profile
    current_volume = data["volume"].iloc[-1]
    avg_volume = indicators.volume_sma
    if current_volume > avg_volume * 1.5:
        if close.iloc[-1] > close.iloc[-2]:
            volume_profile_value = "accumulation"
        else:
            volume_profile_value = "distribution"
    else:
        volume_profile_value = "neutral"
    volume_profile = cast(VolumeProfileLiteral, volume_profile_value)

    # Regime strength (0-100)
    strength = 50.0
    if trend == "bullish" and momentum in ["strong_up", "weak_up"]:
        strength = 75.0
    elif trend == "bearish" and momentum in ["strong_down", "weak_down"]:
        strength = 75.0
    elif trend == "neutral":
        strength = 25.0

    return MarketRegime(
        trend=trend,
        volatility=volatility,
        momentum=momentum,
        volume_profile=volume_profile,
        strength=strength,
    )


def identify_levels(data: pd.DataFrame) -> SupportResistance:
    """Identify support and resistance levels."""
    imm_sup, str_sup, imm_res, str_res, pivot = identify_support_resistance(data)

    return SupportResistance(
        immediate_support=float(imm_sup),
        strong_support=float(str_sup),
        immediate_resistance=float(imm_res),
        strong_resistance=float(str_res),
        pivot_point=float(pivot),
    )


def generate_recommendation(
    indicators: TechnicalIndicators,
    regime: MarketRegime,
    patterns: list[PricePattern],
    signals: list[StrategySignals],
) -> tuple[Recommendation, float]:
    """
    Generate trading recommendation.

    Returns:
        (recommendation, confidence)
    """
    score = 0
    factors = 0

    # RSI signal
    if indicators.rsi < 30:
        score += 2  # Oversold
        factors += 1
    elif indicators.rsi > 70:
        score -= 2  # Overbought
        factors += 1

    # MACD signal
    if indicators.macd > indicators.macd_signal:
        score += 1
        factors += 1
    else:
        score -= 1
        factors += 1

    # Trend alignment
    if regime.trend == "bullish":
        score += 2
        factors += 1
    elif regime.trend == "bearish":
        score -= 2
        factors += 1

    # Strategy consensus
    if signals:
        signal_sum = sum(s.signal for s in signals)
        if signal_sum > 2:
            score += 2
        elif signal_sum < -2:
            score -= 2
        factors += 1

    # Pattern signals
    bullish_patterns = sum(1 for p in patterns if "bull" in p.pattern_type.lower())
    bearish_patterns = sum(1 for p in patterns if "bear" in p.pattern_type.lower())
    score += bullish_patterns - bearish_patterns
    if patterns:
        factors += 1

    # Calculate average score
    avg_score = score / factors if factors > 0 else 0

    # Generate recommendation
    if avg_score >= 1.5:
        recommendation: Recommendation = "strong_buy"
        confidence = min(0.9, 0.5 + avg_score * 0.1)
    elif avg_score >= 0.5:
        recommendation = "buy"
        confidence = 0.6 + avg_score * 0.1
    elif avg_score <= -1.5:
        recommendation = "strong_sell"
        confidence = min(0.9, 0.5 + abs(avg_score) * 0.1)
    elif avg_score <= -0.5:
        recommendation = "sell"
        confidence = 0.6 + abs(avg_score) * 0.1
    else:
        recommendation = "hold"
        confidence = 0.5

    return recommendation, confidence


def calculate_correlations(symbols: list[str], lookback_days: int) -> pd.DataFrame:
    """Calculate correlation matrix for symbols."""
    prices = pd.DataFrame()
    for symbol in symbols:
        try:
            provider = get_data_provider()
            data = provider.get_historical_data(symbol, period=f"{lookback_days}d")
            data.columns = data.columns.str.lower()
            if not data.empty and "close" in data.columns:
                prices[symbol] = data["close"].astype(float)
        except Exception as exc:
            logger.warning("Failed to load historical data for %s: %s", symbol, exc, exc_info=True)

    if prices.empty:
        return pd.DataFrame()

    returns = prices.pct_change().dropna()
    return returns.corr()


def calculate_portfolio_risk(
    symbols: list[str], weights: dict[str, float], correlation_matrix: pd.DataFrame
) -> dict[str, float]:
    """Calculate portfolio risk metrics."""
    # Simplified risk calculations
    if correlation_matrix.empty:
        avg_correlation = 0.0
    else:
        upper_idx = np.triu_indices_from(correlation_matrix.values, k=1)
        upper_values = correlation_matrix.values[upper_idx]
        avg_correlation = float(np.mean(upper_values)) if upper_values.size else 0.0

    # Concentration risk
    max_weight = max(weights.values(), default=0.0)
    concentration_risk = float(max_weight)

    return {
        "beta": 1.0,  # Placeholder
        "correlation_risk": abs(avg_correlation),
        "concentration_risk": concentration_risk,
        "var_95": 0.02,  # Placeholder 2% VaR
        "cvar_95": 0.03,  # Placeholder 3% CVaR
    }


def generate_rebalance_suggestions(
    analyses: dict[str, AnalysisResult], current_weights: dict[str, float]
) -> list[dict]:
    """Generate portfolio rebalancing suggestions."""
    suggestions: list[dict[str, Any]] = []

    for symbol, analysis in analyses.items():
        current_weight = current_weights.get(symbol, 0)

        # Suggest changes based on recommendation
        if analysis.recommendation == "strong_buy" and current_weight < 0.2:
            suggestions.append(
                {
                    "symbol": symbol,
                    "action": "increase",
                    "current_weight": float(current_weight),
                    "target_weight": float(min(0.2, current_weight * 1.5)),
                    "reason": f"Strong buy signal with {analysis.confidence:.1%} confidence",
                }
            )
        elif analysis.recommendation == "strong_sell" and current_weight > 0.05:
            suggestions.append(
                {
                    "symbol": symbol,
                    "action": "decrease",
                    "current_weight": float(current_weight),
                    "target_weight": float(max(0.05, current_weight * 0.5)),
                    "reason": f"Strong sell signal with {analysis.confidence:.1%} confidence",
                }
            )

    return suggestions


def backtest_strategy(strategy: str, data: pd.DataFrame) -> dict[str, float]:
    """Simple backtest of a strategy."""
    # This is simplified - would use actual backtest in production
    return {
        "return": np.random.uniform(-0.1, 0.3),
        "sharpe": np.random.uniform(0.5, 2.0),
        "max_drawdown": np.random.uniform(0.05, 0.2),
        "win_rate": np.random.uniform(0.4, 0.7),
    }


def rank_strategies(metrics: Mapping[str, Mapping[str, float]]) -> dict[str, int]:
    """Rank strategies based on metrics."""
    scores: dict[str, float] = {}
    for strategy, m in metrics.items():
        # Simple scoring: return + sharpe - drawdown
        score = m.get("return", 0.0) + m.get("sharpe", 0.0) - m.get("max_drawdown", 0.0)
        scores[strategy] = float(score)

    # Convert to rankings
    sorted_strategies = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {name: idx + 1 for idx, (name, _) in enumerate(sorted_strategies)}
