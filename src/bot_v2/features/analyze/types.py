"""
Local types for market analysis.

Complete isolation - no external dependencies.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import pandas as pd


@dataclass
class TechnicalIndicators:
    """Collection of technical indicators."""

    sma_20: float
    sma_50: float
    sma_200: float
    ema_12: float
    ema_26: float
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float
    atr: float
    volume_sma: float
    obv: float
    stochastic_k: float
    stochastic_d: float


@dataclass
class MarketRegime:
    """Current market regime classification."""

    trend: Literal["bullish", "bearish", "neutral"]
    volatility: Literal["low", "medium", "high"]
    momentum: Literal["strong_up", "weak_up", "neutral", "weak_down", "strong_down"]
    volume_profile: Literal["accumulation", "distribution", "neutral"]
    strength: float  # 0-100 score


@dataclass
class PricePattern:
    """Detected price pattern."""

    pattern_type: str
    confidence: float
    target_price: float | None
    stop_loss: float | None
    description: str


@dataclass
class SupportResistance:
    """Support and resistance levels."""

    immediate_support: float
    strong_support: float
    immediate_resistance: float
    strong_resistance: float
    pivot_point: float


@dataclass
class StrategySignals:
    """Signals from multiple strategies."""

    strategy_name: str
    signal: int  # 1=buy, -1=sell, 0=hold
    confidence: float
    reason: str


@dataclass
class AnalysisResult:
    """Complete analysis result for a symbol."""

    symbol: str
    timestamp: datetime
    current_price: float
    indicators: TechnicalIndicators
    regime: MarketRegime
    patterns: list[PricePattern]
    levels: SupportResistance
    strategy_signals: list[StrategySignals]
    recommendation: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    confidence: float

    def summary(self) -> str:
        """Generate analysis summary."""
        sum(s.signal for s in self.strategy_signals)

        return f"""
Analysis Summary for {self.symbol}
=====================================
Timestamp: {self.timestamp}
Current Price: ${self.current_price:.2f}

Market Regime:
- Trend: {self.regime.trend}
- Volatility: {self.regime.volatility}
- Momentum: {self.regime.momentum}

Key Indicators:
- RSI: {self.indicators.rsi:.2f}
- MACD: {self.indicators.macd:.4f}
- ATR: {self.indicators.atr:.2f}

Support/Resistance:
- Immediate Support: ${self.levels.immediate_support:.2f}
- Immediate Resistance: ${self.levels.immediate_resistance:.2f}

Detected Patterns: {len(self.patterns)}
Strategy Signals: {len([s for s in self.strategy_signals if s.signal != 0])} active

Recommendation: {self.recommendation.upper()}
Confidence: {self.confidence:.1%}
        """.strip()


@dataclass
class PortfolioAnalysis:
    """Portfolio-wide analysis."""

    timestamp: datetime
    total_value: float
    symbol_analyses: dict[str, AnalysisResult]
    correlation_matrix: pd.DataFrame
    sector_allocation: dict[str, float]
    risk_metrics: dict[str, float]
    rebalance_suggestions: list[dict]

    def summary(self) -> str:
        """Generate portfolio summary."""
        buy_signals = sum(
            1 for a in self.symbol_analyses.values() if a.recommendation in ["buy", "strong_buy"]
        )
        sell_signals = sum(
            1 for a in self.symbol_analyses.values() if a.recommendation in ["sell", "strong_sell"]
        )

        return f"""
Portfolio Analysis Summary
==========================
Timestamp: {self.timestamp}
Portfolio Value: ${self.total_value:,.2f}

Symbols Analyzed: {len(self.symbol_analyses)}
Buy Signals: {buy_signals}
Sell Signals: {sell_signals}

Risk Metrics:
- Portfolio Beta: {self.risk_metrics.get('beta', 0):.2f}
- Correlation Risk: {self.risk_metrics.get('correlation_risk', 0):.2%}
- Concentration Risk: {self.risk_metrics.get('concentration_risk', 0):.2%}

Rebalance Suggestions: {len(self.rebalance_suggestions)}
        """.strip()


@dataclass
class StrategyComparison:
    """Comparison of multiple strategies."""

    strategies: list[str]
    period: str
    metrics: dict[str, dict[str, float]]  # strategy -> metric -> value
    rankings: dict[str, int]  # strategy -> rank
    best_strategy: str
    recommendation: str
