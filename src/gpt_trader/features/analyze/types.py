from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class TechnicalIndicators:
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
    trend: str
    volatility: str
    momentum: str
    volume_profile: str
    strength: float

@dataclass
class PricePattern:
    name: str
    confidence: float
    target: float
    stop_loss: float
    sentiment: str

@dataclass
class StrategySignals:
    name: str
    signal: int
    confidence: float
    comment: str

@dataclass
class SupportResistance:
    support_1: float
    support_2: float
    resistance_1: float
    resistance_2: float
    pivot: float

@dataclass
class AnalysisResult:
    symbol: str
    timestamp: datetime
    current_price: float
    indicators: TechnicalIndicators
    regime: MarketRegime
    patterns: List[PricePattern]
    levels: SupportResistance
    strategy_signals: List[StrategySignals]
    recommendation: str
    confidence: float

@dataclass
class StrategyComparison:
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trades_count: int
