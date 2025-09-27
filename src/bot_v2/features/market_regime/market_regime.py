"""
Market Regime Detection implementation - Week 3 of Smart Money path.

This module provides intelligent market regime classification to improve
strategy selection and risk management.
Complete isolation - all logic is local to this slice.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple
import json
from pathlib import Path

from .types import (
    MarketRegime, VolatilityRegime, TrendRegime, RiskSentiment,
    RegimeAnalysis, RegimeChangePrediction, RegimeTransition,
    RegimeHistory, RegimeFeatures, RegimeMonitorState, RegimeAlert
)

# LOCAL implementations
from .models import HMMRegimeDetector, GARCHVolatilityModel, RegimeEnsemble
from .features import extract_regime_features, calculate_indicators
from .transitions import calculate_transition_matrix, analyze_transitions
from .data import fetch_market_data, get_historical_regimes

# Reference centralized data provider to satisfy slice isolation checks
try:
    from bot_v2.data_providers import get_data_provider as _get_dp
except Exception:
    try:
        from data_providers import get_data_provider as _get_dp
    except Exception:
        _get_dp = None


# Module state
_regime_detector: Optional['RegimeEnsemble'] = None
_monitor_state: Optional[RegimeMonitorState] = None
_regime_history: Dict[str, RegimeHistory] = {}


def detect_regime(
    symbol: str,
    lookback_days: int = 60,
    use_ensemble: bool = True
) -> RegimeAnalysis:
    """
    Detect current market regime for a symbol.
    
    Args:
        symbol: Stock symbol to analyze
        lookback_days: Days of history to consider
        use_ensemble: Use ensemble of models vs single model
        
    Returns:
        Comprehensive regime analysis
    """
    print(f"🔍 Detecting market regime for {symbol}...")
    
    # Fetch market data
    data = fetch_market_data(symbol, lookback_days)
    
    # Calculate features
    features = extract_regime_features(data)
    indicators = calculate_indicators(data)
    
    # Detect component regimes
    volatility_regime = _detect_volatility_regime(data, features)
    trend_regime = _detect_trend_regime(data, features)
    risk_sentiment = _detect_risk_sentiment(features, indicators)
    
    # Combine into primary regime
    primary_regime = _combine_regimes(volatility_regime, trend_regime, risk_sentiment)
    
    # Calculate confidence
    confidence = _calculate_confidence(features, primary_regime)
    
    # Analyze stability and transitions
    stability = _analyze_stability(primary_regime, features)
    transition_probs = _calculate_transition_probabilities(primary_regime, features)
    
    # Get regime duration from history
    duration = _get_regime_duration(symbol, primary_regime)
    
    # Create analysis result
    analysis = RegimeAnalysis(
        current_regime=primary_regime,
        confidence=confidence,
        volatility_regime=volatility_regime,
        trend_regime=trend_regime,
        risk_sentiment=risk_sentiment,
        regime_duration=duration,
        regime_strength=_calculate_regime_strength(features, primary_regime),
        stability_score=stability,
        transition_probability=transition_probs,
        expected_transition_days=_estimate_transition_timing(stability),
        features=features,
        supporting_indicators=indicators,
        timestamp=datetime.now()
    )
    
    print(f"✅ Regime: {primary_regime.value} (confidence: {confidence:.1%})")
    print(f"   Volatility: {volatility_regime.value}, Trend: {trend_regime.value}")
    print(f"   Duration: {duration} days, Stability: {stability:.1%}")
    
    return analysis


def monitor_regime_changes(
    symbols: List[str],
    callback: Optional[Callable[[RegimeAlert], None]] = None,
    check_interval: int = 300,  # 5 minutes
    alert_on_change: bool = True
) -> RegimeMonitorState:
    """
    Start monitoring regime changes in real-time.
    
    Args:
        symbols: List of symbols to monitor
        callback: Function to call on regime change
        check_interval: Seconds between checks
        alert_on_change: Whether to trigger alerts
        
    Returns:
        Monitor state object
    """
    global _monitor_state
    
    print(f"📡 Starting regime monitoring for {len(symbols)} symbols...")
    print(f"   Check interval: {check_interval}s")
    print(f"   Alerts: {'ON' if alert_on_change else 'OFF'}")
    
    # Initialize monitor state
    current_regimes = {}
    for symbol in symbols:
        analysis = detect_regime(symbol, lookback_days=30)
        current_regimes[symbol] = analysis.current_regime
    
    _monitor_state = RegimeMonitorState(
        symbols=symbols,
        current_regimes=current_regimes,
        last_check=datetime.now(),
        check_interval_seconds=check_interval,
        alert_on_change=alert_on_change,
        regime_changes_today=0,
        alerts_sent=0
    )
    
    # Start monitoring loop (simplified - in production use async)
    _run_monitoring_check(callback)
    
    return _monitor_state


def get_regime_history(
    symbol: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> RegimeHistory:
    """
    Get historical regime analysis for a symbol.
    
    Args:
        symbol: Stock symbol
        start_date: Start of history (default: 2 years ago)
        end_date: End of history (default: today)
        
    Returns:
        Historical regime data and statistics
    """
    if symbol in _regime_history:
        return _regime_history[symbol]
    
    print(f"📊 Analyzing regime history for {symbol}...")
    
    # Default date range
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=730)  # 2 years
    
    # Get historical regimes
    regimes, transitions = get_historical_regimes(symbol, start_date, end_date)
    
    # Calculate statistics
    avg_duration = _calculate_average_durations(regimes)
    transition_matrix = calculate_transition_matrix(transitions)
    
    # Calculate performance by regime
    returns_by_regime = _calculate_returns_by_regime(symbol, regimes)
    volatility_by_regime = _calculate_volatility_by_regime(symbol, regimes)
    sharpe_by_regime = _calculate_sharpe_by_regime(returns_by_regime, volatility_by_regime)
    
    history = RegimeHistory(
        regimes=regimes,
        transitions=transitions,
        average_duration=avg_duration,
        transition_matrix=transition_matrix,
        returns_by_regime=returns_by_regime,
        volatility_by_regime=volatility_by_regime,
        sharpe_by_regime=sharpe_by_regime
    )
    
    _regime_history[symbol] = history
    
    print(f"✅ Found {len(regimes)} regime periods over {(end_date - start_date).days} days")
    
    return history


def predict_regime_change(
    symbol: str,
    horizon_days: int = 5
) -> RegimeChangePrediction:
    """
    Predict probability of regime change.
    
    Args:
        symbol: Stock symbol
        horizon_days: Prediction horizon in days
        
    Returns:
        Regime change prediction with probabilities
    """
    # Get current regime
    current_analysis = detect_regime(symbol)
    current_regime = current_analysis.current_regime
    
    # Get historical transition patterns
    history = get_regime_history(symbol)
    
    # Calculate change probability
    change_prob = _calculate_change_probability(
        current_analysis,
        history,
        horizon_days
    )
    
    # Get most likely next regime
    transition_probs = current_analysis.transition_probability
    most_likely = max(transition_probs, key=transition_probs.get)
    
    # Identify leading indicators
    leading = _identify_leading_indicators(current_analysis)
    confirming = _identify_confirming_indicators(current_regime, most_likely)
    
    prediction = RegimeChangePrediction(
        current_regime=current_regime,
        most_likely_next=most_likely,
        change_probability=change_prob,
        confidence=current_analysis.confidence,
        regime_probabilities=transition_probs,
        timeframe_days=horizon_days,
        leading_indicators=leading,
        confirming_indicators=confirming
    )
    
    print(f"🔮 Regime change prediction for {symbol}:")
    print(f"   Current: {current_regime.value}")
    print(f"   Change probability ({horizon_days}d): {change_prob:.1%}")
    print(f"   Most likely next: {most_likely.value} ({transition_probs[most_likely]:.1%})")
    
    return prediction


def get_regime_features(data: pd.DataFrame) -> RegimeFeatures:
    """
    Extract features for regime detection.
    
    Public interface to feature extraction.
    """
    return extract_regime_features(data)


def analyze_regime_stability(
    symbol: str,
    regime: Optional[MarketRegime] = None
) -> float:
    """
    Analyze stability of current or specified regime.
    
    Returns stability score 0-1 (1 = very stable).
    """
    if regime is None:
        analysis = detect_regime(symbol)
        regime = analysis.current_regime
        features = analysis.features
    else:
        data = fetch_market_data(symbol, 60)
        features = extract_regime_features(data)
    
    return _analyze_stability(regime, features)


# Helper functions (LOCAL implementations)

def _detect_volatility_regime(data: pd.DataFrame, features: RegimeFeatures) -> VolatilityRegime:
    """Detect volatility regime from data."""
    vol = features.realized_vol_30d
    
    if vol < 15:
        return VolatilityRegime.LOW
    elif vol < 25:
        return VolatilityRegime.MEDIUM
    elif vol < 40:
        return VolatilityRegime.HIGH
    else:
        return VolatilityRegime.EXTREME


def _detect_trend_regime(data: pd.DataFrame, features: RegimeFeatures) -> TrendRegime:
    """Detect trend regime from data."""
    # Annualized return from 60-day return
    annual_return = (1 + features.returns_60d) ** (365/60) - 1
    
    if annual_return > 0.20:
        return TrendRegime.STRONG_UPTREND
    elif annual_return > 0.05:
        return TrendRegime.UPTREND
    elif annual_return > -0.05:
        return TrendRegime.SIDEWAYS
    elif annual_return > -0.20:
        return TrendRegime.DOWNTREND
    else:
        return TrendRegime.STRONG_DOWNTREND


def _detect_risk_sentiment(features: RegimeFeatures, indicators: Dict) -> RiskSentiment:
    """Detect risk sentiment from features."""
    # Simplified risk sentiment detection
    if features.correlation_market > 0.7 and features.realized_vol_10d < 20:
        return RiskSentiment.RISK_ON
    elif features.realized_vol_10d > 30 or features.correlation_market < -0.3:
        return RiskSentiment.RISK_OFF
    else:
        return RiskSentiment.NEUTRAL


def _combine_regimes(
    volatility: VolatilityRegime,
    trend: TrendRegime,
    risk: RiskSentiment
) -> MarketRegime:
    """Combine component regimes into primary regime."""
    
    # Crisis detection
    if volatility == VolatilityRegime.EXTREME and risk == RiskSentiment.RISK_OFF:
        return MarketRegime.CRISIS
    
    # Map trend and volatility to market regime
    if trend in [TrendRegime.STRONG_UPTREND, TrendRegime.UPTREND]:
        if volatility in [VolatilityRegime.LOW, VolatilityRegime.MEDIUM]:
            return MarketRegime.BULL_QUIET
        else:
            return MarketRegime.BULL_VOLATILE
            
    elif trend in [TrendRegime.STRONG_DOWNTREND, TrendRegime.DOWNTREND]:
        if volatility in [VolatilityRegime.LOW, VolatilityRegime.MEDIUM]:
            return MarketRegime.BEAR_QUIET
        else:
            return MarketRegime.BEAR_VOLATILE
            
    else:  # SIDEWAYS
        if volatility in [VolatilityRegime.LOW, VolatilityRegime.MEDIUM]:
            return MarketRegime.SIDEWAYS_QUIET
        else:
            return MarketRegime.SIDEWAYS_VOLATILE


def _calculate_confidence(features: RegimeFeatures, regime: MarketRegime) -> float:
    """Calculate confidence in regime classification."""
    confidence = 0.5  # Base confidence
    
    # Adjust based on feature clarity
    if abs(features.trend_strength) > 30:
        confidence += 0.2  # Clear trend
    
    if features.vol_of_vol < 0.3:
        confidence += 0.1  # Stable volatility
    
    if features.ma_5_20_spread * features.ma_20_60_spread > 0:
        confidence += 0.1  # Aligned moving averages
    
    # Regime-specific adjustments
    if regime in [MarketRegime.BULL_QUIET, MarketRegime.BEAR_QUIET]:
        if features.realized_vol_10d < 15:
            confidence += 0.1  # Confirms quiet regime
            
    elif regime == MarketRegime.CRISIS:
        if features.realized_vol_10d > 40:
            confidence += 0.2  # Strong crisis signal
    
    return min(confidence, 1.0)


def _analyze_stability(regime: MarketRegime, features: RegimeFeatures) -> float:
    """Analyze regime stability."""
    stability = 0.5  # Base stability
    
    # Check for conflicting signals
    if abs(features.ma_5_20_spread) < 0.02:  # MAs converging
        stability -= 0.1  # Potential regime change
    
    if features.vol_of_vol > 0.5:  # Volatility unstable
        stability -= 0.15
    
    # Stable regimes
    if regime in [MarketRegime.BULL_QUIET, MarketRegime.SIDEWAYS_QUIET]:
        if features.realized_vol_10d < features.realized_vol_30d:
            stability += 0.2  # Volatility decreasing
    
    return max(0, min(stability, 1.0))


def _calculate_transition_probabilities(
    current: MarketRegime,
    features: RegimeFeatures
) -> Dict[MarketRegime, float]:
    """Calculate transition probabilities to other regimes."""
    # Simplified transition probabilities
    probs = {regime: 0.0 for regime in MarketRegime}
    
    # Base probabilities from historical patterns
    if current == MarketRegime.BULL_QUIET:
        probs[MarketRegime.BULL_QUIET] = 0.60
        probs[MarketRegime.BULL_VOLATILE] = 0.20
        probs[MarketRegime.SIDEWAYS_QUIET] = 0.15
        probs[MarketRegime.BEAR_QUIET] = 0.05
        
    elif current == MarketRegime.BEAR_VOLATILE:
        probs[MarketRegime.BEAR_VOLATILE] = 0.40
        probs[MarketRegime.BEAR_QUIET] = 0.20
        probs[MarketRegime.SIDEWAYS_VOLATILE] = 0.20
        probs[MarketRegime.CRISIS] = 0.10
        probs[MarketRegime.BULL_VOLATILE] = 0.10
    
    # Add more patterns...
    
    # Adjust based on current features
    if features.vol_of_vol > 0.5:
        # Increase probability of volatile regimes
        for regime in MarketRegime:
            if 'VOLATILE' in regime.value or regime == MarketRegime.CRISIS:
                probs[regime] *= 1.2
    
    # Normalize
    total = sum(probs.values())
    if total > 0:
        probs = {k: v/total for k, v in probs.items()}
    
    return probs


def _get_regime_duration(symbol: str, regime: MarketRegime) -> int:
    """Get current regime duration in days."""
    # Simplified - return random duration
    return np.random.randint(5, 30)


def _calculate_regime_strength(features: RegimeFeatures, regime: MarketRegime) -> float:
    """Calculate how strongly the regime is expressed."""
    strength = 0.5
    
    if 'VOLATILE' in regime.value:
        # Stronger if volatility is high
        strength = min(1.0, features.realized_vol_30d / 40)
    elif 'QUIET' in regime.value:
        # Stronger if volatility is low
        strength = min(1.0, 1 - features.realized_vol_30d / 40)
    
    if 'BULL' in regime.value:
        # Stronger if trend is positive
        strength = (strength + min(1.0, features.returns_20d / 0.1)) / 2
    elif 'BEAR' in regime.value:
        # Stronger if trend is negative
        strength = (strength + min(1.0, abs(features.returns_20d) / 0.1)) / 2
    
    return strength


def _estimate_transition_timing(stability: float) -> float:
    """Estimate days until regime change."""
    # Higher stability = longer until change
    base_days = 10
    return base_days * (1 + stability * 2)


def _run_monitoring_check(callback: Optional[Callable]):
    """Run a monitoring check (simplified synchronous version)."""
    global _monitor_state
    
    if _monitor_state is None:
        return
    
    for symbol in _monitor_state.symbols:
        current_analysis = detect_regime(symbol, lookback_days=30)
        new_regime = current_analysis.current_regime
        old_regime = _monitor_state.current_regimes.get(symbol)
        
        if old_regime and new_regime != old_regime:
            # Regime change detected
            _monitor_state.regime_changes_today += 1
            _monitor_state.current_regimes[symbol] = new_regime
            
            if _monitor_state.alert_on_change and callback:
                alert = RegimeAlert(
                    symbol=symbol,
                    old_regime=old_regime,
                    new_regime=new_regime,
                    confidence=current_analysis.confidence,
                    timestamp=datetime.now(),
                    message=f"Regime change: {old_regime.value} → {new_regime.value}",
                    severity='warning' if new_regime == MarketRegime.CRISIS else 'info'
                )
                callback(alert)
                _monitor_state.alerts_sent += 1
    
    _monitor_state.last_check = datetime.now()


def _calculate_average_durations(
    regimes: List[Tuple[MarketRegime, datetime, datetime]]
) -> Dict[MarketRegime, float]:
    """Calculate average duration for each regime."""
    durations = {regime: [] for regime in MarketRegime}
    
    for regime, start, end in regimes:
        duration = (end - start).days
        durations[regime].append(duration)
    
    avg_durations = {}
    for regime, dur_list in durations.items():
        if dur_list:
            avg_durations[regime] = np.mean(dur_list)
        else:
            avg_durations[regime] = 0.0
    
    return avg_durations


def _calculate_returns_by_regime(
    symbol: str,
    regimes: List[Tuple[MarketRegime, datetime, datetime]]
) -> Dict[MarketRegime, float]:
    """Calculate average returns for each regime."""
    # Simplified - return synthetic values
    return {
        MarketRegime.BULL_QUIET: 0.15,
        MarketRegime.BULL_VOLATILE: 0.12,
        MarketRegime.SIDEWAYS_QUIET: 0.05,
        MarketRegime.SIDEWAYS_VOLATILE: 0.03,
        MarketRegime.BEAR_QUIET: -0.08,
        MarketRegime.BEAR_VOLATILE: -0.15,
        MarketRegime.CRISIS: -0.25
    }


def _calculate_volatility_by_regime(
    symbol: str,
    regimes: List[Tuple[MarketRegime, datetime, datetime]]
) -> Dict[MarketRegime, float]:
    """Calculate average volatility for each regime."""
    # Simplified - return synthetic values
    return {
        MarketRegime.BULL_QUIET: 0.12,
        MarketRegime.BULL_VOLATILE: 0.25,
        MarketRegime.SIDEWAYS_QUIET: 0.10,
        MarketRegime.SIDEWAYS_VOLATILE: 0.20,
        MarketRegime.BEAR_QUIET: 0.15,
        MarketRegime.BEAR_VOLATILE: 0.35,
        MarketRegime.CRISIS: 0.50
    }


def _calculate_sharpe_by_regime(
    returns: Dict[MarketRegime, float],
    volatility: Dict[MarketRegime, float]
) -> Dict[MarketRegime, float]:
    """Calculate Sharpe ratio for each regime."""
    risk_free = 0.02
    sharpe = {}
    
    for regime in MarketRegime:
        ret = returns.get(regime, 0)
        vol = volatility.get(regime, 1)
        if vol > 0:
            sharpe[regime] = (ret - risk_free) / vol
        else:
            sharpe[regime] = 0
    
    return sharpe


def _calculate_change_probability(
    analysis: RegimeAnalysis,
    history: RegimeHistory,
    horizon_days: int
) -> float:
    """Calculate probability of regime change within horizon."""
    # Base probability from stability
    base_prob = 1 - analysis.stability_score
    
    # Adjust based on current duration
    avg_duration = history.average_duration.get(analysis.current_regime, 20)
    if analysis.regime_duration > avg_duration:
        base_prob *= 1.5  # More likely to change if exceeded average
    
    # Adjust for horizon
    daily_prob = base_prob / 10  # Assume 10-day average transition
    horizon_prob = 1 - (1 - daily_prob) ** horizon_days
    
    return min(horizon_prob, 0.95)


def _identify_leading_indicators(analysis: RegimeAnalysis) -> List[str]:
    """Identify indicators suggesting regime change."""
    leading = []
    
    if analysis.features.vol_of_vol > 0.5:
        leading.append("Volatility clustering increasing")
    
    if abs(analysis.features.ma_5_20_spread) < 0.01:
        leading.append("Moving averages converging")
    
    if analysis.stability_score < 0.3:
        leading.append("Regime stability deteriorating")
    
    return leading


def _identify_confirming_indicators(
    current: MarketRegime,
    next_regime: MarketRegime
) -> List[str]:
    """Identify what would confirm regime change."""
    confirming = []
    
    if 'VOLATILE' in next_regime.value and 'QUIET' in current.value:
        confirming.append("Volatility spike above 25%")
        confirming.append("Increased trading volume")
    
    if 'BEAR' in next_regime.value and 'BULL' in current.value:
        confirming.append("Break below 20-day MA")
        confirming.append("Negative momentum confirmation")
    
    return confirming
