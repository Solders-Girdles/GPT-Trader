"""
Regime transition analysis - LOCAL to this slice.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from .types import MarketRegime, RegimeTransition


def calculate_transition_matrix(
    transitions: List[RegimeTransition]
) -> Dict[MarketRegime, Dict[MarketRegime, float]]:
    """
    Calculate regime transition probability matrix from historical transitions.
    
    Args:
        transitions: List of historical regime transitions
        
    Returns:
        Matrix of transition probabilities
    """
    # Initialize transition counts
    transition_counts = {}
    total_counts = {}
    
    for regime in MarketRegime:
        transition_counts[regime] = {r: 0 for r in MarketRegime}
        total_counts[regime] = 0
    
    # Count transitions
    for transition in transitions:
        from_regime = transition.from_regime
        to_regime = transition.to_regime
        
        transition_counts[from_regime][to_regime] += 1
        total_counts[from_regime] += 1
    
    # Calculate probabilities
    transition_matrix = {}
    for from_regime in MarketRegime:
        transition_matrix[from_regime] = {}
        total = total_counts[from_regime]
        
        if total > 0:
            for to_regime in MarketRegime:
                probability = transition_counts[from_regime][to_regime] / total
                transition_matrix[from_regime][to_regime] = probability
        else:
            # Default uniform distribution if no data
            for to_regime in MarketRegime:
                transition_matrix[from_regime][to_regime] = 1.0 / len(MarketRegime)
    
    return transition_matrix


def analyze_transitions(
    regimes: List[Tuple[MarketRegime, datetime, datetime]]
) -> List[RegimeTransition]:
    """
    Analyze regime transitions from historical regime periods.
    
    Args:
        regimes: List of (regime, start_date, end_date) tuples
        
    Returns:
        List of regime transitions with analysis
    """
    transitions = []
    
    for i in range(1, len(regimes)):
        prev_regime, prev_start, prev_end = regimes[i-1]
        curr_regime, curr_start, curr_end = regimes[i]
        
        # Calculate transition duration
        transition_duration = (curr_start - prev_end).days
        
        # Identify trigger events (simplified)
        trigger_events = identify_transition_triggers(prev_regime, curr_regime)
        
        transition = RegimeTransition(
            from_regime=prev_regime,
            to_regime=curr_regime,
            transition_date=curr_start,
            duration_days=transition_duration,
            trigger_events=trigger_events
        )
        
        transitions.append(transition)
    
    return transitions


def identify_transition_triggers(
    from_regime: MarketRegime,
    to_regime: MarketRegime
) -> List[str]:
    """
    Identify likely triggers for regime transitions.
    
    Based on typical market dynamics.
    """
    triggers = []
    
    # From quiet to volatile regimes
    if 'QUIET' in from_regime.value and 'VOLATILE' in to_regime.value:
        triggers.extend([
            "Volatility spike",
            "News event impact",
            "Market uncertainty increase"
        ])
    
    # From volatile to quiet regimes
    elif 'VOLATILE' in from_regime.value and 'QUIET' in to_regime.value:
        triggers.extend([
            "Volatility normalization",
            "Market stabilization",
            "Uncertainty resolution"
        ])
    
    # Bull to bear transitions
    elif 'BULL' in from_regime.value and 'BEAR' in to_regime.value:
        triggers.extend([
            "Economic concerns",
            "Earnings disappointments",
            "Policy changes",
            "Technical breakdown"
        ])
    
    # Bear to bull transitions
    elif 'BEAR' in from_regime.value and 'BULL' in to_regime.value:
        triggers.extend([
            "Economic recovery signals",
            "Policy support",
            "Oversold bounce",
            "Sentiment improvement"
        ])
    
    # To crisis regime
    elif to_regime == MarketRegime.CRISIS:
        triggers.extend([
            "Market crash",
            "Systemic risk event",
            "Liquidity crisis",
            "Black swan event"
        ])
    
    # From crisis regime
    elif from_regime == MarketRegime.CRISIS:
        triggers.extend([
            "Policy intervention",
            "Liquidity restoration",
            "Market stabilization",
            "Crisis resolution"
        ])
    
    # Sideways transitions
    elif 'SIDEWAYS' in to_regime.value:
        triggers.extend([
            "Trend exhaustion",
            "Consolidation phase",
            "Range establishment"
        ])
    
    return triggers


def calculate_transition_statistics(
    transitions: List[RegimeTransition]
) -> Dict[str, any]:
    """
    Calculate statistics about regime transitions.
    
    Returns:
        Dictionary of transition statistics
    """
    if not transitions:
        return {}
    
    stats = {}
    
    # Transition frequency
    transition_counts = {}
    for transition in transitions:
        key = f"{transition.from_regime.value} → {transition.to_regime.value}"
        transition_counts[key] = transition_counts.get(key, 0) + 1
    
    stats['transition_counts'] = transition_counts
    stats['most_common_transition'] = max(transition_counts, key=transition_counts.get)
    
    # Transition durations
    durations = [t.duration_days for t in transitions]
    stats['avg_transition_duration'] = np.mean(durations)
    stats['median_transition_duration'] = np.median(durations)
    stats['min_transition_duration'] = np.min(durations)
    stats['max_transition_duration'] = np.max(durations)
    
    # Regime stability analysis
    regime_durations = {}
    for transition in transitions:
        regime = transition.from_regime
        if regime not in regime_durations:
            regime_durations[regime] = []
        # Note: This is simplified - would need regime periods for accurate calculation
    
    # Common triggers
    all_triggers = []
    for transition in transitions:
        all_triggers.extend(transition.trigger_events)
    
    trigger_counts = {}
    for trigger in all_triggers:
        trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
    
    stats['common_triggers'] = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)
    
    return stats


def predict_next_regime(
    current_regime: MarketRegime,
    transition_matrix: Dict[MarketRegime, Dict[MarketRegime, float]],
    current_duration: int,
    market_signals: Dict[str, float]
) -> Tuple[MarketRegime, float]:
    """
    Predict next regime based on transition probabilities and current signals.
    
    Args:
        current_regime: Current market regime
        transition_matrix: Historical transition probabilities
        current_duration: Days in current regime
        market_signals: Current market indicators
        
    Returns:
        (predicted_regime, confidence)
    """
    # Get base transition probabilities
    base_probs = transition_matrix.get(current_regime, {})
    
    # Adjust probabilities based on current signals
    adjusted_probs = adjust_probabilities_by_signals(base_probs, market_signals)
    
    # Adjust for regime duration (longer duration = higher change probability)
    adjusted_probs = adjust_probabilities_by_duration(adjusted_probs, current_duration, current_regime)
    
    # Find most likely next regime (excluding current)
    next_regime_probs = {k: v for k, v in adjusted_probs.items() if k != current_regime}
    
    if not next_regime_probs:
        return current_regime, 0.5
    
    most_likely = max(next_regime_probs, key=next_regime_probs.get)
    confidence = next_regime_probs[most_likely]
    
    return most_likely, confidence


def adjust_probabilities_by_signals(
    base_probs: Dict[MarketRegime, float],
    signals: Dict[str, float]
) -> Dict[MarketRegime, float]:
    """
    Adjust transition probabilities based on current market signals.
    """
    adjusted = base_probs.copy()
    
    # Get signal values with defaults
    volatility = signals.get('volatility_20d', 20)
    trend = signals.get('momentum_10d', 0)
    volume_ratio = signals.get('volume_ratio', 1)
    rsi = signals.get('rsi', 50)
    
    # Adjust probabilities based on signals
    for regime in MarketRegime:
        adjustment = 1.0
        
        # Volatility adjustments
        if 'VOLATILE' in regime.value:
            if volatility > 25:
                adjustment *= 1.3  # High vol increases volatile regime probability
        elif 'QUIET' in regime.value:
            if volatility < 15:
                adjustment *= 1.2  # Low vol increases quiet regime probability
        
        # Trend adjustments
        if 'BULL' in regime.value:
            if trend > 5:
                adjustment *= 1.2  # Positive momentum increases bull probability
        elif 'BEAR' in regime.value:
            if trend < -5:
                adjustment *= 1.2  # Negative momentum increases bear probability
        
        # Crisis adjustments
        if regime == MarketRegime.CRISIS:
            if volatility > 40 or (volatility > 30 and trend < -10):
                adjustment *= 2.0  # Extreme conditions increase crisis probability
        
        # Volume confirmation
        if volume_ratio > 1.5:
            adjustment *= 1.1  # High volume confirms transitions
        
        adjusted[regime] = adjusted.get(regime, 0) * adjustment
    
    # Normalize probabilities
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v/total for k, v in adjusted.items()}
    
    return adjusted


def adjust_probabilities_by_duration(
    probs: Dict[MarketRegime, float],
    current_duration: int,
    current_regime: MarketRegime
) -> Dict[MarketRegime, float]:
    """
    Adjust probabilities based on how long we've been in current regime.
    """
    adjusted = probs.copy()
    
    # Typical regime durations (simplified)
    typical_durations = {
        MarketRegime.BULL_QUIET: 60,
        MarketRegime.BULL_VOLATILE: 30,
        MarketRegime.BEAR_QUIET: 45,
        MarketRegime.BEAR_VOLATILE: 25,
        MarketRegime.SIDEWAYS_QUIET: 40,
        MarketRegime.SIDEWAYS_VOLATILE: 20,
        MarketRegime.CRISIS: 15
    }
    
    typical_duration = typical_durations.get(current_regime, 30)
    duration_factor = current_duration / typical_duration
    
    # If we've been in regime longer than typical, increase transition probability
    if duration_factor > 1.5:
        # Increase probability of transitioning away from current regime
        for regime in MarketRegime:
            if regime != current_regime:
                adjusted[regime] = adjusted.get(regime, 0) * (1 + 0.3 * (duration_factor - 1))
    
    # Normalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v/total for k, v in adjusted.items()}
    
    return adjusted


def find_regime_patterns(
    regimes: List[Tuple[MarketRegime, datetime, datetime]]
) -> Dict[str, any]:
    """
    Find patterns in regime sequences.
    
    Returns:
        Dictionary of identified patterns
    """
    patterns = {}
    
    if len(regimes) < 3:
        return patterns
    
    # Look for common sequences
    sequences = {}
    for i in range(len(regimes) - 2):
        seq = (regimes[i][0], regimes[i+1][0], regimes[i+2][0])
        seq_str = " → ".join([r.value for r in seq])
        sequences[seq_str] = sequences.get(seq_str, 0) + 1
    
    patterns['common_sequences'] = sorted(sequences.items(), key=lambda x: x[1], reverse=True)
    
    # Seasonal patterns (month-based)
    monthly_regimes = {}
    for regime, start_date, end_date in regimes:
        month = start_date.month
        if month not in monthly_regimes:
            monthly_regimes[month] = {}
        monthly_regimes[month][regime] = monthly_regimes[month].get(regime, 0) + 1
    
    patterns['monthly_patterns'] = monthly_regimes
    
    # Cycle analysis
    regime_sequence = [r[0] for r in regimes]
    cycle_length = estimate_cycle_length(regime_sequence)
    patterns['estimated_cycle_length'] = cycle_length
    
    return patterns


def estimate_cycle_length(regime_sequence: List[MarketRegime]) -> int:
    """
    Estimate average market cycle length from regime sequence.
    
    Simplified implementation.
    """
    if len(regime_sequence) < 4:
        return 0
    
    # Look for patterns where we return to similar regimes
    cycle_lengths = []
    
    for i in range(len(regime_sequence) - 1):
        current_regime = regime_sequence[i]
        
        # Find next occurrence of same or similar regime
        for j in range(i + 2, len(regime_sequence)):
            if regime_sequence[j] == current_regime:
                cycle_length = j - i
                cycle_lengths.append(cycle_length)
                break
    
    if cycle_lengths:
        return int(np.mean(cycle_lengths))
    
    return 0