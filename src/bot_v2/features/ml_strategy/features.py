"""
Feature engineering for ML strategy selection - LOCAL to this slice.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from .types import MarketConditions


def extract_market_features(conditions: MarketConditions) -> np.ndarray:
    """
    Extract numerical features from market conditions.
    
    Returns feature vector for ML model.
    """
    features = [
        conditions.volatility / 100,  # Normalize to 0-1
        (conditions.trend_strength + 100) / 200,  # Normalize -100 to 100 -> 0 to 1
        min(conditions.volume_ratio / 3, 1),  # Cap at 3x volume
        (conditions.price_momentum + 50) / 100,  # Normalize momentum
        1.0 if conditions.market_regime == 'bull' else 0.0,
        1.0 if conditions.market_regime == 'bear' else 0.0,
        1.0 if conditions.market_regime == 'sideways' else 0.0,
        conditions.vix_level / 100,  # Normalize VIX
        (conditions.correlation_spy + 1) / 2,  # Normalize -1 to 1 -> 0 to 1
    ]
    
    # Add engineered features
    features.extend(engineer_features(conditions))
    
    return np.array(features)


def engineer_features(conditions: MarketConditions) -> List[float]:
    """
    Engineer additional features from base market conditions.
    
    Creates interaction and polynomial features.
    """
    engineered = []
    
    # Volatility regime features
    low_vol = 1.0 if conditions.volatility < 15 else 0.0
    mid_vol = 1.0 if 15 <= conditions.volatility <= 30 else 0.0
    high_vol = 1.0 if conditions.volatility > 30 else 0.0
    engineered.extend([low_vol, mid_vol, high_vol])
    
    # Trend regime features
    strong_uptrend = 1.0 if conditions.trend_strength > 50 else 0.0
    weak_uptrend = 1.0 if 0 < conditions.trend_strength <= 50 else 0.0
    weak_downtrend = 1.0 if -50 <= conditions.trend_strength < 0 else 0.0
    strong_downtrend = 1.0 if conditions.trend_strength < -50 else 0.0
    engineered.extend([strong_uptrend, weak_uptrend, weak_downtrend, strong_downtrend])
    
    # Interaction features
    vol_trend_interaction = conditions.volatility * abs(conditions.trend_strength) / 10000
    momentum_volume_interaction = conditions.price_momentum * conditions.volume_ratio / 100
    vix_correlation_interaction = conditions.vix_level * abs(conditions.correlation_spy) / 100
    
    engineered.extend([
        vol_trend_interaction,
        momentum_volume_interaction,
        vix_correlation_interaction
    ])
    
    # Market stress indicator
    stress = (conditions.vix_level / 100) * (conditions.volatility / 100)
    engineered.append(stress)
    
    # Trend consistency
    trend_consistency = abs(conditions.trend_strength) / 100 * (1 - conditions.volatility / 100)
    engineered.append(trend_consistency)
    
    return engineered


def get_feature_names() -> List[str]:
    """Get human-readable feature names for model interpretation."""
    base_features = [
        'volatility_norm',
        'trend_strength_norm',
        'volume_ratio_norm',
        'price_momentum_norm',
        'is_bull_market',
        'is_bear_market',
        'is_sideways_market',
        'vix_level_norm',
        'spy_correlation_norm'
    ]
    
    engineered_features = [
        'low_volatility',
        'mid_volatility',
        'high_volatility',
        'strong_uptrend',
        'weak_uptrend',
        'weak_downtrend',
        'strong_downtrend',
        'vol_trend_interaction',
        'momentum_volume_interaction',
        'vix_correlation_interaction',
        'market_stress',
        'trend_consistency'
    ]
    
    return base_features + engineered_features