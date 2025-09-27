"""
Feature engineering for ML strategy selection - LOCAL to this slice.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from .types import MarketConditions

# Import error handling
from bot_v2.errors import ValidationError, DataError

logger = logging.getLogger(__name__)


def extract_market_features(conditions: MarketConditions) -> np.ndarray:
    """
    Extract numerical features from market conditions with validation.
    
    Returns feature vector for ML model.
    """
    # Validate input conditions
    if conditions is None:
        raise ValidationError("Market conditions cannot be None")
    
    try:
        # Validate individual condition values before processing
        validated_conditions = _validate_market_conditions(conditions)
        
        features = [
            validated_conditions.volatility / 100,  # Normalize to 0-1
            (validated_conditions.trend_strength + 100) / 200,  # Normalize -100 to 100 -> 0 to 1
            min(validated_conditions.volume_ratio / 3, 1),  # Cap at 3x volume
            (validated_conditions.price_momentum + 50) / 100,  # Normalize momentum
            1.0 if validated_conditions.market_regime == 'bull' else 0.0,
            1.0 if validated_conditions.market_regime == 'bear' else 0.0,
            1.0 if validated_conditions.market_regime == 'sideways' else 0.0,
            validated_conditions.vix_level / 100,  # Normalize VIX
            (validated_conditions.correlation_spy + 1) / 2,  # Normalize -1 to 1 -> 0 to 1
        ]
        
        # Add engineered features with error handling
        try:
            engineered = engineer_features(validated_conditions)
            features.extend(engineered)
        except Exception as e:
            logger.warning(f"Failed to engineer features: {e}, using basic features only")
            # Continue with basic features if engineering fails
        
        # Convert to numpy array and validate
        feature_array = np.array(features)
        
        # Final validation of feature array
        _validate_feature_array(feature_array)
        
        return feature_array
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise DataError(f"Failed to extract features from market conditions: {str(e)}")


def engineer_features(conditions: MarketConditions) -> List[float]:
    """
    Engineer additional features from base market conditions with validation.
    
    Creates interaction and polynomial features.
    """
    if conditions is None:
        raise ValidationError("Market conditions cannot be None")
    
    try:
        # Validate conditions before processing
        validated_conditions = _validate_market_conditions(conditions)
        
        engineered = []
        
        # Volatility regime features with bounds checking
        try:
            vol = validated_conditions.volatility
            low_vol = 1.0 if vol < 15 else 0.0
            mid_vol = 1.0 if 15 <= vol <= 30 else 0.0
            high_vol = 1.0 if vol > 30 else 0.0
            engineered.extend([low_vol, mid_vol, high_vol])
        except Exception as e:
            logger.warning(f"Failed to create volatility regime features: {e}")
            engineered.extend([0.0, 1.0, 0.0])  # Default to mid volatility
        
        # Trend regime features with bounds checking
        try:
            trend = validated_conditions.trend_strength
            strong_uptrend = 1.0 if trend > 50 else 0.0
            weak_uptrend = 1.0 if 0 < trend <= 50 else 0.0
            weak_downtrend = 1.0 if -50 <= trend < 0 else 0.0
            strong_downtrend = 1.0 if trend < -50 else 0.0
            engineered.extend([strong_uptrend, weak_uptrend, weak_downtrend, strong_downtrend])
        except Exception as e:
            logger.warning(f"Failed to create trend regime features: {e}")
            engineered.extend([0.0, 0.0, 0.0, 1.0])  # Default to sideways
        
        # Interaction features with safe calculations
        try:
            # Ensure safe division and multiplication
            vol = max(0.01, validated_conditions.volatility)  # Avoid zero
            trend = validated_conditions.trend_strength
            momentum = validated_conditions.price_momentum
            volume_ratio = max(0.01, validated_conditions.volume_ratio)  # Avoid zero
            vix = max(1.0, validated_conditions.vix_level)  # Avoid zero
            correlation = validated_conditions.correlation_spy
            
            vol_trend_interaction = (vol * abs(trend)) / 10000
            momentum_volume_interaction = (momentum * volume_ratio) / 100
            vix_correlation_interaction = (vix * abs(correlation)) / 100
            
            # Validate interaction features
            interactions = [vol_trend_interaction, momentum_volume_interaction, vix_correlation_interaction]
            for i, interaction in enumerate(interactions):
                if np.isnan(interaction) or np.isinf(interaction):
                    logger.warning(f"Invalid interaction feature {i}: {interaction}")
                    interactions[i] = 0.0
                elif abs(interaction) > 10:  # Cap extreme values
                    interactions[i] = np.sign(interaction) * 10
            
            engineered.extend(interactions)
            
        except Exception as e:
            logger.warning(f"Failed to create interaction features: {e}")
            engineered.extend([0.0, 0.0, 0.0])  # Default neutral interactions
        
        # Market stress indicator with safe calculation
        try:
            stress = (validated_conditions.vix_level / 100) * (validated_conditions.volatility / 100)
            if np.isnan(stress) or np.isinf(stress) or stress < 0:
                stress = 0.2  # Default moderate stress
            elif stress > 2.0:  # Cap extreme stress
                stress = 2.0
            engineered.append(stress)
        except Exception as e:
            logger.warning(f"Failed to create stress indicator: {e}")
            engineered.append(0.2)  # Default moderate stress
        
        # Trend consistency with safe calculation
        try:
            trend_abs = abs(validated_conditions.trend_strength)
            vol_factor = max(0.01, validated_conditions.volatility)  # Avoid division by zero
            trend_consistency = (trend_abs / 100) * (1 - min(vol_factor / 100, 0.99))
            
            if np.isnan(trend_consistency) or np.isinf(trend_consistency):
                trend_consistency = 0.0
            elif trend_consistency > 1.0:
                trend_consistency = 1.0
            
            engineered.append(trend_consistency)
        except Exception as e:
            logger.warning(f"Failed to create trend consistency feature: {e}")
            engineered.append(0.0)  # Default no consistency
        
        # Final validation of all engineered features
        for i, feature in enumerate(engineered):
            if np.isnan(feature) or np.isinf(feature):
                logger.warning(f"Invalid engineered feature at index {i}: {feature}")
                engineered[i] = 0.0
        
        return engineered
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        # Return minimal safe features
        return [0.0] * 11  # 3 vol + 4 trend + 3 interactions + 1 stress + 1 consistency


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


# Helper and validation functions

def _validate_market_conditions(conditions: MarketConditions) -> MarketConditions:
    """Validate and sanitize market conditions."""
    try:
        # Create a copy to avoid modifying the original
        validated = MarketConditions(
            volatility=_validate_numeric_field(conditions.volatility, "volatility", 0.0, 200.0, 20.0),
            trend_strength=_validate_numeric_field(conditions.trend_strength, "trend_strength", -100.0, 100.0, 0.0),
            volume_ratio=_validate_numeric_field(conditions.volume_ratio, "volume_ratio", 0.1, 10.0, 1.0),
            price_momentum=_validate_numeric_field(conditions.price_momentum, "price_momentum", -100.0, 100.0, 0.0),
            market_regime=_validate_regime_field(conditions.market_regime),
            vix_level=_validate_numeric_field(conditions.vix_level, "vix_level", 1.0, 200.0, 20.0),
            correlation_spy=_validate_numeric_field(conditions.correlation_spy, "correlation_spy", -1.0, 1.0, 0.7)
        )
        
        return validated
        
    except Exception as e:
        logger.error(f"Failed to validate market conditions: {e}")
        # Return default safe conditions
        return MarketConditions(
            volatility=20.0,
            trend_strength=0.0,
            volume_ratio=1.0,
            price_momentum=0.0,
            market_regime='sideways',
            vix_level=20.0,
            correlation_spy=0.7
        )


def _validate_numeric_field(value: float, field_name: str, min_val: float, max_val: float, default_val: float) -> float:
    """Validate and sanitize a numeric field."""
    try:
        if value is None:
            logger.warning(f"{field_name} is None, using default {default_val}")
            return default_val
        
        if np.isnan(value) or np.isinf(value):
            logger.warning(f"{field_name} is NaN/inf ({value}), using default {default_val}")
            return default_val
        
        if value < min_val:
            logger.warning(f"{field_name} below minimum ({value} < {min_val}), clipping to {min_val}")
            return min_val
        
        if value > max_val:
            logger.warning(f"{field_name} above maximum ({value} > {max_val}), clipping to {max_val}")
            return max_val
        
        return float(value)
        
    except Exception as e:
        logger.warning(f"Failed to validate {field_name}: {e}, using default {default_val}")
        return default_val


def _validate_regime_field(regime: str) -> str:
    """Validate and sanitize market regime field."""
    try:
        if regime is None:
            return 'sideways'
        
        regime_str = str(regime).lower().strip()
        valid_regimes = ['bull', 'bear', 'sideways']
        
        if regime_str in valid_regimes:
            return regime_str
        
        # Try to match partial strings
        for valid_regime in valid_regimes:
            if valid_regime in regime_str:
                logger.info(f"Matched partial regime '{regime_str}' to '{valid_regime}'")
                return valid_regime
        
        logger.warning(f"Invalid market regime '{regime}', using 'sideways'")
        return 'sideways'
        
    except Exception as e:
        logger.warning(f"Failed to validate market regime: {e}, using 'sideways'")
        return 'sideways'


def _validate_feature_array(features: np.ndarray):
    """Validate the final feature array."""
    if features is None:
        raise DataError("Feature array cannot be None")
    
    if len(features) == 0:
        raise DataError("Feature array cannot be empty")
    
    if np.any(np.isnan(features)):
        nan_indices = np.where(np.isnan(features))[0]
        raise DataError(f"Feature array contains NaN values at indices: {nan_indices}")
    
    if np.any(np.isinf(features)):
        inf_indices = np.where(np.isinf(features))[0]
        raise DataError(f"Feature array contains infinite values at indices: {inf_indices}")
    
    # Check for reasonable feature ranges
    if np.any(np.abs(features) > 100):
        extreme_indices = np.where(np.abs(features) > 100)[0]
        logger.warning(f"Feature array contains extreme values at indices: {extreme_indices}")
        # Clip extreme values
        features[extreme_indices] = np.clip(features[extreme_indices], -100, 100)
    
    expected_length = len(get_feature_names())
    if len(features) != expected_length:
        raise DataError(f"Feature array length mismatch: expected {expected_length}, got {len(features)}")


def validate_feature_input(features: np.ndarray, feature_name: str = "features") -> np.ndarray:
    """Public function to validate feature arrays."""
    try:
        if features is None:
            raise ValidationError(f"{feature_name} cannot be None")
        
        if len(features) == 0:
            raise ValidationError(f"{feature_name} cannot be empty")
        
        # Convert to numpy array if needed
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # Validate the array
        _validate_feature_array(features)
        
        return features
        
    except Exception as e:
        logger.error(f"Feature validation failed for {feature_name}: {e}")
        raise DataError(f"Invalid {feature_name}: {str(e)}")


def get_feature_bounds() -> Dict[str, tuple]:
    """Get expected bounds for each feature."""
    feature_names = get_feature_names()
    
    # Define reasonable bounds for each feature
    bounds = {}
    
    # Base features
    bounds['volatility_norm'] = (0.0, 2.0)
    bounds['trend_strength_norm'] = (0.0, 1.0)
    bounds['volume_ratio_norm'] = (0.0, 1.0)
    bounds['price_momentum_norm'] = (0.0, 1.0)
    bounds['is_bull_market'] = (0.0, 1.0)
    bounds['is_bear_market'] = (0.0, 1.0)
    bounds['is_sideways_market'] = (0.0, 1.0)
    bounds['vix_level_norm'] = (0.0, 2.0)
    bounds['spy_correlation_norm'] = (0.0, 1.0)
    
    # Engineered features
    bounds['low_volatility'] = (0.0, 1.0)
    bounds['mid_volatility'] = (0.0, 1.0)
    bounds['high_volatility'] = (0.0, 1.0)
    bounds['strong_uptrend'] = (0.0, 1.0)
    bounds['weak_uptrend'] = (0.0, 1.0)
    bounds['weak_downtrend'] = (0.0, 1.0)
    bounds['strong_downtrend'] = (0.0, 1.0)
    bounds['vol_trend_interaction'] = (0.0, 10.0)
    bounds['momentum_volume_interaction'] = (-10.0, 10.0)
    bounds['vix_correlation_interaction'] = (0.0, 2.0)
    bounds['market_stress'] = (0.0, 2.0)
    bounds['trend_consistency'] = (0.0, 1.0)
    
    return bounds


def create_safe_features(symbol: str = "UNKNOWN") -> np.ndarray:
    """Create a safe fallback feature array when feature extraction fails."""
    logger.info(f"Creating safe fallback features for {symbol}")
    
    # Create conservative default features
    safe_conditions = MarketConditions(
        volatility=20.0,      # Moderate volatility
        trend_strength=0.0,   # Neutral trend
        volume_ratio=1.0,     # Normal volume
        price_momentum=0.0,   # No momentum
        market_regime='sideways',  # Neutral regime
        vix_level=20.0,       # Normal VIX
        correlation_spy=0.7   # Moderate correlation
    )
    
    try:
        return extract_market_features(safe_conditions)
    except Exception as e:
        logger.error(f"Failed to create safe features: {e}")
        # Return manual fallback array
        feature_count = len(get_feature_names())
        return np.full(feature_count, 0.5)  # All features at neutral level