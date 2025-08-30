#!/usr/bin/env python3
"""
Predictive Features Integration Module
===================================
Generated automatically by ml_features_predictive.py

Usage in training scripts:
from scripts.ml_features_predictive_integration import create_enhanced_predictive_features

features = create_enhanced_predictive_features(data)
"""

import pandas as pd
import numpy as np
from typing import List

# Selected robust features (20 total)
SELECTED_FEATURES = ['volume_surge', 'opening_gap', 'high_low_ratio', 'close_to_high', 'upper_wick_ratio', 'lower_wick_ratio', 'wick_imbalance', 'price_vs_vwap', 'momentum_1d', 'momentum_5d', 'momentum_20d', 'momentum_60d', 'momentum_acceleration_5d', 'momentum_acceleration_20d', 'momentum_divergence_short', 'momentum_divergence_long', 'momentum_roc_5d', 'momentum_roc_20d', 'velocity_consistency_5d', 'velocity_consistency_10d']

# Scaling parameters - simplified for compatibility
SCALING_PARAMS = {
    'volume_surge': {'median': 0.92679587, 'scale': 0.29858072}, 
    'opening_gap': {'median': 5.64330899e-05, 'scale': 0.00978967}, 
    'high_low_ratio': {'median': 0.01862545, 'scale': 0.01213355}, 
    'close_to_high': {'median': 0.53958779, 'scale': 0.5972739}, 
    'upper_wick_ratio': {'median': 0.46357114, 'scale': 1.40066706}, 
    'lower_wick_ratio': {'median': 0.4689056, 'scale': 1.30626508}, 
    'wick_imbalance': {'median': -0.00017486, 'scale': 0.00677662}, 
    'price_vs_vwap': {'median': 0.00148153, 'scale': 0.0235703}, 
    'momentum_1d': {'median': 0.0006632, 'scale': 0.01900556}, 
    'momentum_5d': {'median': 0.00167494, 'scale': 0.04968716}, 
    'momentum_20d': {'median': 0.0103228, 'scale': 0.11770507}, 
    'momentum_60d': {'median': 0.01889273, 'scale': 0.21405433}, 
    'momentum_acceleration_5d': {'median': -0.01178444, 'scale': 0.0953752}, 
    'momentum_acceleration_20d': {'median': -0.01290181, 'scale': 0.17176609}, 
    'momentum_divergence_short': {'median': 0.0, 'scale': 1.0}, 
    'momentum_divergence_long': {'median': 0.0, 'scale': 1.0}, 
    'momentum_roc_5d': {'median': -0.8151473, 'scale': 1.98121101}, 
    'momentum_roc_20d': {'median': -0.25811224, 'scale': 1.17497059}, 
    'velocity_consistency_5d': {'median': 0.03333954, 'scale': 0.72073}, 
    'velocity_consistency_10d': {'median': 0.06379086, 'scale': 0.54426364}
}

def create_enhanced_predictive_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create enhanced predictive features for ML models.
    
    This replaces the create_enhanced_features function in train_ml_profitable.py
    with features that actually predict future movements instead of describing past ones.
    
    Args:
        data: DataFrame with OHLCV columns
        
    Returns:
        DataFrame with scaled predictive features
    """
    # Generate all predictive features
    from ml_features_predictive import create_predictive_features, apply_robust_scaling
    
    all_features = create_predictive_features(data)
    
    # Select only robust features
    available_features = [f for f in SELECTED_FEATURES if f in all_features.columns]
    selected_data = all_features[available_features]
    
    # Apply scaling
    scaled_features, _ = apply_robust_scaling(selected_data)
    
    return scaled_features

def get_feature_importance() -> pd.DataFrame:
    """Return feature importance analysis."""
    import pandas as pd
    return pd.read_csv("models/feature_importance.csv", index_col=0)
