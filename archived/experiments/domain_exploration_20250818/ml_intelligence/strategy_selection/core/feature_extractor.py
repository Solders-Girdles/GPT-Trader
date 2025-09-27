"""
Production-grade Feature Extraction for ML Strategy Selection.

This module implements comprehensive feature engineering for converting
market data into ML-ready feature vectors. Provides standardized,
validated, and optimized feature extraction with extensive monitoring.

Key Features:
- Normalized feature extraction from market conditions
- Advanced feature engineering (interactions, polynomials)
- Feature validation and quality assessment
- Performance optimization for real-time inference
- Comprehensive logging and monitoring

Production Standards:
- Complete type hints with runtime validation
- Comprehensive error handling with specific exceptions
- Structured logging for all operations
- Performance optimizations
- Thread-safe design
- Cyclomatic complexity <10 per function
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd

from ..interfaces.types import (
    MarketConditions, MarketRegime, FeatureExtractor as FeatureExtractorProtocol,
    FeatureExtractionError, InvalidMarketDataError
)

# Configure module logger
logger = logging.getLogger(__name__)


class FeatureExtractor(FeatureExtractorProtocol):
    """
    Production-grade feature extractor for market conditions.
    
    Converts raw market conditions into normalized, engineered feature
    vectors suitable for ML model training and inference. Includes
    comprehensive validation, monitoring, and optimization.
    
    Features:
    - Base market indicators (normalized)
    - Regime one-hot encoding
    - Interaction features
    - Technical indicator derivatives
    - Quality assessment and validation
    
    Thread Safety:
        All public methods are thread-safe using internal locks.
    """
    
    def __init__(
        self,
        enable_feature_engineering: bool = True,
        enable_feature_selection: bool = False,
        feature_selection_k: int = 20,
        normalization_method: str = "standard",
        cache_features: bool = True,
        feature_validation: bool = True
    ) -> None:
        """
        Initialize Feature Extractor.
        
        Args:
            enable_feature_engineering: Whether to include engineered features
            enable_feature_selection: Whether to perform feature selection
            feature_selection_k: Number of features to select (if enabled)
            normalization_method: "standard", "robust", or "none"
            cache_features: Whether to cache extracted features
            feature_validation: Whether to validate feature quality
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if feature_selection_k < 1:
            raise ValueError(f"feature_selection_k must be positive, got {feature_selection_k}")
        
        if normalization_method not in ["standard", "robust", "none"]:
            raise ValueError(f"Invalid normalization method: {normalization_method}")
        
        self.enable_feature_engineering = enable_feature_engineering
        self.enable_feature_selection = enable_feature_selection
        self.feature_selection_k = feature_selection_k
        self.normalization_method = normalization_method
        self.cache_features = cache_features
        self.feature_validation = feature_validation
        
        # Feature extraction state
        self._feature_names: List[str] = []
        self._scaler: Optional[StandardScaler] = None
        self._feature_selector: Optional[SelectKBest] = None
        self._is_fitted = False
        
        # Performance tracking
        self._extraction_times: List[float] = []
        self._feature_stats: Dict[str, Any] = {}
        
        # Caching
        self._feature_cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize feature names
        self._initialize_feature_names()
        
        logger.info(f"Initialized FeatureExtractor with {len(self._feature_names)} features")
    
    def extract_features(self, market_conditions: MarketConditions) -> np.ndarray:
        """
        Extract numerical features from market conditions.
        
        Args:
            market_conditions: Market state to extract features from
            
        Returns:
            Feature vector as numpy array
            
        Raises:
            FeatureExtractionError: If feature extraction fails
            InvalidMarketDataError: If market conditions are invalid
        """
        with self._lock:
            start_time = time.time()
            
            try:
                # Validate market conditions
                if self.feature_validation:
                    self._validate_market_conditions(market_conditions)
                
                # Check cache
                cache_key = self._get_cache_key(market_conditions)
                if self.cache_features and cache_key in self._feature_cache:
                    self._cache_hits += 1
                    return self._feature_cache[cache_key].copy()
                
                self._cache_misses += 1
                
                # Extract base features
                features = self._extract_base_features(market_conditions)
                
                # Add regime features
                regime_features = self._extract_regime_features(market_conditions)
                features.extend(regime_features)
                
                # Add engineered features
                if self.enable_feature_engineering:
                    engineered_features = self._extract_engineered_features(market_conditions)
                    features.extend(engineered_features)
                
                # Convert to numpy array
                feature_vector = np.array(features, dtype=np.float64)
                
                # Validate feature vector
                if self.feature_validation:
                    self._validate_feature_vector(feature_vector)
                
                # Apply normalization if fitted
                if self._is_fitted and self._scaler is not None:
                    feature_vector = self._scaler.transform(feature_vector.reshape(1, -1))[0]
                
                # Apply feature selection if fitted
                if self._is_fitted and self._feature_selector is not None:
                    feature_vector = self._feature_selector.transform(feature_vector.reshape(1, -1))[0]
                
                # Cache result
                if self.cache_features:
                    self._feature_cache[cache_key] = feature_vector.copy()
                
                # Track performance
                extraction_time = time.time() - start_time
                self._extraction_times.append(extraction_time)
                
                # Update statistics
                self._update_feature_stats(feature_vector)
                
                logger.debug(f"Extracted {len(feature_vector)} features in {extraction_time:.4f}s")
                
                return feature_vector
                
            except InvalidMarketDataError:
                raise
            except Exception as e:
                logger.error(f"Feature extraction failed: {str(e)}")
                raise FeatureExtractionError(f"Failed to extract features: {str(e)}") from e
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of extracted features.
        
        Returns:
            List of feature names in order
        """
        with self._lock:
            if self.enable_feature_selection and self._feature_selector is not None:
                # Return names of selected features
                selected_indices = self._feature_selector.get_support(indices=True)
                return [self._feature_names[i] for i in selected_indices]
            
            return self._feature_names.copy()
    
    def fit(
        self, 
        market_conditions_list: List[MarketConditions],
        target_values: Optional[np.ndarray] = None
    ) -> None:
        """
        Fit feature normalization and selection.
        
        Args:
            market_conditions_list: List of market conditions for fitting
            target_values: Target values for feature selection (optional)
            
        Raises:
            ValueError: If insufficient data provided
            FeatureExtractionError: If fitting fails
        """
        with self._lock:
            if len(market_conditions_list) < 10:
                raise ValueError(f"Insufficient data for fitting: {len(market_conditions_list)} samples")
            
            try:
                logger.info(f"Fitting feature extractor on {len(market_conditions_list)} samples")
                
                # Extract features from all samples
                feature_matrix = []
                for conditions in market_conditions_list:
                    features = self._extract_base_features(conditions)
                    regime_features = self._extract_regime_features(conditions)
                    features.extend(regime_features)
                    
                    if self.enable_feature_engineering:
                        engineered_features = self._extract_engineered_features(conditions)
                        features.extend(engineered_features)
                    
                    feature_matrix.append(features)
                
                X = np.array(feature_matrix, dtype=np.float64)
                
                # Fit normalization
                if self.normalization_method == "standard":
                    self._scaler = StandardScaler()
                elif self.normalization_method == "robust":
                    self._scaler = RobustScaler()
                
                if self._scaler is not None:
                    X_scaled = self._scaler.fit_transform(X)
                else:
                    X_scaled = X
                
                # Fit feature selection
                if self.enable_feature_selection and target_values is not None:
                    if len(target_values) != len(X_scaled):
                        raise ValueError("Target values length must match samples")
                    
                    self._feature_selector = SelectKBest(
                        score_func=f_regression, 
                        k=min(self.feature_selection_k, X_scaled.shape[1])
                    )
                    self._feature_selector.fit(X_scaled, target_values)
                
                self._is_fitted = True
                
                # Calculate feature statistics
                self._calculate_feature_statistics(X_scaled)
                
                logger.info("Feature extractor fitting completed successfully")
                
            except Exception as e:
                logger.error(f"Feature extractor fitting failed: {str(e)}")
                raise FeatureExtractionError(f"Failed to fit feature extractor: {str(e)}") from e
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from feature selection.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        with self._lock:
            if not self._is_fitted or self._feature_selector is None:
                return {}
            
            try:
                scores = self._feature_selector.scores_
                selected_indices = self._feature_selector.get_support(indices=True)
                
                importance = {}
                for i, idx in enumerate(selected_indices):
                    feature_name = self._feature_names[idx]
                    importance[feature_name] = float(scores[idx])
                
                return importance
                
            except Exception as e:
                logger.warning(f"Failed to get feature importance: {str(e)}")
                return {}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get feature extraction performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        with self._lock:
            if not self._extraction_times:
                return {}
            
            return {
                "total_extractions": len(self._extraction_times),
                "avg_extraction_time_ms": np.mean(self._extraction_times) * 1000,
                "max_extraction_time_ms": np.max(self._extraction_times) * 1000,
                "min_extraction_time_ms": np.min(self._extraction_times) * 1000,
                "cache_hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses),
                "cache_size": len(self._feature_cache),
                "feature_stats": self._feature_stats.copy()
            }
    
    def clear_cache(self) -> None:
        """Clear feature cache."""
        with self._lock:
            self._feature_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            logger.info("Feature cache cleared")
    
    # Private methods
    
    def _initialize_feature_names(self) -> None:
        """Initialize feature names based on configuration."""
        # Base features
        base_names = [
            "volatility_norm", "trend_strength_norm", "volume_ratio_norm",
            "price_momentum_norm", "vix_level_norm", "spy_correlation_norm",
            "rsi_norm", "bollinger_position_norm", "atr_normalized"
        ]
        
        # Regime features
        regime_names = [f"regime_{regime.value}" for regime in MarketRegime]
        
        # Engineered features (if enabled)
        engineered_names = []
        if self.enable_feature_engineering:
            engineered_names = [
                "vol_trend_interaction", "momentum_volume_interaction",
                "vix_correlation_interaction", "market_stress",
                "trend_consistency", "volatility_regime_low",
                "volatility_regime_mid", "volatility_regime_high",
                "trend_momentum_alignment", "volume_volatility_ratio",
                "regime_stability_indicator", "momentum_persistence",
                "volatility_trend_divergence", "market_sentiment_composite"
            ]
        
        self._feature_names = base_names + regime_names + engineered_names
    
    def _extract_base_features(self, conditions: MarketConditions) -> List[float]:
        """Extract normalized base features."""
        return [
            conditions.volatility / 100,  # Normalize to 0-1
            (conditions.trend_strength + 100) / 200,  # Normalize -100 to 100 -> 0 to 1
            min(conditions.volume_ratio / 3, 1),  # Cap at 3x volume
            (conditions.price_momentum + 100) / 200,  # Normalize momentum
            conditions.vix_level / 100,  # Normalize VIX
            (conditions.correlation_spy + 1) / 2,  # Normalize -1 to 1 -> 0 to 1
            conditions.rsi / 100,  # Normalize RSI
            (conditions.bollinger_position + 2) / 4,  # Normalize to 0-1
            conditions.atr_normalized  # Already normalized
        ]
    
    def _extract_regime_features(self, conditions: MarketConditions) -> List[float]:
        """Extract one-hot encoded regime features."""
        regime_features = [0.0] * len(MarketRegime)
        regime_index = list(MarketRegime).index(conditions.market_regime)
        regime_features[regime_index] = 1.0
        return regime_features
    
    def _extract_engineered_features(self, conditions: MarketConditions) -> List[float]:
        """Extract advanced engineered features."""
        if not self.enable_feature_engineering:
            return []
        
        engineered = []
        
        # Interaction features
        vol_trend = conditions.volatility * abs(conditions.trend_strength) / 10000
        momentum_volume = conditions.price_momentum * conditions.volume_ratio / 100
        vix_correlation = conditions.vix_level * abs(conditions.correlation_spy) / 100
        
        # Market stress indicator
        stress = (conditions.vix_level / 100) * (conditions.volatility / 100)
        
        # Trend consistency
        trend_consistency = (
            abs(conditions.trend_strength) / 100 * 
            (1 - conditions.volatility / 100)
        )
        
        # Volatility regime indicators
        vol_low = 1.0 if conditions.volatility < 15 else 0.0
        vol_mid = 1.0 if 15 <= conditions.volatility <= 30 else 0.0
        vol_high = 1.0 if conditions.volatility > 30 else 0.0
        
        # Advanced engineered features
        trend_momentum_alignment = (
            conditions.trend_strength * conditions.price_momentum / 10000
        )
        
        volume_volatility_ratio = (
            conditions.volume_ratio / max(conditions.volatility / 20, 0.1)
        )
        
        # Regime stability (how well current conditions match regime)
        regime_stability = self._calculate_regime_stability(conditions)
        
        # Momentum persistence indicator
        momentum_persistence = abs(conditions.price_momentum) * (1 - conditions.volatility / 100)
        
        # Volatility-trend divergence
        vol_trend_divergence = abs(
            (conditions.volatility / 50) - abs(conditions.trend_strength / 100)
        )
        
        # Market sentiment composite
        sentiment_composite = (
            (conditions.rsi - 50) / 50 * 0.3 +
            conditions.bollinger_position * 0.3 +
            (conditions.vix_level - 20) / 20 * 0.4
        )
        
        engineered.extend([
            vol_trend, momentum_volume, vix_correlation, stress, trend_consistency,
            vol_low, vol_mid, vol_high, trend_momentum_alignment,
            volume_volatility_ratio, regime_stability, momentum_persistence,
            vol_trend_divergence, sentiment_composite
        ])
        
        return engineered
    
    def _calculate_regime_stability(self, conditions: MarketConditions) -> float:
        """Calculate how stable/consistent the current regime is."""
        regime = conditions.market_regime
        
        # Define regime-specific stability criteria
        if regime == MarketRegime.BULL_TRENDING:
            return min(
                conditions.trend_strength / 100,
                (100 - conditions.volatility) / 100
            )
        elif regime == MarketRegime.BEAR_TRENDING:
            return min(
                abs(conditions.trend_strength) / 100,
                (100 - conditions.volatility) / 100
            )
        elif regime == MarketRegime.SIDEWAYS_RANGE:
            return (100 - abs(conditions.trend_strength)) / 100
        elif regime == MarketRegime.HIGH_VOLATILITY:
            return conditions.volatility / 100
        elif regime == MarketRegime.LOW_VOLATILITY:
            return (100 - conditions.volatility) / 100
        else:
            return 0.5  # Neutral for other regimes
    
    def _validate_market_conditions(self, conditions: MarketConditions) -> None:
        """Validate market conditions for feature extraction."""
        # Check for NaN or infinite values
        values_to_check = [
            conditions.volatility, conditions.trend_strength,
            conditions.volume_ratio, conditions.price_momentum,
            conditions.vix_level, conditions.correlation_spy,
            conditions.rsi, conditions.bollinger_position,
            conditions.atr_normalized
        ]
        
        for value in values_to_check:
            if np.isnan(value) or np.isinf(value):
                raise InvalidMarketDataError(f"Invalid market data: NaN or infinite value {value}")
        
        # Additional range checks beyond dataclass validation
        if conditions.volume_ratio > 10:
            logger.warning(f"Unusually high volume ratio: {conditions.volume_ratio}")
        
        if abs(conditions.price_momentum) > 50:
            logger.warning(f"Extreme price momentum: {conditions.price_momentum}")
    
    def _validate_feature_vector(self, features: np.ndarray) -> None:
        """Validate extracted feature vector."""
        if np.any(np.isnan(features)):
            raise FeatureExtractionError("Feature vector contains NaN values")
        
        if np.any(np.isinf(features)):
            raise FeatureExtractionError("Feature vector contains infinite values")
        
        if len(features) != len(self._feature_names):
            raise FeatureExtractionError(
                f"Feature vector length mismatch: expected {len(self._feature_names)}, got {len(features)}"
            )
    
    def _get_cache_key(self, conditions: MarketConditions) -> str:
        """Generate cache key for market conditions."""
        # Create a hash-like key from market conditions
        key_parts = [
            f"{conditions.volatility:.2f}",
            f"{conditions.trend_strength:.2f}",
            f"{conditions.volume_ratio:.2f}",
            f"{conditions.price_momentum:.2f}",
            conditions.market_regime.value,
            f"{conditions.vix_level:.2f}",
            f"{conditions.correlation_spy:.3f}",
            f"{conditions.rsi:.1f}",
            f"{conditions.bollinger_position:.3f}",
            f"{conditions.atr_normalized:.4f}"
        ]
        return "|".join(key_parts)
    
    def _update_feature_stats(self, features: np.ndarray) -> None:
        """Update feature statistics for monitoring."""
        if "sample_count" not in self._feature_stats:
            self._feature_stats["feature_means"] = features.copy()
            self._feature_stats["feature_vars"] = np.zeros_like(features)
            self._feature_stats["sample_count"] = 1
        else:
            # Running average and variance calculation
            n = self._feature_stats["sample_count"]
            old_mean = self._feature_stats["feature_means"]
            
            # Update mean
            new_mean = old_mean + (features - old_mean) / (n + 1)
            
            # Update variance (Welford's algorithm)
            self._feature_stats["feature_vars"] += (features - old_mean) * (features - new_mean)
            
            self._feature_stats["feature_means"] = new_mean
            self._feature_stats["sample_count"] = n + 1
    
    def _calculate_feature_statistics(self, feature_matrix: np.ndarray) -> None:
        """Calculate comprehensive feature statistics after fitting."""
        self._feature_stats.update({
            "n_features": feature_matrix.shape[1],
            "n_samples": feature_matrix.shape[0],
            "feature_means": np.mean(feature_matrix, axis=0),
            "feature_stds": np.std(feature_matrix, axis=0),
            "feature_mins": np.min(feature_matrix, axis=0),
            "feature_maxs": np.max(feature_matrix, axis=0),
            "correlation_matrix": np.corrcoef(feature_matrix.T),
            "condition_number": np.linalg.cond(feature_matrix)
        })
        
        # Identify potential issues
        issues = []
        
        # Check for zero variance features
        zero_var_features = np.where(self._feature_stats["feature_stds"] < 1e-10)[0]
        if len(zero_var_features) > 0:
            issues.append(f"{len(zero_var_features)} features with zero variance")
        
        # Check for high correlation
        corr_matrix = self._feature_stats["correlation_matrix"]
        high_corr_pairs = np.where((np.abs(corr_matrix) > 0.95) & (corr_matrix != 1))
        if len(high_corr_pairs[0]) > 0:
            issues.append(f"{len(high_corr_pairs[0])} highly correlated feature pairs")
        
        # Check condition number
        if self._feature_stats["condition_number"] > 1000:
            issues.append(f"High condition number: {self._feature_stats['condition_number']:.2f}")
        
        if issues:
            logger.warning(f"Feature matrix issues detected: {', '.join(issues)}")
        
        self._feature_stats["issues"] = issues