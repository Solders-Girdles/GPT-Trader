"""
Unit tests for FeatureExtractor.

Comprehensive testing of feature extraction including normalization,
engineering, validation, and performance optimization.

Test Coverage:
- Feature extraction accuracy and consistency
- Normalization and scaling
- Feature engineering capabilities
- Caching and performance optimization
- Error handling and validation
- Thread safety and concurrency
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import threading
import time

from ..core.feature_extractor import FeatureExtractor
from ..interfaces.types import (
    MarketConditions, MarketRegime, FeatureExtractionError, 
    InvalidMarketDataError
)
from . import (
    create_sample_market_conditions, create_sample_performance_records,
    sample_market_conditions, extreme_market_conditions
)


class TestFeatureExtractor:
    """Test suite for FeatureExtractor class."""
    
    def test_initialization_default_parameters(self):
        """Test FeatureExtractor initialization with default parameters."""
        extractor = FeatureExtractor()
        
        assert extractor.enable_feature_engineering
        assert not extractor.enable_feature_selection
        assert extractor.feature_selection_k == 20
        assert extractor.normalization_method == "standard"
        assert extractor.cache_features
        assert extractor.feature_validation
        assert not extractor._is_fitted
    
    def test_initialization_custom_parameters(self):
        """Test FeatureExtractor initialization with custom parameters."""
        extractor = FeatureExtractor(
            enable_feature_engineering=False,
            enable_feature_selection=True,
            feature_selection_k=15,
            normalization_method="robust",
            cache_features=False,
            feature_validation=False
        )
        
        assert not extractor.enable_feature_engineering
        assert extractor.enable_feature_selection
        assert extractor.feature_selection_k == 15
        assert extractor.normalization_method == "robust"
        assert not extractor.cache_features
        assert not extractor.feature_validation
    
    def test_initialization_invalid_parameters(self):
        """Test FeatureExtractor initialization with invalid parameters."""
        with pytest.raises(ValueError, match="feature_selection_k must be positive"):
            FeatureExtractor(feature_selection_k=0)
        
        with pytest.raises(ValueError, match="Invalid normalization method"):
            FeatureExtractor(normalization_method="invalid")
    
    def test_extract_features_basic(self, sample_market_conditions):
        """Test basic feature extraction."""
        extractor = FeatureExtractor(enable_feature_engineering=False)
        
        features = extractor.extract_features(sample_market_conditions)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
        
        # Check that features are in reasonable ranges
        assert np.all(features >= -10)  # No extremely negative values
        assert np.all(features <= 10)   # No extremely positive values
    
    def test_extract_features_with_engineering(self, sample_market_conditions):
        """Test feature extraction with engineering enabled."""
        extractor_basic = FeatureExtractor(enable_feature_engineering=False)
        extractor_engineered = FeatureExtractor(enable_feature_engineering=True)
        
        features_basic = extractor_basic.extract_features(sample_market_conditions)
        features_engineered = extractor_engineered.extract_features(sample_market_conditions)
        
        # Engineered features should have more features
        assert len(features_engineered) > len(features_basic)
        
        # Basic features should be the same (first N features)
        np.testing.assert_array_almost_equal(
            features_basic[:len(features_basic)], 
            features_engineered[:len(features_basic)],
            decimal=6
        )
    
    def test_get_feature_names(self):
        """Test feature name retrieval."""
        extractor = FeatureExtractor(enable_feature_engineering=True)
        
        feature_names = extractor.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)
        
        # Check for expected base features
        expected_base_features = [
            "volatility_norm", "trend_strength_norm", "volume_ratio_norm"
        ]
        for expected_feature in expected_base_features:
            assert expected_feature in feature_names
        
        # Check for regime features
        regime_features = [name for name in feature_names if name.startswith("regime_")]
        assert len(regime_features) == len(MarketRegime)
    
    def test_feature_consistency(self, sample_market_conditions):
        """Test feature extraction consistency for same input."""
        extractor = FeatureExtractor()
        
        features1 = extractor.extract_features(sample_market_conditions)
        features2 = extractor.extract_features(sample_market_conditions)
        
        np.testing.assert_array_equal(features1, features2)
    
    def test_feature_caching(self, sample_market_conditions):
        """Test feature caching functionality."""
        extractor = FeatureExtractor(cache_features=True)
        
        # First extraction (cache miss)
        features1 = extractor.extract_features(sample_market_conditions)
        stats1 = extractor.get_performance_stats()
        
        # Second extraction (cache hit)
        features2 = extractor.extract_features(sample_market_conditions)
        stats2 = extractor.get_performance_stats()
        
        np.testing.assert_array_equal(features1, features2)
        assert stats2["cache_hit_rate"] > 0
        assert stats1["cache_hit_rate"] == 0
    
    def test_cache_disabled(self, sample_market_conditions):
        """Test feature extraction with caching disabled."""
        extractor = FeatureExtractor(cache_features=False)
        
        extractor.extract_features(sample_market_conditions)
        extractor.extract_features(sample_market_conditions)
        
        stats = extractor.get_performance_stats()
        assert "cache_hit_rate" not in stats or stats["cache_hit_rate"] == 0
    
    def test_clear_cache(self, sample_market_conditions):
        """Test cache clearing functionality."""
        extractor = FeatureExtractor(cache_features=True)
        
        # Add something to cache
        extractor.extract_features(sample_market_conditions)
        stats_before = extractor.get_performance_stats()
        assert stats_before["cache_size"] > 0
        
        # Clear cache
        extractor.clear_cache()
        stats_after = extractor.get_performance_stats()
        assert stats_after["cache_size"] == 0
    
    def test_fit_functionality(self):
        """Test fitting normalization and feature selection."""
        extractor = FeatureExtractor(
            enable_feature_selection=True,
            feature_selection_k=10,
            normalization_method="standard"
        )
        
        # Create sample data
        conditions_list = [create_sample_market_conditions() for _ in range(50)]
        target_values = np.random.randn(50)
        
        # Fit extractor
        extractor.fit(conditions_list, target_values)
        
        assert extractor._is_fitted
        assert extractor._scaler is not None
        assert extractor._feature_selector is not None
        
        # Test that feature extraction works after fitting
        features = extractor.extract_features(conditions_list[0])
        assert len(features) <= 10  # Should be limited by feature selection
    
    def test_fit_insufficient_data(self):
        """Test fitting with insufficient data."""
        extractor = FeatureExtractor()
        
        # Too few samples
        conditions_list = [create_sample_market_conditions() for _ in range(5)]
        
        with pytest.raises(ValueError, match="Insufficient data for fitting"):
            extractor.fit(conditions_list)
    
    def test_fit_without_target_values(self):
        """Test fitting without target values (no feature selection)."""
        extractor = FeatureExtractor(enable_feature_selection=False)
        
        conditions_list = [create_sample_market_conditions() for _ in range(20)]
        
        # Should work without target values when feature selection is disabled
        extractor.fit(conditions_list)
        
        assert extractor._is_fitted
        assert extractor._scaler is not None
        assert extractor._feature_selector is None
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        extractor = FeatureExtractor(enable_feature_selection=True)
        
        conditions_list = [create_sample_market_conditions() for _ in range(50)]
        target_values = np.random.randn(50)
        
        extractor.fit(conditions_list, target_values)
        
        importance = extractor.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
        assert all(isinstance(v, float) for v in importance.values())
        assert all(v >= 0 for v in importance.values())
    
    def test_feature_importance_not_fitted(self):
        """Test feature importance when not fitted."""
        extractor = FeatureExtractor()
        
        importance = extractor.get_feature_importance()
        assert importance == {}
    
    def test_performance_stats(self, sample_market_conditions):
        """Test performance statistics collection."""
        extractor = FeatureExtractor()
        
        # Make some extractions
        for _ in range(5):
            extractor.extract_features(sample_market_conditions)
        
        stats = extractor.get_performance_stats()
        
        assert "total_extractions" in stats
        assert stats["total_extractions"] == 5
        assert "avg_extraction_time_ms" in stats
        assert stats["avg_extraction_time_ms"] > 0
        assert "cache_hit_rate" in stats
    
    def test_invalid_market_data_validation(self):
        """Test validation of invalid market data."""
        extractor = FeatureExtractor(feature_validation=True)
        
        # Create invalid conditions with NaN
        invalid_conditions = create_sample_market_conditions(volatility=float('nan'))
        
        with pytest.raises(InvalidMarketDataError):
            extractor.extract_features(invalid_conditions)
    
    def test_validation_disabled(self):
        """Test feature extraction with validation disabled."""
        extractor = FeatureExtractor(feature_validation=False)
        
        # Create conditions that would normally fail validation
        conditions = create_sample_market_conditions(volatility=float('nan'))
        
        # Should not raise validation error, but might produce NaN features
        features = extractor.extract_features(conditions)
        
        # Features might contain NaN, but extraction should complete
        assert isinstance(features, np.ndarray)
    
    def test_thread_safety(self, sample_market_conditions):
        """Test thread safety of feature extraction."""
        extractor = FeatureExtractor(cache_features=True)
        
        results = []
        errors = []
        
        def extract_features(thread_id):
            try:
                conditions = create_sample_market_conditions(
                    volatility=20 + thread_id,  # Vary conditions slightly
                    trend_strength=10 + thread_id
                )
                features = extractor.extract_features(conditions)
                results.append((thread_id, len(features)))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Run multiple threads concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=extract_features, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 5
        
        # All extractions should produce same number of features
        feature_counts = [count for _, count in results]
        assert len(set(feature_counts)) == 1  # All counts should be the same
    
    def test_extreme_conditions(self, extreme_market_conditions):
        """Test feature extraction with extreme market conditions."""
        extractor = FeatureExtractor()
        
        for conditions in extreme_market_conditions:
            features = extractor.extract_features(conditions)
            
            assert isinstance(features, np.ndarray)
            assert len(features) > 0
            assert not np.any(np.isnan(features))
            assert not np.any(np.isinf(features))
    
    def test_regime_feature_encoding(self):
        """Test proper one-hot encoding of market regimes."""
        extractor = FeatureExtractor(enable_feature_engineering=False)
        
        # Test each regime
        for regime in MarketRegime:
            conditions = create_sample_market_conditions(market_regime=regime)
            features = extractor.extract_features(conditions)
            
            # Extract regime features (should be one-hot encoded)
            feature_names = extractor.get_feature_names()
            regime_start_idx = len([n for n in feature_names if not n.startswith("regime_")])
            regime_features = features[regime_start_idx:regime_start_idx + len(MarketRegime)]
            
            # Exactly one regime feature should be 1.0, others should be 0.0
            assert np.sum(regime_features) == 1.0
            assert np.max(regime_features) == 1.0
            assert np.min(regime_features) == 0.0
    
    def test_feature_normalization_fitting(self):
        """Test feature normalization after fitting."""
        extractor = FeatureExtractor(normalization_method="standard")
        
        # Create varied data for fitting
        conditions_list = []
        for i in range(50):
            conditions = create_sample_market_conditions(
                volatility=np.random.uniform(5, 50),
                trend_strength=np.random.uniform(-80, 80)
            )
            conditions_list.append(conditions)
        
        # Fit the extractor
        extractor.fit(conditions_list)
        
        # Extract features from fitted extractor
        test_conditions = create_sample_market_conditions()
        features = extractor.extract_features(test_conditions)
        
        # Features should be roughly normalized (mean ~0, std ~1)
        # (This is approximate due to single sample)
        assert np.all(np.abs(features) < 5)  # Should be within reasonable bounds
    
    def test_feature_statistics_calculation(self):
        """Test feature statistics calculation during fitting."""
        extractor = FeatureExtractor()
        
        conditions_list = [create_sample_market_conditions() for _ in range(30)]
        extractor.fit(conditions_list)
        
        stats = extractor.get_performance_stats()
        feature_stats = stats.get("feature_stats", {})
        
        assert "n_features" in feature_stats
        assert "n_samples" in feature_stats
        assert feature_stats["n_samples"] == 30
        assert feature_stats["n_features"] > 0
    
    def test_robust_normalization(self):
        """Test robust normalization method."""
        extractor = FeatureExtractor(normalization_method="robust")
        
        conditions_list = [create_sample_market_conditions() for _ in range(30)]
        extractor.fit(conditions_list)
        
        # Should work without errors
        features = extractor.extract_features(conditions_list[0])
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
    
    def test_no_normalization(self):
        """Test extraction without normalization."""
        extractor = FeatureExtractor(normalization_method="none")
        
        conditions_list = [create_sample_market_conditions() for _ in range(20)]
        extractor.fit(conditions_list)
        
        features = extractor.extract_features(conditions_list[0])
        
        # Features should not be normalized (values in original ranges)
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
    
    def test_engineered_features_content(self, sample_market_conditions):
        """Test content of engineered features."""
        extractor = FeatureExtractor(enable_feature_engineering=True)
        
        features = extractor.extract_features(sample_market_conditions)
        feature_names = extractor.get_feature_names()
        
        # Check for specific engineered features
        engineered_feature_names = [
            "vol_trend_interaction", "momentum_volume_interaction",
            "market_stress", "trend_consistency"
        ]
        
        for feature_name in engineered_feature_names:
            assert feature_name in feature_names
    
    def test_regime_stability_calculation(self):
        """Test regime stability indicator calculation."""
        extractor = FeatureExtractor(enable_feature_engineering=True)
        
        # Test bull trending regime with consistent conditions
        bull_conditions = create_sample_market_conditions(
            market_regime=MarketRegime.BULL_TRENDING,
            trend_strength=70.0,
            volatility=15.0
        )
        
        features = extractor.extract_features(bull_conditions)
        feature_names = extractor.get_feature_names()
        
        # Find regime stability feature
        stability_idx = feature_names.index("regime_stability_indicator")
        stability_value = features[stability_idx]
        
        # Should indicate high stability for consistent bull trend
        assert 0 <= stability_value <= 1
        assert stability_value > 0.5  # Should be relatively high
    
    @patch('logging.getLogger')
    def test_logging_integration(self, mock_logger, sample_market_conditions):
        """Test logging integration."""
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        
        extractor = FeatureExtractor()
        extractor.extract_features(sample_market_conditions)
        
        # Should have debug logging for feature extraction
        assert mock_logger_instance.debug.call_count > 0
    
    def test_feature_vector_validation(self):
        """Test feature vector validation."""
        extractor = FeatureExtractor(feature_validation=True)
        
        # Mock the extraction to produce invalid features
        with patch.object(extractor, '_extract_base_features') as mock_extract:
            mock_extract.return_value = [float('nan'), 1.0, 2.0]
            
            conditions = create_sample_market_conditions()
            
            with pytest.raises(FeatureExtractionError, match="Feature vector contains NaN"):
                extractor.extract_features(conditions)
    
    def test_performance_under_load(self, sample_market_conditions):
        """Test feature extraction performance under load."""
        extractor = FeatureExtractor(cache_features=True)
        
        # Extract features many times
        start_time = time.time()
        for _ in range(100):
            extractor.extract_features(sample_market_conditions)
        total_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert total_time < 5.0  # 5 seconds for 100 extractions
        
        stats = extractor.get_performance_stats()
        assert stats["cache_hit_rate"] > 0.9  # Should have high cache hit rate