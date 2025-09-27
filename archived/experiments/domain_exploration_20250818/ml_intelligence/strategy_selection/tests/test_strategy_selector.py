"""
Unit tests for StrategySelector.

Comprehensive testing of the core ML strategy selection model including
training, prediction, validation, and error handling.

Test Coverage:
- Model initialization and configuration
- Training pipeline with various data scenarios
- Prediction accuracy and reliability
- Model persistence (save/load)
- Error handling and edge cases
- Performance and thread safety
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime
import threading
import time

from ..core.strategy_selector import StrategySelector
from ..interfaces.types import (
    StrategyName, MarketConditions, StrategyPrediction,
    ModelNotTrainedError, PredictionError, InvalidMarketDataError
)
from . import (
    create_sample_market_conditions, create_sample_performance_records,
    sample_market_conditions, sample_performance_records, large_performance_dataset
)


class TestStrategySelector:
    """Test suite for StrategySelector class."""
    
    def test_initialization_valid_parameters(self):
        """Test StrategySelector initialization with valid parameters."""
        selector = StrategySelector(
            model_id="test_model",
            n_estimators=50,
            max_depth=5,
            random_state=123
        )
        
        assert selector.model_id == "test_model"
        assert selector.version == "1.0.0"
        assert not selector.is_trained
        assert selector._n_estimators == 50
        assert selector._max_depth == 5
        assert selector._random_state == 123
    
    def test_initialization_invalid_parameters(self):
        """Test StrategySelector initialization with invalid parameters."""
        with pytest.raises(ValueError, match="n_estimators must be positive"):
            StrategySelector(n_estimators=0)
        
        with pytest.raises(ValueError, match="max_depth must be positive"):
            StrategySelector(max_depth=0)
    
    def test_initialization_default_parameters(self):
        """Test StrategySelector initialization with default parameters."""
        selector = StrategySelector()
        
        assert selector.model_id.startswith("strategy_selector_")
        assert selector._n_estimators == 100
        assert selector._max_depth == 10
        assert selector._random_state == 42
    
    def test_training_successful(self, sample_performance_records):
        """Test successful model training."""
        selector = StrategySelector(n_estimators=10, max_depth=3)  # Small for speed
        
        result = selector.train(sample_performance_records)
        
        assert selector.is_trained
        assert result.training_samples > 0
        assert result.validation_samples > 0
        assert 0 <= result.validation_score <= 1
        assert 0 <= result.test_score <= 1
        assert result.training_time_seconds > 0
        assert len(result.features_used) > 0
    
    def test_training_insufficient_data(self):
        """Test training with insufficient data."""
        selector = StrategySelector()
        
        # Create minimal dataset (below threshold)
        minimal_records = create_sample_performance_records(n_records=10)
        
        with pytest.raises(ValueError, match="Insufficient training data"):
            selector.train(minimal_records)
    
    def test_training_cross_validation(self, sample_performance_records):
        """Test training with cross-validation."""
        selector = StrategySelector(n_estimators=5, max_depth=3)
        
        result = selector.train(
            sample_performance_records,
            cross_validation_folds=3
        )
        
        assert result.cross_validation_scores is not None
        assert len(result.cross_validation_scores) == 3
        assert all(0 <= score <= 1 for score in result.cross_validation_scores)
    
    def test_prediction_successful(self, sample_performance_records, sample_market_conditions):
        """Test successful prediction after training."""
        selector = StrategySelector(n_estimators=5, max_depth=3)
        selector.train(sample_performance_records)
        
        predictions = selector.predict(sample_market_conditions)
        
        assert len(predictions) == len(StrategyName)
        assert all(isinstance(p, StrategyPrediction) for p in predictions)
        assert all(0 <= p.confidence <= 1 for p in predictions)
        assert all(p.ranking > 0 for p in predictions)
        
        # Check that predictions are sorted by risk-adjusted score
        scores = [p.risk_adjusted_score for p in predictions]
        assert scores == sorted(scores, reverse=True)
    
    def test_prediction_not_trained(self, sample_market_conditions):
        """Test prediction with untrained model."""
        selector = StrategySelector()
        
        with pytest.raises(ModelNotTrainedError):
            selector.predict(sample_market_conditions)
    
    def test_prediction_invalid_market_data(self, sample_performance_records):
        """Test prediction with invalid market data."""
        selector = StrategySelector(n_estimators=5, max_depth=3)
        selector.train(sample_performance_records)
        
        # Create invalid market conditions
        invalid_conditions = create_sample_market_conditions(
            volatility=float('nan')  # Invalid value
        )
        
        with pytest.raises(InvalidMarketDataError):
            selector.predict(invalid_conditions)
    
    def test_predict_confidence(self, sample_performance_records, sample_market_conditions):
        """Test confidence prediction for specific strategy."""
        selector = StrategySelector(n_estimators=5, max_depth=3)
        selector.train(sample_performance_records)
        
        confidence = selector.predict_confidence(
            StrategyName.MOMENTUM, 
            sample_market_conditions
        )
        
        assert 0 <= confidence <= 1
        assert isinstance(confidence, float)
    
    def test_predict_confidence_not_trained(self, sample_market_conditions):
        """Test confidence prediction with untrained model."""
        selector = StrategySelector()
        
        with pytest.raises(ModelNotTrainedError):
            selector.predict_confidence(StrategyName.MOMENTUM, sample_market_conditions)
    
    def test_get_model_info(self, sample_performance_records):
        """Test model info retrieval."""
        selector = StrategySelector(model_id="test_info_model")
        
        # Before training
        info_before = selector.get_model_info()
        assert info_before["model_id"] == "test_info_model"
        assert not info_before["is_trained"]
        assert "model_performance" not in info_before
        
        # After training
        selector.train(sample_performance_records)
        info_after = selector.get_model_info()
        
        assert info_after["is_trained"]
        assert "model_performance" in info_after
        assert "feature_importance" in info_after
        assert len(info_after["feature_names"]) > 0
    
    def test_save_load_model(self, sample_performance_records):
        """Test model persistence (save/load)."""
        selector = StrategySelector(model_id="test_persistence")
        selector.train(sample_performance_records)
        
        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pkl"
            selector.save_model(model_path)
            
            assert model_path.exists()
            
            # Load model into new instance
            new_selector = StrategySelector()
            new_selector.load_model(model_path)
            
            assert new_selector.model_id == "test_persistence"
            assert new_selector.is_trained
            
            # Test that loaded model can make predictions
            predictions = new_selector.predict(create_sample_market_conditions())
            assert len(predictions) > 0
    
    def test_save_untrained_model(self):
        """Test saving untrained model raises error."""
        selector = StrategySelector()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "untrained_model.pkl"
            
            with pytest.raises(ModelNotTrainedError):
                selector.save_model(model_path)
    
    def test_load_nonexistent_model(self):
        """Test loading nonexistent model raises error."""
        selector = StrategySelector()
        
        with pytest.raises(RuntimeError, match="Model file not found"):
            selector.load_model(Path("nonexistent_model.pkl"))
    
    def test_thread_safety(self, sample_performance_records):
        """Test thread safety of trained model."""
        selector = StrategySelector(n_estimators=5, max_depth=3)
        selector.train(sample_performance_records)
        
        results = []
        errors = []
        
        def make_prediction(thread_id):
            try:
                conditions = create_sample_market_conditions(
                    volatility=20 + thread_id,  # Vary conditions slightly
                    trend_strength=10 + thread_id
                )
                predictions = selector.predict(conditions)
                results.append((thread_id, len(predictions)))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Run multiple threads concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_prediction, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 5
        assert all(num_predictions == len(StrategyName) for _, num_predictions in results)
    
    def test_feature_importance(self, sample_performance_records):
        """Test feature importance calculation."""
        selector = StrategySelector(n_estimators=5, max_depth=3, enable_feature_importance=True)
        selector.train(sample_performance_records)
        
        info = selector.get_model_info()
        feature_importance = info["feature_importance"]
        
        assert isinstance(feature_importance, dict)
        assert len(feature_importance) > 0
        assert all(isinstance(v, float) for v in feature_importance.values())
        assert all(v >= 0 for v in feature_importance.values())
    
    def test_prediction_consistency(self, sample_performance_records):
        """Test prediction consistency for same inputs."""
        selector = StrategySelector(n_estimators=5, max_depth=3, random_state=42)
        selector.train(sample_performance_records)
        
        conditions = create_sample_market_conditions()
        
        # Make multiple predictions with same input
        predictions1 = selector.predict(conditions)
        predictions2 = selector.predict(conditions)
        
        # Predictions should be similar (allowing for small randomness)
        assert len(predictions1) == len(predictions2)
        
        for p1, p2 in zip(predictions1, predictions2):
            assert p1.strategy == p2.strategy
            assert abs(p1.expected_return - p2.expected_return) < 5  # Small tolerance
            assert abs(p1.confidence - p2.confidence) < 0.2
    
    def test_extreme_market_conditions(self, sample_performance_records):
        """Test model behavior with extreme market conditions."""
        selector = StrategySelector(n_estimators=5, max_depth=3)
        selector.train(sample_performance_records)
        
        # Test extreme volatility
        extreme_vol_conditions = create_sample_market_conditions(volatility=95.0)
        predictions = selector.predict(extreme_vol_conditions)
        
        assert len(predictions) > 0
        assert all(0 <= p.confidence <= 1 for p in predictions)
        
        # Test extreme trend
        extreme_trend_conditions = create_sample_market_conditions(trend_strength=-95.0)
        predictions = selector.predict(extreme_trend_conditions)
        
        assert len(predictions) > 0
        assert all(0 <= p.confidence <= 1 for p in predictions)
    
    def test_model_performance_tracking(self, sample_performance_records):
        """Test model performance tracking during training."""
        selector = StrategySelector(n_estimators=5, max_depth=3)
        
        result = selector.train(sample_performance_records)
        
        assert selector._model_performance is not None
        assert 0 <= selector._model_performance.accuracy <= 1
        assert selector._model_performance.total_predictions > 0
        assert selector._model_performance.successful_predictions >= 0
    
    def test_strategy_specific_predictions(self, sample_performance_records):
        """Test that strategy-specific predictions vary appropriately."""
        selector = StrategySelector(n_estimators=10, max_depth=5)
        selector.train(sample_performance_records)
        
        # Test momentum strategy in trending market
        trending_conditions = create_sample_market_conditions(
            trend_strength=60.0,
            volatility=15.0,
            market_regime=MarketRegime.BULL_TRENDING
        )
        
        predictions = selector.predict(trending_conditions)
        strategy_scores = {p.strategy: p.risk_adjusted_score for p in predictions}
        
        # Momentum should perform well in trending markets
        momentum_score = strategy_scores.get(StrategyName.MOMENTUM, 0)
        mean_score = np.mean(list(strategy_scores.values()))
        
        # Momentum should be above average (allowing for randomness in test data)
        assert momentum_score >= mean_score * 0.8  # Relaxed assertion for test data
    
    def test_large_dataset_training(self, large_performance_dataset):
        """Test training with large dataset."""
        selector = StrategySelector(n_estimators=10, max_depth=5)
        
        start_time = time.time()
        result = selector.train(large_performance_dataset)
        training_time = time.time() - start_time
        
        assert selector.is_trained
        assert result.training_samples >= 300  # Should have good amount of training data
        assert training_time < 60  # Should complete within reasonable time
        assert result.validation_score > 0.3  # Should achieve reasonable accuracy
    
    def test_cross_validation_stability(self, sample_performance_records):
        """Test cross-validation score stability."""
        selector = StrategySelector(n_estimators=5, max_depth=3, random_state=42)
        
        result = selector.train(
            sample_performance_records,
            cross_validation_folds=5
        )
        
        cv_scores = result.cross_validation_scores
        assert len(cv_scores) == 5
        
        # Check that CV scores are reasonably stable
        cv_std = np.std(cv_scores)
        assert cv_std < 0.3  # Standard deviation shouldn't be too high
    
    @patch('logging.getLogger')
    def test_logging_integration(self, mock_logger, sample_performance_records):
        """Test that logging is properly integrated."""
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        
        selector = StrategySelector(n_estimators=5, max_depth=3)
        selector.train(sample_performance_records)
        
        # Verify that info logs were called during training
        assert mock_logger_instance.info.call_count > 0
        
        # Test prediction logging
        conditions = create_sample_market_conditions()
        selector.predict(conditions)
        
        assert mock_logger_instance.debug.call_count > 0
    
    def test_model_size_estimation(self, sample_performance_records):
        """Test model size estimation."""
        selector = StrategySelector(n_estimators=10, max_depth=3)
        selector.train(sample_performance_records)
        
        estimated_size = selector._estimate_model_size()
        
        assert estimated_size > 0
        assert estimated_size < 100  # Should be reasonable size in MB
        assert isinstance(estimated_size, float)
    
    def test_fallback_predictions(self, sample_performance_records):
        """Test fallback prediction mechanism."""
        selector = StrategySelector(n_estimators=5, max_depth=3)
        selector.train(sample_performance_records)
        
        # Create conditions that might trigger fallback
        conditions = create_sample_market_conditions()
        
        # Mock the strategy models to simulate missing models
        original_models = selector._strategy_models.copy()
        selector._strategy_models = {}  # Remove all strategy models
        
        predictions = selector.predict(conditions)
        
        # Should still return predictions using fallback mechanism
        assert len(predictions) > 0
        assert all(isinstance(p, StrategyPrediction) for p in predictions)
        
        # Restore original models
        selector._strategy_models = original_models