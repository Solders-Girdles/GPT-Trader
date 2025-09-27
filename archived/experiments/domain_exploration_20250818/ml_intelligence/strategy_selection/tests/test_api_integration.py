"""
Integration tests for ML Strategy Selection API.

Comprehensive testing of the high-level API functions that integrate
multiple components together. Tests realistic workflows and end-to-end
functionality.

Test Coverage:
- Complete training pipeline integration
- Real-time recommendation workflows
- Model evaluation and validation
- Batch prediction processing
- Error handling across components
- Performance and scalability
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from ..api import (
    train_strategy_model, get_strategy_recommendations,
    evaluate_model_performance, create_prediction_request,
    batch_predict_strategies
)
from ..interfaces.types import (
    StrategyName, MarketConditions, PredictionRequest,
    StrategySelectionError, ModelNotTrainedError
)
from . import (
    create_sample_market_conditions, create_sample_performance_records,
    sample_market_conditions, large_performance_dataset
)


class TestAPIIntegration:
    """Integration test suite for strategy selection API."""
    
    def test_complete_training_pipeline(self, sample_performance_records):
        """Test complete model training pipeline."""
        # Train model with all components
        model, extractor, scorer = train_strategy_model(
            training_records=sample_performance_records,
            validation_split=0.2,
            test_split=0.1,
            enable_feature_engineering=True,
            enable_confidence_scoring=True
        )
        
        # Verify all components are properly trained
        assert model is not None
        assert model.is_trained
        assert extractor is not None
        assert extractor._is_fitted
        assert scorer is not None
        assert scorer._is_fitted
        
        # Verify model info
        model_info = model.get_model_info()
        assert model_info["is_trained"]
        assert len(model_info["feature_names"]) > 0
        
        # Verify feature extractor
        feature_names = extractor.get_feature_names()
        assert len(feature_names) > 10  # Should have many features
        
        # Verify confidence scorer
        scorer_stats = scorer.get_performance_stats()
        assert scorer_stats["is_fitted"]
    
    def test_training_insufficient_data(self):
        """Test training with insufficient data."""
        minimal_records = create_sample_performance_records(n_records=50)
        
        with pytest.raises(ValueError, match="Insufficient training data"):
            train_strategy_model(minimal_records)
    
    def test_training_invalid_splits(self, sample_performance_records):
        """Test training with invalid split parameters."""
        with pytest.raises(ValueError, match="Invalid validation split"):
            train_strategy_model(
                sample_performance_records,
                validation_split=1.5
            )
        
        with pytest.raises(ValueError, match="sum to less than 1"):
            train_strategy_model(
                sample_performance_records,
                validation_split=0.6,
                test_split=0.5
            )
    
    def test_training_custom_config(self, sample_performance_records):
        """Test training with custom model configuration."""
        custom_config = {
            "n_estimators": 20,
            "max_depth": 5,
            "random_state": 123
        }
        
        model, extractor, scorer = train_strategy_model(
            training_records=sample_performance_records,
            model_config=custom_config
        )
        
        model_info = model.get_model_info()
        assert model_info["n_estimators"] == 20
        assert model_info["max_depth"] == 5
    
    def test_training_without_confidence_scoring(self, sample_performance_records):
        """Test training without confidence scoring."""
        model, extractor, scorer = train_strategy_model(
            training_records=sample_performance_records,
            enable_confidence_scoring=False
        )
        
        assert model is not None
        assert extractor is not None
        assert scorer is None
    
    def test_get_recommendations_basic(self, sample_performance_records, sample_market_conditions):
        """Test basic strategy recommendations."""
        model, extractor, scorer = train_strategy_model(sample_performance_records)
        
        recommendations = get_strategy_recommendations(
            model=model,
            feature_extractor=extractor,
            market_conditions=sample_market_conditions,
            top_n=3,
            min_confidence=0.3
        )
        
        assert len(recommendations) <= 3
        assert all(pred.confidence >= 0.3 for pred in recommendations)
        assert all(pred.ranking > 0 for pred in recommendations)
        
        # Check ranking order
        rankings = [pred.ranking for pred in recommendations]
        assert rankings == sorted(rankings)
    
    def test_get_recommendations_with_confidence_scorer(self, sample_performance_records, sample_market_conditions):
        """Test recommendations with confidence scorer enhancement."""
        model, extractor, scorer = train_strategy_model(
            sample_performance_records,
            enable_confidence_scoring=True
        )
        
        recommendations = get_strategy_recommendations(
            model=model,
            feature_extractor=extractor,
            market_conditions=sample_market_conditions,
            confidence_scorer=scorer
        )
        
        assert len(recommendations) > 0
        assert all(0 <= pred.confidence <= 1 for pred in recommendations)
        
        # With confidence scorer, should have enhanced confidence scores
        # (exact values depend on model, but should be reasonable)
        avg_confidence = np.mean([pred.confidence for pred in recommendations])
        assert 0.2 <= avg_confidence <= 0.9
    
    def test_get_recommendations_high_confidence_threshold(self, sample_performance_records, sample_market_conditions):
        """Test recommendations with high confidence threshold."""
        model, extractor, scorer = train_strategy_model(sample_performance_records)
        
        recommendations = get_strategy_recommendations(
            model=model,
            feature_extractor=extractor,
            market_conditions=sample_market_conditions,
            min_confidence=0.9  # Very high threshold
        )
        
        # Might return empty list or best prediction despite low confidence
        if recommendations:
            assert all(pred.confidence >= 0.5 for pred in recommendations)  # Should still return something reasonable
    
    def test_get_recommendations_untrained_model(self, sample_market_conditions):
        """Test recommendations with untrained model."""
        from ..core.strategy_selector import StrategySelector
        from ..core.feature_extractor import FeatureExtractor
        
        untrained_model = StrategySelector()
        extractor = FeatureExtractor()
        
        with pytest.raises(ModelNotTrainedError):
            get_strategy_recommendations(
                model=untrained_model,
                feature_extractor=extractor,
                market_conditions=sample_market_conditions
            )
    
    def test_evaluate_model_walk_forward(self, sample_performance_records):
        """Test model evaluation with walk-forward validation."""
        model, extractor, _ = train_strategy_model(sample_performance_records)
        
        evaluation = evaluate_model_performance(
            model=model,
            validation_records=sample_performance_records,
            feature_extractor=extractor,
            validation_type="walk_forward"
        )
        
        assert evaluation["validation_type"] == "walk_forward"
        assert 0 <= evaluation["accuracy"] <= 1
        assert 0 <= evaluation["f1_score"] <= 1
        assert "overall_score" in evaluation
        assert "is_reliable" in evaluation
        assert evaluation["validation_samples"] > 0
    
    def test_evaluate_model_cross_validation(self, sample_performance_records):
        """Test model evaluation with cross-validation."""
        model, extractor, _ = train_strategy_model(sample_performance_records)
        
        evaluation = evaluate_model_performance(
            model=model,
            validation_records=sample_performance_records,
            validation_type="cross_validation"
        )
        
        assert evaluation["validation_type"] == "cross_validation"
        assert 0 <= evaluation["accuracy"] <= 1
        assert evaluation["validation_periods"] == 5  # Default CV folds
    
    def test_evaluate_model_insufficient_data(self, sample_performance_records):
        """Test model evaluation with insufficient validation data."""
        model, extractor, _ = train_strategy_model(sample_performance_records)
        
        minimal_validation = create_sample_performance_records(n_records=20)
        
        with pytest.raises(ValueError, match="Insufficient validation data"):
            evaluate_model_performance(
                model=model,
                validation_records=minimal_validation
            )
    
    def test_create_prediction_request_valid(self, sample_market_conditions):
        """Test creation of valid prediction request."""
        request = create_prediction_request(
            symbol="AAPL",
            market_conditions=sample_market_conditions,
            top_n_strategies=5,
            min_confidence=0.6,
            risk_tolerance=1.5
        )
        
        assert isinstance(request, PredictionRequest)
        assert request.symbol == "AAPL"
        assert request.top_n_strategies == 5
        assert request.min_confidence == 0.6
        assert request.risk_tolerance == 1.5
        assert request.exclude_strategies == []
    
    def test_create_prediction_request_with_exclusions(self, sample_market_conditions):
        """Test prediction request with strategy exclusions."""
        exclude_strategies = [StrategyName.VOLATILITY, StrategyName.BREAKOUT]
        
        request = create_prediction_request(
            symbol="GOOGL",
            market_conditions=sample_market_conditions,
            exclude_strategies=exclude_strategies
        )
        
        assert request.exclude_strategies == exclude_strategies
    
    def test_create_prediction_request_invalid_symbol(self, sample_market_conditions):
        """Test prediction request with invalid symbol."""
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            create_prediction_request("", sample_market_conditions)
        
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            create_prediction_request(None, sample_market_conditions)
    
    def test_batch_predict_strategies(self, sample_performance_records):
        """Test batch prediction for multiple requests."""
        model, extractor, scorer = train_strategy_model(sample_performance_records)
        
        # Create multiple prediction requests
        requests = []
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            conditions = create_sample_market_conditions()
            request = create_prediction_request(symbol, conditions, top_n_strategies=2)
            requests.append(request)
        
        results = batch_predict_strategies(
            model=model,
            feature_extractor=extractor,
            requests=requests,
            confidence_scorer=scorer
        )
        
        assert isinstance(results, dict)
        assert len(results) == 3
        assert "AAPL" in results
        assert "GOOGL" in results
        assert "MSFT" in results
        
        # Each symbol should have predictions
        for symbol, predictions in results.items():
            assert len(predictions) <= 2  # top_n=2
            assert all(hasattr(pred, 'strategy') for pred in predictions)
    
    def test_batch_predict_empty_requests(self, sample_performance_records):
        """Test batch prediction with empty request list."""
        model, extractor, _ = train_strategy_model(sample_performance_records)
        
        results = batch_predict_strategies(
            model=model,
            feature_extractor=extractor,
            requests=[]
        )
        
        assert results == {}
    
    def test_batch_predict_with_exclusions(self, sample_performance_records):
        """Test batch prediction with strategy exclusions."""
        model, extractor, _ = train_strategy_model(sample_performance_records)
        
        # Create request with exclusions
        conditions = create_sample_market_conditions()
        request = create_prediction_request(
            symbol="AAPL",
            market_conditions=conditions,
            top_n_strategies=5,
            exclude_strategies=[StrategyName.VOLATILITY]
        )
        
        results = batch_predict_strategies(
            model=model,
            feature_extractor=extractor,
            requests=[request]
        )
        
        predictions = results["AAPL"]
        excluded_strategies = {pred.strategy for pred in predictions}
        assert StrategyName.VOLATILITY not in excluded_strategies
    
    def test_end_to_end_workflow(self, large_performance_dataset):
        """Test complete end-to-end workflow."""
        # 1. Train model
        model, extractor, scorer = train_strategy_model(
            training_records=large_performance_dataset,
            validation_split=0.2,
            enable_confidence_scoring=True
        )
        
        # 2. Evaluate model
        evaluation = evaluate_model_performance(
            model=model,
            validation_records=large_performance_dataset[-100:],  # Use recent data for validation
            feature_extractor=extractor,
            validation_type="cross_validation"
        )
        
        assert evaluation["overall_score"] > 0
        
        # 3. Get recommendations
        market_conditions = create_sample_market_conditions()
        recommendations = get_strategy_recommendations(
            model=model,
            feature_extractor=extractor,
            market_conditions=market_conditions,
            confidence_scorer=scorer
        )
        
        assert len(recommendations) > 0
        best_strategy = recommendations[0].strategy
        assert isinstance(best_strategy, StrategyName)
        
        # 4. Batch predictions
        requests = [
            create_prediction_request("AAPL", market_conditions),
            create_prediction_request("GOOGL", market_conditions),
        ]
        
        batch_results = batch_predict_strategies(
            model=model,
            feature_extractor=extractor,
            requests=requests,
            confidence_scorer=scorer
        )
        
        assert len(batch_results) == 2
        assert all(len(preds) > 0 for preds in batch_results.values())
    
    def test_model_persistence_integration(self, sample_performance_records, sample_market_conditions):
        """Test model persistence in integration workflow."""
        # Train model
        model, extractor, scorer = train_strategy_model(sample_performance_records)
        
        # Get initial predictions
        initial_predictions = get_strategy_recommendations(
            model=model,
            feature_extractor=extractor,
            market_conditions=sample_market_conditions
        )
        
        # Save and reload model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pkl"
            model.save_model(model_path)
            
            # Create new model and load
            from ..core.strategy_selector import StrategySelector
            new_model = StrategySelector()
            new_model.load_model(model_path)
            
            # Get predictions from loaded model
            loaded_predictions = get_strategy_recommendations(
                model=new_model,
                feature_extractor=extractor,
                market_conditions=sample_market_conditions
            )
            
            # Predictions should be similar (allowing for small differences)
            assert len(loaded_predictions) == len(initial_predictions)
            
            initial_strategies = [p.strategy for p in initial_predictions]
            loaded_strategies = [p.strategy for p in loaded_predictions]
            
            # At least top strategies should match
            assert initial_strategies[0] == loaded_strategies[0]
    
    def test_performance_under_load(self, sample_performance_records):
        """Test API performance under load."""
        import time
        
        model, extractor, scorer = train_strategy_model(sample_performance_records)
        
        # Test multiple rapid predictions
        conditions = create_sample_market_conditions()
        
        start_time = time.time()
        for _ in range(50):
            recommendations = get_strategy_recommendations(
                model=model,
                feature_extractor=extractor,
                market_conditions=conditions,
                confidence_scorer=scorer
            )
            assert len(recommendations) > 0
        
        total_time = time.time() - start_time
        
        # Should complete within reasonable time (50 predictions in <10 seconds)
        assert total_time < 10.0
        
        # Average time per prediction should be reasonable
        avg_time_per_prediction = total_time / 50
        assert avg_time_per_prediction < 0.2  # Less than 200ms per prediction
    
    def test_error_handling_chain(self, sample_performance_records):
        """Test error handling across component chain."""
        model, extractor, scorer = train_strategy_model(sample_performance_records)
        
        # Test with invalid market conditions
        invalid_conditions = create_sample_market_conditions(volatility=float('nan'))
        
        with pytest.raises((InvalidMarketDataError, PredictionError)):
            get_strategy_recommendations(
                model=model,
                feature_extractor=extractor,
                market_conditions=invalid_conditions
            )
    
    def test_different_market_regimes(self, sample_performance_records):
        """Test recommendations across different market regimes."""
        from ..interfaces.types import MarketRegime
        
        model, extractor, scorer = train_strategy_model(sample_performance_records)
        
        regime_results = {}
        
        # Test each market regime
        for regime in MarketRegime:
            conditions = create_sample_market_conditions(market_regime=regime)
            
            recommendations = get_strategy_recommendations(
                model=model,
                feature_extractor=extractor,
                market_conditions=conditions,
                confidence_scorer=scorer
            )
            
            regime_results[regime] = recommendations
            assert len(recommendations) > 0
        
        # Different regimes should potentially give different recommendations
        # (This is probabilistic, so we just check that we get valid results)
        for regime, recommendations in regime_results.items():
            assert all(0 <= pred.confidence <= 1 for pred in recommendations)
            assert all(pred.ranking > 0 for pred in recommendations)
    
    @patch('logging.getLogger')
    def test_logging_integration(self, mock_logger, sample_performance_records, sample_market_conditions):
        """Test logging across integrated workflow."""
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        
        # Train model (should generate logs)
        model, extractor, scorer = train_strategy_model(sample_performance_records)
        
        # Get recommendations (should generate logs)
        get_strategy_recommendations(
            model=model,
            feature_extractor=extractor,
            market_conditions=sample_market_conditions
        )
        
        # Verify logging occurred
        assert mock_logger_instance.info.call_count > 0
        assert mock_logger_instance.debug.call_count > 0
    
    def test_memory_efficiency(self, large_performance_dataset):
        """Test memory efficiency with large datasets."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Train model with large dataset
        model, extractor, scorer = train_strategy_model(large_performance_dataset)
        
        # Make multiple predictions
        for _ in range(20):
            conditions = create_sample_market_conditions()
            get_strategy_recommendations(
                model=model,
                feature_extractor=extractor,
                market_conditions=conditions,
                confidence_scorer=scorer
            )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for this test)
        assert memory_increase < 500