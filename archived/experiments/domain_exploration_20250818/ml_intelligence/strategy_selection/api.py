"""
High-level API for ML Strategy Selection Domain.

This module provides convenient high-level functions for common strategy
selection tasks. Designed for ease of use while maintaining full access
to underlying production-grade components.

Key Functions:
- train_strategy_model: Complete model training pipeline
- get_strategy_recommendations: Real-time strategy recommendations
- evaluate_model_performance: Comprehensive model evaluation
- batch_predict_strategies: Batch prediction for multiple conditions

Production Standards:
- Complete type hints with runtime validation
- Comprehensive error handling with specific exceptions
- Structured logging for all operations
- Performance optimizations
- Input validation and sanitization
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from .core import StrategySelector, FeatureExtractor, ConfidenceScorer, ValidationEngine
from .interfaces.types import (
    StrategyName, MarketConditions, StrategyPrediction, ModelPerformance,
    TrainingResult, StrategyPerformanceRecord, PredictionRequest,
    StrategySelectionError, ModelNotTrainedError, InvalidMarketDataError,
    PredictionError
)

# Configure module logger
logger = logging.getLogger(__name__)


def train_strategy_model(
    training_records: List[StrategyPerformanceRecord],
    validation_split: float = 0.2,
    test_split: float = 0.1,
    enable_feature_engineering: bool = True,
    enable_confidence_scoring: bool = True,
    model_config: Optional[Dict[str, Any]] = None
) -> Tuple[StrategySelector, FeatureExtractor, Optional[ConfidenceScorer]]:
    """
    Train a complete strategy selection model with all components.
    
    This is the main entry point for training strategy selection models.
    Handles the complete pipeline including feature extraction, model training,
    and confidence scoring setup.
    
    Args:
        training_records: Historical performance records for training
        validation_split: Fraction of data for validation (0-1)
        test_split: Fraction of data for testing (0-1)
        enable_feature_engineering: Whether to use advanced feature engineering
        enable_confidence_scoring: Whether to train confidence scorer
        model_config: Optional configuration for model components
        
    Returns:
        Tuple of (trained_model, feature_extractor, confidence_scorer)
        
    Raises:
        ValueError: If training data is insufficient or invalid
        StrategySelectionError: If training fails
        
    Example:
        >>> records = load_training_data()
        >>> model, extractor, scorer = train_strategy_model(records)
        >>> predictions = model.predict(market_conditions)
    """
    logger.info(f"Starting strategy model training with {len(training_records)} records")
    
    try:
        # Validate inputs
        if len(training_records) < 100:
            raise ValueError(f"Insufficient training data: {len(training_records)} records (minimum 100)")
        
        if not 0 < validation_split < 1:
            raise ValueError(f"Invalid validation split: {validation_split}")
        
        if not 0 <= test_split < 1:
            raise ValueError(f"Invalid test split: {test_split}")
        
        if validation_split + test_split >= 1:
            raise ValueError("Validation and test splits must sum to less than 1")
        
        # Use default config if none provided
        if model_config is None:
            model_config = {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            }
        
        # Initialize feature extractor
        logger.info("Initializing feature extractor")
        feature_extractor = FeatureExtractor(
            enable_feature_engineering=enable_feature_engineering,
            enable_feature_selection=True,
            feature_selection_k=20
        )
        
        # Fit feature extractor
        market_conditions = [record.market_conditions for record in training_records]
        target_values = np.array([record.actual_return for record in training_records])
        
        feature_extractor.fit(market_conditions, target_values)
        logger.info(f"Feature extractor fitted with {len(feature_extractor.get_feature_names())} features")
        
        # Initialize and train strategy selector
        logger.info("Training strategy selector")
        strategy_selector = StrategySelector(**model_config)
        
        training_result = strategy_selector.train(
            training_records=training_records,
            validation_split=validation_split,
            test_split=test_split
        )
        
        logger.info(f"Strategy selector trained - validation score: {training_result.validation_score:.4f}")
        
        # Initialize confidence scorer if enabled
        confidence_scorer = None
        if enable_confidence_scoring:
            logger.info("Training confidence scorer")
            confidence_scorer = ConfidenceScorer()
            
            # Extract features for confidence training
            feature_matrix = np.array([
                feature_extractor.extract_features(record.market_conditions)
                for record in training_records
            ])
            
            confidence_scorer.fit(training_records, feature_matrix)
            logger.info("Confidence scorer training completed")
        
        # Log training summary
        logger.info(
            f"Model training completed successfully:\n"
            f"  - Training samples: {training_result.training_samples}\n"
            f"  - Validation score: {training_result.validation_score:.4f}\n"
            f"  - Test score: {training_result.test_score:.4f}\n"
            f"  - Features used: {len(training_result.features_used)}\n"
            f"  - Training time: {training_result.training_time_seconds:.2f}s"
        )
        
        return strategy_selector, feature_extractor, confidence_scorer
        
    except Exception as e:
        logger.error(f"Strategy model training failed: {str(e)}")
        raise StrategySelectionError(f"Model training failed: {str(e)}") from e


def get_strategy_recommendations(
    model: StrategySelector,
    feature_extractor: FeatureExtractor,
    market_conditions: MarketConditions,
    top_n: int = 3,
    min_confidence: float = 0.5,
    confidence_scorer: Optional[ConfidenceScorer] = None
) -> List[StrategyPrediction]:
    """
    Get strategy recommendations for current market conditions.
    
    This is the main entry point for getting real-time strategy recommendations.
    Handles feature extraction, prediction, and confidence scoring.
    
    Args:
        model: Trained strategy selector model
        feature_extractor: Fitted feature extractor
        market_conditions: Current market state
        top_n: Number of top strategies to return
        min_confidence: Minimum confidence threshold for recommendations
        confidence_scorer: Optional confidence scorer for enhanced scoring
        
    Returns:
        List of strategy predictions sorted by performance
        
    Raises:
        ModelNotTrainedError: If model hasn't been trained
        InvalidMarketDataError: If market conditions are invalid
        PredictionError: If prediction fails
        
    Example:
        >>> conditions = MarketConditions(volatility=20, trend_strength=30, ...)
        >>> predictions = get_strategy_recommendations(model, extractor, conditions)
        >>> best_strategy = predictions[0].strategy
    """
    logger.debug(f"Getting strategy recommendations for {market_conditions.market_regime.value} market")
    
    try:
        # Validate inputs
        if not model.is_trained:
            raise ModelNotTrainedError("Strategy selector model is not trained")
        
        if top_n < 1:
            raise ValueError(f"top_n must be positive, got {top_n}")
        
        if not 0 <= min_confidence <= 1:
            raise ValueError(f"min_confidence must be between 0 and 1, got {min_confidence}")
        
        # Get base predictions from model
        predictions = model.predict(market_conditions)
        
        # Enhance with confidence scorer if available
        if confidence_scorer is not None:
            logger.debug("Enhancing predictions with confidence scorer")
            
            features = feature_extractor.extract_features(market_conditions)
            
            for prediction in predictions:
                # Get enhanced confidence score
                enhanced_confidence = confidence_scorer.score_confidence(
                    strategy=prediction.strategy,
                    market_conditions=market_conditions,
                    features=features,
                    base_prediction=prediction.expected_return
                )
                
                # Update prediction with enhanced confidence
                prediction.confidence = enhanced_confidence
        
        # Filter by minimum confidence
        filtered_predictions = [
            p for p in predictions 
            if p.confidence >= min_confidence
        ]
        
        if not filtered_predictions:
            logger.warning(f"No predictions meet minimum confidence threshold {min_confidence}")
            # Return best prediction even if below threshold, but log warning
            if predictions:
                filtered_predictions = [predictions[0]]
        
        # Re-sort by risk-adjusted score and limit to top_n
        filtered_predictions.sort(key=lambda p: p.risk_adjusted_score, reverse=True)
        top_predictions = filtered_predictions[:top_n]
        
        # Update rankings
        for i, prediction in enumerate(top_predictions):
            prediction.ranking = i + 1
        
        logger.debug(f"Returning {len(top_predictions)} strategy recommendations")
        
        return top_predictions
        
    except (ModelNotTrainedError, InvalidMarketDataError, PredictionError):
        raise
    except Exception as e:
        logger.error(f"Strategy recommendation failed: {str(e)}")
        raise PredictionError(f"Failed to get strategy recommendations: {str(e)}") from e


def evaluate_model_performance(
    model: StrategySelector,
    validation_records: List[StrategyPerformanceRecord],
    feature_extractor: Optional[FeatureExtractor] = None,
    validation_type: str = "walk_forward"
) -> Dict[str, Any]:
    """
    Comprehensively evaluate model performance.
    
    Performs thorough validation using multiple methodologies to assess
    model reliability, stability, and real-world performance.
    
    Args:
        model: Trained strategy selector to evaluate
        validation_records: Historical records for validation
        feature_extractor: Optional feature extractor for processing
        validation_type: Type of validation ("walk_forward", "cross_validation", "holdout")
        
    Returns:
        Dictionary containing comprehensive evaluation results
        
    Raises:
        ValueError: If insufficient validation data
        StrategySelectionError: If evaluation fails
        
    Example:
        >>> evaluation = evaluate_model_performance(model, validation_data)
        >>> print(f"Model reliability: {evaluation['is_reliable']}")
        >>> print(f"Overall score: {evaluation['overall_score']:.3f}")
    """
    logger.info(f"Evaluating model performance using {validation_type} validation")
    
    try:
        # Validate inputs
        if len(validation_records) < 50:
            raise ValueError(f"Insufficient validation data: {len(validation_records)} records")
        
        if validation_type not in ["walk_forward", "cross_validation", "holdout"]:
            raise ValueError(f"Invalid validation type: {validation_type}")
        
        # Initialize validation engine
        validation_engine = ValidationEngine()
        
        # Perform validation based on type
        if validation_type == "walk_forward":
            # Use the date range from validation records
            dates = [record.date for record in validation_records]
            start_date = min(dates)
            end_date = max(dates)
            
            result = validation_engine.validate_walk_forward(
                model=model,
                training_records=validation_records,
                start_date=start_date,
                end_date=end_date,
                feature_extractor=feature_extractor
            )
            
        elif validation_type == "cross_validation":
            result = validation_engine.validate_cross_validation(
                model=model,
                training_records=validation_records,
                cv_folds=5,
                feature_extractor=feature_extractor
            )
            
        else:  # holdout
            # Split data for holdout validation
            split_point = int(len(validation_records) * 0.7)
            train_data = validation_records[:split_point]
            test_data = validation_records[split_point:]
            
            # Retrain model on training portion
            model.train(train_data)
            
            # Evaluate on test portion
            result = validation_engine.validate_cross_validation(
                model=model,
                training_records=test_data,
                cv_folds=3,
                feature_extractor=feature_extractor
            )
        
        # Create comprehensive evaluation report
        evaluation = {
            "validation_type": validation_type,
            "validation_date": result.validation_date,
            "model_id": result.model_id,
            
            # Core performance metrics
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1_score": result.f1_score,
            
            # Financial performance
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "average_return": result.average_return,
            
            # Statistical validation
            "p_value": result.p_value,
            "confidence_interval": result.confidence_interval,
            "statistical_significance": result.statistical_significance,
            
            # Robustness metrics
            "stability_score": result.stability_score,
            "consistency_score": result.consistency_score,
            "degradation_risk": result.degradation_risk,
            
            # Overall assessment
            "overall_score": result.overall_score,
            "is_reliable": result.is_reliable,
            
            # Detailed results
            "validation_samples": result.validation_samples,
            "validation_periods": result.validation_periods,
            "period_results": result.period_results
        }
        
        logger.info(
            f"Model evaluation completed:\n"
            f"  - Overall score: {result.overall_score:.3f}\n"
            f"  - Accuracy: {result.accuracy:.3f}\n"
            f"  - Sharpe ratio: {result.sharpe_ratio:.3f}\n"
            f"  - Reliability: {'Yes' if result.is_reliable else 'No'}"
        )
        
        return evaluation
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        raise StrategySelectionError(f"Evaluation failed: {str(e)}") from e


def create_prediction_request(
    symbol: str,
    market_conditions: MarketConditions,
    top_n_strategies: int = 3,
    min_confidence: float = 0.5,
    risk_tolerance: float = 1.0,
    exclude_strategies: Optional[List[StrategyName]] = None
) -> PredictionRequest:
    """
    Create a standardized prediction request.
    
    Convenience function for creating properly validated prediction requests
    with sensible defaults and comprehensive validation.
    
    Args:
        symbol: Trading symbol (e.g., "AAPL", "SPY")
        market_conditions: Current market state
        top_n_strategies: Number of strategies to recommend
        min_confidence: Minimum confidence threshold
        risk_tolerance: Risk tolerance level (0-3)
        exclude_strategies: Strategies to exclude from recommendations
        
    Returns:
        Validated prediction request
        
    Raises:
        ValueError: If parameters are invalid
        
    Example:
        >>> conditions = MarketConditions(volatility=20, ...)
        >>> request = create_prediction_request("AAPL", conditions, top_n_strategies=5)
    """
    try:
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        symbol = symbol.upper().strip()
        
        # Create and validate request
        request = PredictionRequest(
            symbol=symbol,
            market_conditions=market_conditions,
            top_n_strategies=top_n_strategies,
            min_confidence=min_confidence,
            risk_tolerance=risk_tolerance,
            exclude_strategies=exclude_strategies or []
        )
        
        logger.debug(f"Created prediction request for {symbol}")
        return request
        
    except Exception as e:
        logger.error(f"Failed to create prediction request: {str(e)}")
        raise ValueError(f"Invalid prediction request: {str(e)}") from e


def batch_predict_strategies(
    model: StrategySelector,
    feature_extractor: FeatureExtractor,
    requests: List[PredictionRequest],
    confidence_scorer: Optional[ConfidenceScorer] = None,
    max_workers: int = 4
) -> Dict[str, List[StrategyPrediction]]:
    """
    Perform batch prediction for multiple symbols/conditions.
    
    Efficiently processes multiple prediction requests with optional
    parallel processing and comprehensive error handling.
    
    Args:
        model: Trained strategy selector model
        feature_extractor: Fitted feature extractor
        requests: List of prediction requests to process
        confidence_scorer: Optional confidence scorer
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary mapping symbols to their strategy predictions
        
    Raises:
        ModelNotTrainedError: If model hasn't been trained
        PredictionError: If batch prediction fails
        
    Example:
        >>> requests = [create_prediction_request("AAPL", conditions1),
        ...            create_prediction_request("GOOGL", conditions2)]
        >>> results = batch_predict_strategies(model, extractor, requests)
        >>> aapl_strategies = results["AAPL"]
    """
    logger.info(f"Starting batch prediction for {len(requests)} requests")
    
    try:
        # Validate inputs
        if not model.is_trained:
            raise ModelNotTrainedError("Strategy selector model is not trained")
        
        if not requests:
            return {}
        
        results = {}
        successful_predictions = 0
        failed_predictions = 0
        
        # Process requests (simplified sequential processing for now)
        for request in requests:
            try:
                predictions = get_strategy_recommendations(
                    model=model,
                    feature_extractor=feature_extractor,
                    market_conditions=request.market_conditions,
                    top_n=request.top_n_strategies,
                    min_confidence=request.min_confidence,
                    confidence_scorer=confidence_scorer
                )
                
                # Filter excluded strategies
                if request.exclude_strategies:
                    predictions = [
                        p for p in predictions 
                        if p.strategy not in request.exclude_strategies
                    ]
                
                results[request.symbol] = predictions
                successful_predictions += 1
                
            except Exception as e:
                logger.warning(f"Prediction failed for {request.symbol}: {str(e)}")
                results[request.symbol] = []
                failed_predictions += 1
        
        logger.info(
            f"Batch prediction completed: {successful_predictions} successful, "
            f"{failed_predictions} failed"
        )
        
        return results
        
    except ModelNotTrainedError:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise PredictionError(f"Batch prediction failed: {str(e)}") from e