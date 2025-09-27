"""
Production-grade Validation Engine for ML Strategy Selection.

This module implements comprehensive validation for strategy selection models,
including walk-forward analysis, cross-validation, statistical tests, and
performance benchmarking. Ensures model robustness and reliability.

Key Features:
- Walk-forward validation with realistic market conditions
- Statistical significance testing
- Performance degradation detection
- Model comparison and benchmarking
- Comprehensive validation reporting

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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy import stats
from dataclasses import dataclass, field
import warnings

from ..interfaces.types import (
    StrategyName, MarketConditions, StrategyPrediction, ModelPerformance,
    StrategyPerformanceRecord, TrainingResult,
    ModelNotTrainedError, PredictionError
)

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Comprehensive validation result for a strategy selection model.
    
    Contains statistical metrics, performance analysis, and detailed
    diagnostics for model validation assessment.
    """
    model_id: str
    validation_date: datetime
    validation_type: str  # "walk_forward", "cross_validation", "holdout"
    
    # Core metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Statistical significance
    p_value: float
    confidence_interval: Tuple[float, float]
    statistical_significance: bool
    
    # Performance metrics
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    average_return: float
    
    # Robustness metrics
    stability_score: float
    consistency_score: float
    degradation_risk: float
    
    # Detailed results
    period_results: List[Dict[str, Any]] = field(default_factory=list)
    feature_importance_stability: Optional[Dict[str, float]] = None
    benchmark_comparison: Optional[Dict[str, float]] = None
    
    # Validation metadata
    validation_samples: int = 0
    validation_periods: int = 0
    validation_duration_days: int = 0
    
    @property
    def overall_score(self) -> float:
        """Calculate overall validation score (0-1)."""
        # Weighted combination of key metrics
        return (
            0.3 * self.accuracy +
            0.2 * self.f1_score +
            0.2 * min(self.sharpe_ratio / 2, 1.0) +  # Normalize Sharpe
            0.1 * self.stability_score +
            0.1 * self.consistency_score +
            0.1 * (1 - self.degradation_risk)
        )
    
    @property
    def is_reliable(self) -> bool:
        """Check if model passes reliability criteria."""
        return (
            self.accuracy > 0.55 and
            self.statistical_significance and
            self.stability_score > 0.7 and
            self.degradation_risk < 0.3
        )


class ValidationEngine:
    """
    Production-grade validation engine for strategy selection models.
    
    Implements comprehensive validation methodologies including:
    - Walk-forward analysis with rolling windows
    - Time series cross-validation
    - Statistical significance testing
    - Performance benchmarking
    - Model stability analysis
    
    Thread Safety:
        All public methods are thread-safe using internal locks.
    """
    
    def __init__(
        self,
        walk_forward_window: int = 252,  # Trading days
        rebalance_frequency: int = 21,   # Monthly
        min_validation_samples: int = 100,
        significance_level: float = 0.05,
        benchmark_strategy: StrategyName = StrategyName.SIMPLE_MA,
        enable_statistical_tests: bool = True
    ) -> None:
        """
        Initialize Validation Engine.
        
        Args:
            walk_forward_window: Size of training window in days
            rebalance_frequency: Frequency of rebalancing in days
            min_validation_samples: Minimum samples for validation
            significance_level: Statistical significance threshold
            benchmark_strategy: Strategy to use as benchmark
            enable_statistical_tests: Whether to perform statistical tests
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if walk_forward_window < 50:
            raise ValueError(f"Walk forward window too small: {walk_forward_window}")
        
        if rebalance_frequency < 1:
            raise ValueError(f"Rebalance frequency must be positive: {rebalance_frequency}")
        
        if not 0 < significance_level < 1:
            raise ValueError(f"Significance level must be between 0 and 1: {significance_level}")
        
        self.walk_forward_window = walk_forward_window
        self.rebalance_frequency = rebalance_frequency
        self.min_validation_samples = min_validation_samples
        self.significance_level = significance_level
        self.benchmark_strategy = benchmark_strategy
        self.enable_statistical_tests = enable_statistical_tests
        
        # Validation state
        self._validation_history: List[ValidationResult] = []
        self._benchmark_performance: Optional[Dict[str, float]] = None
        
        # Performance tracking
        self._validation_times: List[float] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Initialized ValidationEngine with {walk_forward_window}-day window")
    
    def validate_walk_forward(
        self,
        model: Any,  # Strategy selector model
        training_records: List[StrategyPerformanceRecord],
        start_date: datetime,
        end_date: datetime,
        feature_extractor: Any = None
    ) -> ValidationResult:
        """
        Perform walk-forward validation.
        
        Args:
            model: Trained strategy selector model
            training_records: Historical performance records
            start_date: Start date for validation
            end_date: End date for validation
            feature_extractor: Feature extractor for processing data
            
        Returns:
            Comprehensive validation result
            
        Raises:
            ValueError: If insufficient data or invalid parameters
            RuntimeError: If validation fails
        """
        with self._lock:
            start_time = time.time()
            
            try:
                logger.info(f"Starting walk-forward validation from {start_date} to {end_date}")
                
                # Validate inputs
                if len(training_records) < self.min_validation_samples:
                    raise ValueError(f"Insufficient validation data: {len(training_records)}")
                
                # Prepare validation periods
                validation_periods = self._create_validation_periods(
                    training_records, start_date, end_date
                )
                
                if len(validation_periods) < 3:
                    raise ValueError(f"Insufficient validation periods: {len(validation_periods)}")
                
                # Run validation for each period
                period_results = []
                all_predictions = []
                all_actuals = []
                all_returns = []
                
                for i, (train_data, test_data) in enumerate(validation_periods):
                    logger.debug(f"Validating period {i+1}/{len(validation_periods)}")
                    
                    period_result = self._validate_period(
                        model, train_data, test_data, feature_extractor
                    )
                    
                    period_results.append(period_result)
                    all_predictions.extend(period_result["predictions"])
                    all_actuals.extend(period_result["actuals"])
                    all_returns.extend(period_result["returns"])
                
                # Calculate aggregate metrics
                aggregate_metrics = self._calculate_aggregate_metrics(
                    all_predictions, all_actuals, all_returns
                )
                
                # Statistical significance testing
                statistical_metrics = self._perform_statistical_tests(
                    all_predictions, all_actuals
                )
                
                # Stability analysis
                stability_metrics = self._analyze_stability(period_results)
                
                # Create validation result
                result = ValidationResult(
                    model_id=getattr(model, 'model_id', 'unknown'),
                    validation_date=datetime.now(),
                    validation_type="walk_forward",
                    accuracy=aggregate_metrics["accuracy"],
                    precision=aggregate_metrics["precision"],
                    recall=aggregate_metrics["recall"],
                    f1_score=aggregate_metrics["f1_score"],
                    p_value=statistical_metrics["p_value"],
                    confidence_interval=statistical_metrics["confidence_interval"],
                    statistical_significance=statistical_metrics["significant"],
                    sharpe_ratio=aggregate_metrics["sharpe_ratio"],
                    max_drawdown=aggregate_metrics["max_drawdown"],
                    win_rate=aggregate_metrics["win_rate"],
                    average_return=aggregate_metrics["average_return"],
                    stability_score=stability_metrics["stability_score"],
                    consistency_score=stability_metrics["consistency_score"],
                    degradation_risk=stability_metrics["degradation_risk"],
                    period_results=period_results,
                    validation_samples=len(all_predictions),
                    validation_periods=len(validation_periods),
                    validation_duration_days=(end_date - start_date).days
                )
                
                # Store validation result
                self._validation_history.append(result)
                
                validation_time = time.time() - start_time
                self._validation_times.append(validation_time)
                
                logger.info(
                    f"Walk-forward validation completed in {validation_time:.2f}s. "
                    f"Overall score: {result.overall_score:.3f}"
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Walk-forward validation failed: {str(e)}")
                raise RuntimeError(f"Validation failed: {str(e)}") from e
    
    def validate_cross_validation(
        self,
        model: Any,
        training_records: List[StrategyPerformanceRecord],
        cv_folds: int = 5,
        feature_extractor: Any = None
    ) -> ValidationResult:
        """
        Perform time series cross-validation.
        
        Args:
            model: Strategy selector model to validate
            training_records: Historical performance records
            cv_folds: Number of cross-validation folds
            feature_extractor: Feature extractor for processing data
            
        Returns:
            Cross-validation result
            
        Raises:
            ValueError: If insufficient data
            RuntimeError: If validation fails
        """
        with self._lock:
            start_time = time.time()
            
            try:
                logger.info(f"Starting {cv_folds}-fold cross-validation")
                
                # Prepare data
                if feature_extractor:
                    X = np.array([
                        feature_extractor.extract_features(record.market_conditions)
                        for record in training_records
                    ])
                else:
                    # Use simplified features if no extractor provided
                    X = np.array([
                        self._extract_simple_features(record.market_conditions)
                        for record in training_records
                    ])
                
                y = np.array([record.strategy.value for record in training_records])
                
                # Time series split
                tscv = TimeSeriesSplit(n_splits=cv_folds)
                
                fold_results = []
                all_scores = []
                
                for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                    logger.debug(f"Cross-validation fold {fold+1}/{cv_folds}")
                    
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    # Create temporary model for this fold
                    fold_model = self._create_fold_model(model)
                    
                    # Train on fold data
                    fold_training_records = [training_records[i] for i in train_idx]
                    fold_model.train(fold_training_records)
                    
                    # Predict on test data
                    fold_predictions = []
                    fold_actuals = []
                    
                    for i in test_idx:
                        record = training_records[i]
                        predictions = fold_model.predict(record.market_conditions)
                        
                        if predictions:
                            best_strategy = predictions[0].strategy
                            fold_predictions.append(best_strategy.value)
                            fold_actuals.append(record.strategy.value)
                    
                    # Calculate fold metrics
                    if fold_predictions:
                        fold_accuracy = accuracy_score(fold_actuals, fold_predictions)
                        all_scores.append(fold_accuracy)
                        
                        fold_results.append({
                            "fold": fold,
                            "accuracy": fold_accuracy,
                            "samples": len(fold_predictions)
                        })
                
                # Calculate aggregate metrics
                cv_accuracy = np.mean(all_scores)
                cv_std = np.std(all_scores)
                
                # Statistical significance (t-test against random chance)
                if self.enable_statistical_tests and len(all_scores) > 1:
                    t_stat, p_value = stats.ttest_1samp(all_scores, 0.5)  # Test against 50%
                    significant = p_value < self.significance_level
                    
                    # Confidence interval
                    ci_lower = cv_accuracy - 1.96 * cv_std / np.sqrt(len(all_scores))
                    ci_upper = cv_accuracy + 1.96 * cv_std / np.sqrt(len(all_scores))
                    confidence_interval = (ci_lower, ci_upper)
                else:
                    p_value = 1.0
                    significant = False
                    confidence_interval = (cv_accuracy, cv_accuracy)
                
                # Create result
                result = ValidationResult(
                    model_id=getattr(model, 'model_id', 'unknown'),
                    validation_date=datetime.now(),
                    validation_type="cross_validation",
                    accuracy=cv_accuracy,
                    precision=cv_accuracy,  # Simplified for CV
                    recall=cv_accuracy,
                    f1_score=cv_accuracy,
                    p_value=p_value,
                    confidence_interval=confidence_interval,
                    statistical_significance=significant,
                    sharpe_ratio=0.0,  # Not calculated in CV
                    max_drawdown=0.0,
                    win_rate=cv_accuracy,
                    average_return=0.0,
                    stability_score=1 - cv_std,  # Lower std = higher stability
                    consistency_score=cv_accuracy,
                    degradation_risk=cv_std,
                    period_results=fold_results,
                    validation_samples=len(training_records),
                    validation_periods=cv_folds
                )
                
                validation_time = time.time() - start_time
                self._validation_times.append(validation_time)
                
                logger.info(
                    f"Cross-validation completed in {validation_time:.2f}s. "
                    f"Accuracy: {cv_accuracy:.3f} ± {cv_std:.3f}"
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Cross-validation failed: {str(e)}")
                raise RuntimeError(f"Cross-validation failed: {str(e)}") from e
    
    def compare_models(
        self,
        models: List[Any],
        validation_data: List[StrategyPerformanceRecord]
    ) -> Dict[str, ValidationResult]:
        """
        Compare multiple models using standardized validation.
        
        Args:
            models: List of models to compare
            validation_data: Data for comparison
            
        Returns:
            Dictionary mapping model IDs to validation results
        """
        with self._lock:
            logger.info(f"Comparing {len(models)} models")
            
            results = {}
            
            for model in models:
                try:
                    model_id = getattr(model, 'model_id', f'model_{id(model)}')
                    logger.debug(f"Validating model {model_id}")
                    
                    # Use cross-validation for comparison
                    result = self.validate_cross_validation(model, validation_data)
                    results[model_id] = result
                    
                except Exception as e:
                    logger.error(f"Failed to validate model {model_id}: {str(e)}")
                    continue
            
            # Log comparison summary
            if results:
                best_model = max(results.keys(), key=lambda k: results[k].overall_score)
                logger.info(f"Best model: {best_model} (score: {results[best_model].overall_score:.3f})")
            
            return results
    
    def get_validation_history(self) -> List[ValidationResult]:
        """Get validation history."""
        with self._lock:
            return self._validation_history.copy()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get validation engine performance statistics."""
        with self._lock:
            stats = {
                "total_validations": len(self._validation_history),
                "validation_methods": list(set(r.validation_type for r in self._validation_history))
            }
            
            if self._validation_times:
                stats.update({
                    "avg_validation_time_s": np.mean(self._validation_times),
                    "max_validation_time_s": np.max(self._validation_times)
                })
            
            if self._validation_history:
                recent_scores = [r.overall_score for r in self._validation_history[-10:]]
                stats.update({
                    "recent_avg_score": np.mean(recent_scores),
                    "score_trend": "improving" if len(recent_scores) > 5 and 
                                   recent_scores[-1] > np.mean(recent_scores[:-1]) else "stable"
                })
            
            return stats
    
    # Private methods
    
    def _create_validation_periods(
        self,
        records: List[StrategyPerformanceRecord],
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[List[StrategyPerformanceRecord], List[StrategyPerformanceRecord]]]:
        """Create training/testing periods for walk-forward validation."""
        # Sort records by date
        sorted_records = sorted(records, key=lambda r: r.date)
        
        periods = []
        current_date = start_date
        
        while current_date + timedelta(days=self.walk_forward_window) < end_date:
            # Training window
            train_start = current_date - timedelta(days=self.walk_forward_window)
            train_end = current_date
            
            # Test window
            test_start = current_date
            test_end = current_date + timedelta(days=self.rebalance_frequency)
            
            # Filter records for training and testing
            train_records = [
                r for r in sorted_records 
                if train_start <= r.date < train_end
            ]
            
            test_records = [
                r for r in sorted_records 
                if test_start <= r.date < test_end
            ]
            
            if len(train_records) >= 20 and len(test_records) >= 5:
                periods.append((train_records, test_records))
            
            current_date += timedelta(days=self.rebalance_frequency)
        
        return periods
    
    def _validate_period(
        self,
        model: Any,
        train_data: List[StrategyPerformanceRecord],
        test_data: List[StrategyPerformanceRecord],
        feature_extractor: Any = None
    ) -> Dict[str, Any]:
        """Validate model on a single period."""
        # Train model on period data
        period_model = self._create_period_model(model)
        period_model.train(train_data)
        
        predictions = []
        actuals = []
        returns = []
        
        for record in test_data:
            try:
                # Get model prediction
                model_predictions = period_model.predict(record.market_conditions)
                
                if model_predictions:
                    predicted_strategy = model_predictions[0].strategy
                    predictions.append(predicted_strategy.value)
                    actuals.append(record.strategy.value)
                    returns.append(record.actual_return)
                    
            except Exception as e:
                logger.warning(f"Prediction failed for record: {str(e)}")
                continue
        
        # Calculate period metrics
        if predictions:
            accuracy = accuracy_score(actuals, predictions)
            avg_return = np.mean(returns)
            sharpe = avg_return / np.std(returns) if np.std(returns) > 0 else 0
        else:
            accuracy = 0.0
            avg_return = 0.0
            sharpe = 0.0
        
        return {
            "predictions": predictions,
            "actuals": actuals,
            "returns": returns,
            "accuracy": accuracy,
            "average_return": avg_return,
            "sharpe_ratio": sharpe,
            "samples": len(predictions)
        }
    
    def _calculate_aggregate_metrics(
        self,
        predictions: List[str],
        actuals: List[str],
        returns: List[float]
    ) -> Dict[str, float]:
        """Calculate aggregate metrics across all periods."""
        metrics = {}
        
        if predictions:
            # Classification metrics
            metrics["accuracy"] = accuracy_score(actuals, predictions)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics["precision"] = precision_score(
                    actuals, predictions, average='weighted', zero_division=0
                )
                metrics["recall"] = recall_score(
                    actuals, predictions, average='weighted', zero_division=0
                )
                
                f1_num = 2 * metrics["precision"] * metrics["recall"]
                f1_den = metrics["precision"] + metrics["recall"]
                metrics["f1_score"] = f1_num / f1_den if f1_den > 0 else 0
        
        if returns:
            # Financial metrics
            returns_array = np.array(returns)
            metrics["average_return"] = np.mean(returns_array)
            metrics["win_rate"] = np.mean(returns_array > 0)
            
            if np.std(returns_array) > 0:
                metrics["sharpe_ratio"] = metrics["average_return"] / np.std(returns_array)
            else:
                metrics["sharpe_ratio"] = 0.0
            
            # Calculate max drawdown
            cumulative_returns = np.cumprod(1 + returns_array / 100)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            metrics["max_drawdown"] = np.min(drawdowns)
        
        return metrics
    
    def _perform_statistical_tests(
        self,
        predictions: List[str],
        actuals: List[str]
    ) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        if not self.enable_statistical_tests or len(predictions) < 10:
            return {
                "p_value": 1.0,
                "confidence_interval": (0.0, 1.0),
                "significant": False
            }
        
        # Calculate accuracy
        accuracy = accuracy_score(actuals, predictions)
        n = len(predictions)
        
        # Binomial test against random chance (assuming balanced classes)
        successes = int(accuracy * n)
        p_value = stats.binom_test(successes, n, 0.5, alternative='greater')
        
        # Confidence interval for accuracy
        ci_lower = accuracy - 1.96 * np.sqrt(accuracy * (1 - accuracy) / n)
        ci_upper = accuracy + 1.96 * np.sqrt(accuracy * (1 - accuracy) / n)
        
        return {
            "p_value": p_value,
            "confidence_interval": (max(0, ci_lower), min(1, ci_upper)),
            "significant": p_value < self.significance_level
        }
    
    def _analyze_stability(self, period_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze model stability across periods."""
        if len(period_results) < 3:
            return {
                "stability_score": 0.0,
                "consistency_score": 0.0,
                "degradation_risk": 1.0
            }
        
        accuracies = [r["accuracy"] for r in period_results]
        returns = [r["average_return"] for r in period_results]
        
        # Stability: inverse of coefficient of variation
        accuracy_cv = np.std(accuracies) / np.mean(accuracies) if np.mean(accuracies) > 0 else 1
        stability_score = max(0, 1 - accuracy_cv)
        
        # Consistency: how often performance exceeds baseline
        baseline_accuracy = 0.5  # Random chance
        consistency_score = np.mean([a > baseline_accuracy for a in accuracies])
        
        # Degradation risk: trend analysis
        if len(accuracies) >= 5:
            # Linear regression slope
            x = np.arange(len(accuracies))
            slope, _, _, _, _ = stats.linregress(x, accuracies)
            degradation_risk = max(0, -slope * 10)  # Scale slope
        else:
            degradation_risk = 0.5  # Neutral
        
        return {
            "stability_score": stability_score,
            "consistency_score": consistency_score,
            "degradation_risk": min(1.0, degradation_risk)
        }
    
    def _create_period_model(self, original_model: Any) -> Any:
        """Create a copy of model for period validation."""
        # This is a simplified approach - in practice, you'd want proper model cloning
        # For now, return the original model (assumes it can be retrained)
        return original_model
    
    def _create_fold_model(self, original_model: Any) -> Any:
        """Create a copy of model for cross-validation fold."""
        # Simplified approach - return original model
        return original_model
    
    def _extract_simple_features(self, conditions: MarketConditions) -> np.ndarray:
        """Extract simple features when no feature extractor is provided."""
        return np.array([
            conditions.volatility / 100,
            (conditions.trend_strength + 100) / 200,
            min(conditions.volume_ratio / 3, 1),
            (conditions.price_momentum + 100) / 200,
            conditions.vix_level / 100,
            (conditions.correlation_spy + 1) / 2
        ])