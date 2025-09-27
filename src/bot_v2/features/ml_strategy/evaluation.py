"""
Model evaluation utilities - LOCAL to this slice.
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from .types import ModelPerformance

# Import error handling
from bot_v2.errors import ValidationError, DataError

logger = logging.getLogger(__name__)


def evaluate_predictions(predictions: np.ndarray, actuals: np.ndarray) -> ModelPerformance:
    """
    Evaluate model predictions against actual results with validation.
    
    Returns comprehensive performance metrics.
    """
    # Validate inputs
    try:
        _validate_evaluation_inputs(predictions, actuals)
    except Exception as e:
        raise ValidationError(f"Invalid evaluation inputs: {str(e)}")
    
    try:
        # Convert to comparable format with error handling
        pred_strategies = _extract_strategies(predictions)
        actual_strategies = _extract_strategies(actuals)
        
        # Validate extracted strategies
        if len(pred_strategies) != len(actual_strategies):
            raise DataError(f"Strategy extraction length mismatch: {len(pred_strategies)} vs {len(actual_strategies)}")
        
        # Calculate classification metrics with validation
        accuracy = calculate_accuracy(pred_strategies, actual_strategies)
        precision = calculate_precision(pred_strategies, actual_strategies)
        recall = calculate_recall(pred_strategies, actual_strategies)
        f1 = calculate_f1_score(precision, recall)
        
        # Calculate regression metrics for returns
        pred_returns = _extract_returns(predictions)
        actual_returns = _extract_returns(actuals)
        
        # Validate extracted returns
        if len(pred_returns) != len(actual_returns):
            logger.warning(f"Return extraction length mismatch: {len(pred_returns)} vs {len(actual_returns)}")
            # Pad shorter array with zeros
            max_len = max(len(pred_returns), len(actual_returns))
            pred_returns = np.pad(pred_returns, (0, max_len - len(pred_returns)), 'constant')
            actual_returns = np.pad(actual_returns, (0, max_len - len(actual_returns)), 'constant')
        
        mae = calculate_mae(pred_returns, actual_returns)
        r2 = calculate_r_squared(pred_returns, actual_returns)
        correlation = calculate_correlation(pred_returns, actual_returns)
        
        # Count successes with validation
        successful = np.sum(pred_strategies == actual_strategies) if len(pred_strategies) > 0 else 0
        
        # Validate all metrics before creating ModelPerformance
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mean_absolute_error': mae,
            'r_squared': r2,
            'backtest_correlation': correlation
        }
        
        validated_metrics = _validate_metrics(metrics)
        
        return ModelPerformance(
            accuracy=validated_metrics['accuracy'],
            precision=validated_metrics['precision'],
            recall=validated_metrics['recall'],
            f1_score=validated_metrics['f1_score'],
            mean_absolute_error=validated_metrics['mean_absolute_error'],
            r_squared=validated_metrics['r_squared'],
            backtest_correlation=validated_metrics['backtest_correlation'],
            total_predictions=len(predictions),
            successful_predictions=int(successful)
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        # Return default metrics
        return _create_default_performance(len(predictions))


def calculate_metrics(predictions: Dict, actuals: Dict) -> Dict[str, float]:
    """Calculate various performance metrics with validation."""
    # Validate inputs
    if predictions is None or actuals is None:
        raise ValidationError("Predictions and actuals cannot be None")
    
    if not isinstance(predictions, dict) or not isinstance(actuals, dict):
        raise ValidationError("Predictions and actuals must be dictionaries")
    
    metrics = {}
    
    try:
        # Strategy selection accuracy
        if 'strategy' in predictions and 'strategy' in actuals:
            try:
                accuracy = float(predictions['strategy'] == actuals['strategy'])
                metrics['strategy_accuracy'] = _validate_metric_value(accuracy, 'strategy_accuracy', 0.0, 1.0)
            except Exception as e:
                logger.warning(f"Failed to calculate strategy accuracy: {e}")
                metrics['strategy_accuracy'] = 0.0
        
        # Return prediction error
        if 'return' in predictions and 'return' in actuals:
            try:
                pred_return = _validate_numeric_value(predictions['return'], 'predicted_return')
                actual_return = _validate_numeric_value(actuals['return'], 'actual_return')
                
                mae = abs(pred_return - actual_return)
                mse = (pred_return - actual_return) ** 2
                
                metrics['return_mae'] = _validate_metric_value(mae, 'return_mae', 0.0, 1000.0)
                metrics['return_mse'] = _validate_metric_value(mse, 'return_mse', 0.0, 1000000.0)
            except Exception as e:
                logger.warning(f"Failed to calculate return metrics: {e}")
                metrics['return_mae'] = 100.0  # Default high error
                metrics['return_mse'] = 10000.0
        
        # Sharpe prediction error
        if 'sharpe' in predictions and 'sharpe' in actuals:
            try:
                pred_sharpe = _validate_numeric_value(predictions['sharpe'], 'predicted_sharpe')
                actual_sharpe = _validate_numeric_value(actuals['sharpe'], 'actual_sharpe')
                
                sharpe_mae = abs(pred_sharpe - actual_sharpe)
                metrics['sharpe_mae'] = _validate_metric_value(sharpe_mae, 'sharpe_mae', 0.0, 10.0)
            except Exception as e:
                logger.warning(f"Failed to calculate Sharpe metrics: {e}")
                metrics['sharpe_mae'] = 2.0  # Default moderate error
        
        # Drawdown prediction error
        if 'drawdown' in predictions and 'drawdown' in actuals:
            try:
                pred_drawdown = _validate_numeric_value(predictions['drawdown'], 'predicted_drawdown')
                actual_drawdown = _validate_numeric_value(actuals['drawdown'], 'actual_drawdown')
                
                drawdown_mae = abs(pred_drawdown - actual_drawdown)
                metrics['drawdown_mae'] = _validate_metric_value(drawdown_mae, 'drawdown_mae', 0.0, 100.0)
            except Exception as e:
                logger.warning(f"Failed to calculate drawdown metrics: {e}")
                metrics['drawdown_mae'] = 20.0  # Default moderate error
        
        return metrics
        
    except Exception as e:
        logger.error(f"Metric calculation failed: {e}")
        return {'error': 1.0}  # Return error indicator


# Metric calculation functions

def calculate_accuracy(pred: np.ndarray, actual: np.ndarray) -> float:
    """Calculate classification accuracy with validation."""
    try:
        # Validate inputs
        _validate_array_inputs(pred, actual, 'accuracy')
        
        if len(pred) == 0:
            return 0.0
        
        # Calculate accuracy
        correct = np.sum(pred == actual)
        accuracy = correct / len(pred)
        
        # Validate result
        if np.isnan(accuracy) or np.isinf(accuracy):
            logger.warning("Invalid accuracy calculated, returning 0")
            return 0.0
        
        return float(np.clip(accuracy, 0.0, 1.0))
        
    except Exception as e:
        logger.error(f"Accuracy calculation failed: {e}")
        return 0.0


def calculate_precision(pred: np.ndarray, actual: np.ndarray) -> float:
    """Calculate precision score (simplified for multi-class) with validation."""
    try:
        # Validate inputs
        _validate_array_inputs(pred, actual, 'precision')
        
        if len(pred) == 0:
            return 0.0
        
        # Get unique classes safely
        try:
            unique_classes = np.unique(np.concatenate([pred, actual]))
        except Exception as e:
            logger.warning(f"Failed to get unique classes: {e}")
            return 0.0
        
        if len(unique_classes) == 0:
            return 0.0
        
        # Calculate precision for each class
        precisions = []
        
        for cls in unique_classes:
            try:
                true_positives = np.sum((pred == cls) & (actual == cls))
                predicted_positives = np.sum(pred == cls)
                
                if predicted_positives > 0:
                    precision = true_positives / predicted_positives
                    if not (np.isnan(precision) or np.isinf(precision)):
                        precisions.append(precision)
            except Exception as e:
                logger.debug(f"Failed to calculate precision for class {cls}: {e}")
                continue
        
        if not precisions:
            return 0.0
        
        mean_precision = np.mean(precisions)
        
        # Validate result
        if np.isnan(mean_precision) or np.isinf(mean_precision):
            return 0.0
        
        return float(np.clip(mean_precision, 0.0, 1.0))
        
    except Exception as e:
        logger.error(f"Precision calculation failed: {e}")
        return 0.0


def calculate_recall(pred: np.ndarray, actual: np.ndarray) -> float:
    """Calculate recall score (simplified for multi-class) with validation."""
    try:
        # Validate inputs
        _validate_array_inputs(pred, actual, 'recall')
        
        if len(actual) == 0:
            return 0.0
        
        # Get unique classes safely
        try:
            unique_classes = np.unique(np.concatenate([pred, actual]))
        except Exception as e:
            logger.warning(f"Failed to get unique classes: {e}")
            return 0.0
        
        if len(unique_classes) == 0:
            return 0.0
        
        # Calculate recall for each class
        recalls = []
        
        for cls in unique_classes:
            try:
                true_positives = np.sum((pred == cls) & (actual == cls))
                actual_positives = np.sum(actual == cls)
                
                if actual_positives > 0:
                    recall = true_positives / actual_positives
                    if not (np.isnan(recall) or np.isinf(recall)):
                        recalls.append(recall)
            except Exception as e:
                logger.debug(f"Failed to calculate recall for class {cls}: {e}")
                continue
        
        if not recalls:
            return 0.0
        
        mean_recall = np.mean(recalls)
        
        # Validate result
        if np.isnan(mean_recall) or np.isinf(mean_recall):
            return 0.0
        
        return float(np.clip(mean_recall, 0.0, 1.0))
        
    except Exception as e:
        logger.error(f"Recall calculation failed: {e}")
        return 0.0


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score with validation."""
    try:
        # Validate inputs
        if precision is None or recall is None:
            return 0.0
        
        if np.isnan(precision) or np.isinf(precision) or np.isnan(recall) or np.isinf(recall):
            return 0.0
        
        if precision < 0 or precision > 1 or recall < 0 or recall > 1:
            logger.warning(f"Invalid precision ({precision}) or recall ({recall}) for F1 calculation")
            return 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        # Validate result
        if np.isnan(f1) or np.isinf(f1):
            return 0.0
        
        return float(np.clip(f1, 0.0, 1.0))
        
    except Exception as e:
        logger.error(f"F1 score calculation failed: {e}")
        return 0.0


def calculate_mae(pred: np.ndarray, actual: np.ndarray) -> float:
    """Calculate mean absolute error with validation."""
    try:
        # Validate inputs
        _validate_array_inputs(pred, actual, 'MAE')
        
        if len(pred) == 0:
            return 0.0
        
        # Calculate MAE
        differences = pred - actual
        
        # Check for invalid differences
        if np.any(np.isnan(differences)) or np.any(np.isinf(differences)):
            # Filter out invalid values
            valid_mask = ~(np.isnan(differences) | np.isinf(differences))
            if not np.any(valid_mask):
                logger.warning("All differences are invalid for MAE calculation")
                return 100.0  # High error indicator
            differences = differences[valid_mask]
        
        mae = np.mean(np.abs(differences))
        
        # Validate result
        if np.isnan(mae) or np.isinf(mae):
            logger.warning("Invalid MAE calculated")
            return 100.0
        
        # Cap extremely large MAE values
        if mae > 1000.0:
            logger.warning(f"Very large MAE calculated: {mae}, capping at 1000")
            mae = 1000.0
        
        return float(mae)
        
    except Exception as e:
        logger.error(f"MAE calculation failed: {e}")
        return 100.0


def calculate_r_squared(pred: np.ndarray, actual: np.ndarray) -> float:
    """Calculate R-squared value with validation."""
    try:
        # Validate inputs
        _validate_array_inputs(pred, actual, 'R-squared')
        
        if len(actual) == 0:
            return 0.0
        
        # Remove any invalid values
        valid_mask = ~(np.isnan(pred) | np.isinf(pred) | np.isnan(actual) | np.isinf(actual))
        
        if not np.any(valid_mask):
            logger.warning("No valid values for R-squared calculation")
            return 0.0
        
        valid_pred = pred[valid_mask]
        valid_actual = actual[valid_mask]
        
        if len(valid_actual) < 2:
            logger.warning("Insufficient valid data for R-squared calculation")
            return 0.0
        
        # Calculate R-squared
        mean_actual = np.mean(valid_actual)
        
        ss_res = np.sum((valid_actual - valid_pred) ** 2)
        ss_tot = np.sum((valid_actual - mean_actual) ** 2)
        
        if ss_tot == 0:
            # If actual values have no variance, return 0
            return 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        
        # Validate result
        if np.isnan(r2) or np.isinf(r2):
            logger.warning("Invalid R-squared calculated")
            return 0.0
        
        # R-squared can be negative for very poor fits
        # Cap at reasonable bounds
        r2 = np.clip(r2, -10.0, 1.0)
        
        return float(r2)
        
    except Exception as e:
        logger.error(f"R-squared calculation failed: {e}")
        return 0.0


def calculate_correlation(pred: np.ndarray, actual: np.ndarray) -> float:
    """Calculate correlation coefficient with validation."""
    try:
        # Validate inputs
        _validate_array_inputs(pred, actual, 'correlation')
        
        if len(pred) < 2:
            return 0.0
        
        # Remove any invalid values
        valid_mask = ~(np.isnan(pred) | np.isinf(pred) | np.isnan(actual) | np.isinf(actual))
        
        if not np.any(valid_mask):
            logger.warning("No valid values for correlation calculation")
            return 0.0
        
        valid_pred = pred[valid_mask]
        valid_actual = actual[valid_mask]
        
        if len(valid_pred) < 2:
            logger.warning("Insufficient valid data for correlation calculation")
            return 0.0
        
        # Handle edge cases - check for zero variance
        pred_std = np.std(valid_pred)
        actual_std = np.std(valid_actual)
        
        if pred_std == 0 or actual_std == 0:
            logger.debug("Zero variance in data, correlation is 0")
            return 0.0
        
        # Calculate correlation
        try:
            correlation_matrix = np.corrcoef(valid_pred, valid_actual)
            
            if correlation_matrix.shape != (2, 2):
                logger.warning("Invalid correlation matrix shape")
                return 0.0
            
            correlation = correlation_matrix[0, 1]
            
            # Validate result
            if np.isnan(correlation) or np.isinf(correlation):
                logger.warning("Invalid correlation calculated")
                return 0.0
            
            # Correlation should be between -1 and 1
            correlation = np.clip(correlation, -1.0, 1.0)
            
            return float(correlation)
            
        except Exception as e:
            logger.warning(f"Correlation matrix calculation failed: {e}")
            # Fallback to manual calculation
            try:
                correlation = np.sum((valid_pred - np.mean(valid_pred)) * (valid_actual - np.mean(valid_actual))) / \
                             (np.sqrt(np.sum((valid_pred - np.mean(valid_pred))**2)) * 
                              np.sqrt(np.sum((valid_actual - np.mean(valid_actual))**2)))
                
                if np.isnan(correlation) or np.isinf(correlation):
                    return 0.0
                
                return float(np.clip(correlation, -1.0, 1.0))
                
            except Exception as e2:
                logger.error(f"Manual correlation calculation also failed: {e2}")
                return 0.0
        
    except Exception as e:
        logger.error(f"Correlation calculation failed: {e}")
        return 0.0


# Helper functions

def _extract_strategies(data: np.ndarray) -> np.ndarray:
    """Extract strategy names from data."""
    strategies = []
    for item in data:
        if isinstance(item, str) and '_' in item:
            strategy = item.split('_')[0]
            strategies.append(strategy)
        else:
            strategies.append(str(item))
    return np.array(strategies)


def _extract_returns(data: np.ndarray) -> np.ndarray:
    """Extract return values from data."""
    returns = []
    for item in data:
        if isinstance(item, str) and '_' in item:
            try:
                ret = float(item.split('_')[1])
                returns.append(ret)
            except (IndexError, ValueError, TypeError) as exc:
                logger.debug("Failed to parse return from %s: %s", item, exc)
                returns.append(0.0)
        else:
            returns.append(0.0)
    return np.array(returns)


# Helper and validation functions

def _validate_evaluation_inputs(predictions: np.ndarray, actuals: np.ndarray):
    """Validate inputs for evaluation functions."""
    if predictions is None or actuals is None:
        raise ValidationError("Predictions and actuals cannot be None")
    
    if len(predictions) == 0 or len(actuals) == 0:
        raise ValidationError("Predictions and actuals cannot be empty")
    
    if len(predictions) != len(actuals):
        raise ValidationError(f"Predictions and actuals length mismatch: {len(predictions)} vs {len(actuals)}")
    
    # Check for basic data types
    try:
        np.array(predictions)
        np.array(actuals)
    except Exception as e:
        raise ValidationError(f"Cannot convert inputs to arrays: {str(e)}")


def _validate_array_inputs(pred: np.ndarray, actual: np.ndarray, metric_name: str):
    """Validate array inputs for metric calculations."""
    if pred is None or actual is None:
        raise ValidationError(f"Arrays cannot be None for {metric_name}")
    
    if len(pred) != len(actual):
        raise ValidationError(f"Array length mismatch for {metric_name}: {len(pred)} vs {len(actual)}")
    
    # Convert to numpy arrays if needed
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(actual, np.ndarray):
        actual = np.array(actual)


def _validate_numeric_value(value, value_name: str) -> float:
    """Validate and sanitize a numeric value."""
    if value is None:
        raise ValidationError(f"{value_name} cannot be None")
    
    try:
        numeric_value = float(value)
        
        if np.isnan(numeric_value) or np.isinf(numeric_value):
            raise ValidationError(f"{value_name} is NaN or infinite: {numeric_value}")
        
        # Check for reasonable bounds
        if abs(numeric_value) > 10000:
            logger.warning(f"{value_name} has extreme value: {numeric_value}")
            return np.clip(numeric_value, -10000, 10000)
        
        return numeric_value
        
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Cannot convert {value_name} to numeric: {str(e)}")


def _validate_metric_value(value: float, metric_name: str, min_val: float, max_val: float) -> float:
    """Validate and clip metric values to reasonable bounds."""
    try:
        if value is None:
            logger.warning(f"{metric_name} is None, using 0")
            return 0.0
        
        if np.isnan(value) or np.isinf(value):
            logger.warning(f"{metric_name} is NaN/infinite ({value}), using 0")
            return 0.0
        
        # Clip to bounds
        if value < min_val:
            logger.debug(f"{metric_name} below minimum ({value} < {min_val}), clipping")
            return min_val
        
        if value > max_val:
            logger.debug(f"{metric_name} above maximum ({value} > {max_val}), clipping")
            return max_val
        
        return float(value)
        
    except Exception as e:
        logger.warning(f"Failed to validate {metric_name}: {e}")
        return 0.0


def _validate_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Validate all metrics in a dictionary."""
    validated = {}
    
    for key, value in metrics.items():
        try:
            if key in ['accuracy', 'precision', 'recall', 'f1_score']:
                # Classification metrics should be 0-1
                validated[key] = _validate_metric_value(value, key, 0.0, 1.0)
            elif key == 'mean_absolute_error':
                # MAE should be positive
                validated[key] = _validate_metric_value(value, key, 0.0, 1000.0)
            elif key == 'r_squared':
                # R-squared can be negative but cap at reasonable bounds
                validated[key] = _validate_metric_value(value, key, -10.0, 1.0)
            elif key == 'backtest_correlation':
                # Correlation should be -1 to 1
                validated[key] = _validate_metric_value(value, key, -1.0, 1.0)
            else:
                # Generic validation for other metrics
                validated[key] = _validate_metric_value(value, key, -1000.0, 1000.0)
                
        except Exception as e:
            logger.warning(f"Failed to validate metric {key}: {e}")
            validated[key] = 0.0
    
    return validated


def _create_default_performance(total_predictions: int) -> ModelPerformance:
    """Create default performance metrics when evaluation fails."""
    logger.info("Creating default performance metrics due to evaluation failure")
    
    return ModelPerformance(
        accuracy=0.0,
        precision=0.0,
        recall=0.0,
        f1_score=0.0,
        mean_absolute_error=100.0,  # High error to indicate problem
        r_squared=0.0,
        backtest_correlation=0.0,
        total_predictions=total_predictions,
        successful_predictions=0
    )


def validate_model_performance(performance: ModelPerformance) -> ModelPerformance:
    """Validate and sanitize a ModelPerformance object."""
    try:
        if performance is None:
            return _create_default_performance(0)
        
        # Validate each field
        validated_metrics = {
            'accuracy': performance.accuracy,
            'precision': performance.precision,
            'recall': performance.recall,
            'f1_score': performance.f1_score,
            'mean_absolute_error': performance.mean_absolute_error,
            'r_squared': performance.r_squared,
            'backtest_correlation': performance.backtest_correlation
        }
        
        validated = _validate_metrics(validated_metrics)
        
        # Validate integer fields
        total_predictions = max(0, int(performance.total_predictions)) if performance.total_predictions is not None else 0
        successful_predictions = max(0, int(performance.successful_predictions)) if performance.successful_predictions is not None else 0
        
        # Ensure successful <= total
        if successful_predictions > total_predictions:
            logger.warning(f"Successful predictions ({successful_predictions}) > total ({total_predictions}), fixing")
            successful_predictions = total_predictions
        
        return ModelPerformance(
            accuracy=validated['accuracy'],
            precision=validated['precision'],
            recall=validated['recall'],
            f1_score=validated['f1_score'],
            mean_absolute_error=validated['mean_absolute_error'],
            r_squared=validated['r_squared'],
            backtest_correlation=validated['backtest_correlation'],
            total_predictions=total_predictions,
            successful_predictions=successful_predictions
        )
        
    except Exception as e:
        logger.error(f"Failed to validate ModelPerformance: {e}")
        return _create_default_performance(0)


def evaluate_single_prediction(prediction: Dict, actual: Dict) -> Dict[str, float]:
    """Evaluate a single prediction with comprehensive error handling."""
    try:
        if prediction is None or actual is None:
            raise ValidationError("Prediction and actual cannot be None")
        
        return calculate_metrics(prediction, actual)
        
    except Exception as e:
        logger.error(f"Single prediction evaluation failed: {e}")
        return {'error': 1.0, 'accuracy': 0.0}


def get_evaluation_summary(performance: ModelPerformance) -> Dict[str, any]:
    """Get a summary of model performance with health indicators."""
    try:
        performance = validate_model_performance(performance)
        
        # Calculate overall health score (0-100)
        health_score = (
            performance.accuracy * 30 +          # 30% weight on accuracy
            performance.f1_score * 25 +          # 25% weight on F1
            (1 - min(performance.mean_absolute_error / 100, 1)) * 20 +  # 20% weight on MAE (inverted)
            max(performance.r_squared, 0) * 15 + # 15% weight on R² (only positive)
            (performance.backtest_correlation + 1) / 2 * 10  # 10% weight on correlation (normalized)
        )
        
        # Determine health status
        if health_score >= 70:
            status = "Excellent"
        elif health_score >= 50:
            status = "Good"
        elif health_score >= 30:
            status = "Fair"
        else:
            status = "Poor"
        
        return {
            'health_score': round(health_score, 2),
            'status': status,
            'total_predictions': performance.total_predictions,
            'success_rate': performance.successful_predictions / max(performance.total_predictions, 1),
            'key_metrics': {
                'accuracy': performance.accuracy,
                'f1_score': performance.f1_score,
                'mae': performance.mean_absolute_error,
                'r_squared': performance.r_squared
            },
            'warnings': _get_performance_warnings(performance)
        }
        
    except Exception as e:
        logger.error(f"Failed to create evaluation summary: {e}")
        return {
            'health_score': 0.0,
            'status': 'Error',
            'total_predictions': 0,
            'success_rate': 0.0,
            'error': str(e)
        }


def _get_performance_warnings(performance: ModelPerformance) -> List[str]:
    """Get list of performance warnings."""
    warnings = []
    
    if performance.accuracy < 0.3:
        warnings.append("Very low accuracy - model may need retraining")
    
    if performance.mean_absolute_error > 50:
        warnings.append("High prediction errors - check data quality")
    
    if performance.r_squared < 0:
        warnings.append("Negative R² - model performing worse than naive baseline")
    
    if performance.total_predictions < 10:
        warnings.append("Very few predictions - results may not be reliable")
    
    if performance.successful_predictions == 0:
        warnings.append("No successful predictions - model may be broken")
    
    return warnings
