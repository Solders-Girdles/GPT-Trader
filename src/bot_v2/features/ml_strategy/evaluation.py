"""
Model evaluation utilities - LOCAL to this slice.
"""

import numpy as np
from typing import Dict, List
from .types import ModelPerformance


def evaluate_predictions(predictions: np.ndarray, actuals: np.ndarray) -> ModelPerformance:
    """
    Evaluate model predictions against actual results.
    
    Returns comprehensive performance metrics.
    """
    # Convert to comparable format
    pred_strategies = _extract_strategies(predictions)
    actual_strategies = _extract_strategies(actuals)
    
    # Calculate classification metrics
    accuracy = calculate_accuracy(pred_strategies, actual_strategies)
    precision = calculate_precision(pred_strategies, actual_strategies)
    recall = calculate_recall(pred_strategies, actual_strategies)
    f1 = calculate_f1_score(precision, recall)
    
    # Calculate regression metrics for returns
    pred_returns = _extract_returns(predictions)
    actual_returns = _extract_returns(actuals)
    
    mae = calculate_mae(pred_returns, actual_returns)
    r2 = calculate_r_squared(pred_returns, actual_returns)
    correlation = calculate_correlation(pred_returns, actual_returns)
    
    # Count successes
    successful = np.sum(pred_strategies == actual_strategies)
    
    return ModelPerformance(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        mean_absolute_error=mae,
        r_squared=r2,
        backtest_correlation=correlation,
        total_predictions=len(predictions),
        successful_predictions=int(successful)
    )


def calculate_metrics(predictions: Dict, actuals: Dict) -> Dict[str, float]:
    """Calculate various performance metrics."""
    metrics = {}
    
    # Strategy selection accuracy
    if 'strategy' in predictions and 'strategy' in actuals:
        metrics['strategy_accuracy'] = float(
            predictions['strategy'] == actuals['strategy']
        )
    
    # Return prediction error
    if 'return' in predictions and 'return' in actuals:
        metrics['return_mae'] = abs(predictions['return'] - actuals['return'])
        metrics['return_mse'] = (predictions['return'] - actuals['return']) ** 2
    
    # Sharpe prediction error
    if 'sharpe' in predictions and 'sharpe' in actuals:
        metrics['sharpe_mae'] = abs(predictions['sharpe'] - actuals['sharpe'])
    
    # Drawdown prediction error
    if 'drawdown' in predictions and 'drawdown' in actuals:
        metrics['drawdown_mae'] = abs(predictions['drawdown'] - actuals['drawdown'])
    
    return metrics


# Metric calculation functions

def calculate_accuracy(pred: np.ndarray, actual: np.ndarray) -> float:
    """Calculate classification accuracy."""
    if len(pred) == 0:
        return 0.0
    return np.mean(pred == actual)


def calculate_precision(pred: np.ndarray, actual: np.ndarray) -> float:
    """Calculate precision score (simplified for multi-class)."""
    if len(pred) == 0:
        return 0.0
    
    # Average precision across all classes
    unique_classes = np.unique(np.concatenate([pred, actual]))
    precisions = []
    
    for cls in unique_classes:
        true_positives = np.sum((pred == cls) & (actual == cls))
        predicted_positives = np.sum(pred == cls)
        
        if predicted_positives > 0:
            precisions.append(true_positives / predicted_positives)
    
    return np.mean(precisions) if precisions else 0.0


def calculate_recall(pred: np.ndarray, actual: np.ndarray) -> float:
    """Calculate recall score (simplified for multi-class)."""
    if len(actual) == 0:
        return 0.0
    
    # Average recall across all classes
    unique_classes = np.unique(np.concatenate([pred, actual]))
    recalls = []
    
    for cls in unique_classes:
        true_positives = np.sum((pred == cls) & (actual == cls))
        actual_positives = np.sum(actual == cls)
        
        if actual_positives > 0:
            recalls.append(true_positives / actual_positives)
    
    return np.mean(recalls) if recalls else 0.0


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_mae(pred: np.ndarray, actual: np.ndarray) -> float:
    """Calculate mean absolute error."""
    if len(pred) == 0:
        return 0.0
    return np.mean(np.abs(pred - actual))


def calculate_r_squared(pred: np.ndarray, actual: np.ndarray) -> float:
    """Calculate R-squared value."""
    if len(actual) == 0:
        return 0.0
    
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def calculate_correlation(pred: np.ndarray, actual: np.ndarray) -> float:
    """Calculate correlation coefficient."""
    if len(pred) < 2:
        return 0.0
    
    # Handle edge cases
    if np.std(pred) == 0 or np.std(actual) == 0:
        return 0.0
    
    correlation_matrix = np.corrcoef(pred, actual)
    return correlation_matrix[0, 1]


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
            except:
                returns.append(0.0)
        else:
            returns.append(0.0)
    return np.array(returns)