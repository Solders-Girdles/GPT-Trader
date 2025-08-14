"""
ML Pipeline Logging Integration
Phase 3, Week 7: Operational Excellence

This module provides integration utilities for enhanced structured logging
with existing ML components, ensuring seamless adoption and minimal code changes.
"""

from functools import wraps
from typing import Any, Dict, Optional, Union
import time
import numpy as np
import pandas as pd

from .structured_logger import (
    get_logger,
    SpanType,
    traced_operation,
    correlation_id
)


class MLLoggingMixin:
    """Mixin class to add structured logging to ML components."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(f"ml.{self.__class__.__name__.lower()}")
    
    def log_operation_start(self, operation: str, **kwargs):
        """Log the start of an ML operation."""
        self.logger.info(
            f"Starting {operation}",
            operation=operation,
            component=self.__class__.__name__,
            **kwargs
        )
    
    def log_operation_end(self, operation: str, duration_ms: float, success: bool = True, **kwargs):
        """Log the end of an ML operation."""
        level = "info" if success else "error"
        getattr(self.logger, level)(
            f"{'Completed' if success else 'Failed'} {operation} in {duration_ms:.2f}ms",
            operation=operation,
            component=self.__class__.__name__,
            duration_ms=duration_ms,
            success=success,
            **kwargs
        )
    
    def log_metric(self, metric_name: str, value: Union[float, int], **kwargs):
        """Log an ML metric."""
        self.logger.metric(
            f"Metric {metric_name}: {value}",
            value=value,
            metric_name=metric_name,
            component=self.__class__.__name__,
            **kwargs
        )
    
    def log_model_performance(self, metrics: Dict[str, float], **kwargs):
        """Log model performance metrics."""
        for metric_name, value in metrics.items():
            self.log_metric(metric_name, value, **kwargs)
        
        self.logger.info(
            "Model performance logged",
            operation="performance_evaluation",
            component=self.__class__.__name__,
            attributes=metrics,
            **kwargs
        )


def log_ml_operation(
    operation_name: Optional[str] = None,
    span_type: SpanType = SpanType.ML_PREDICTION,
    log_inputs: bool = False,
    log_outputs: bool = False,
    log_performance: bool = True
):
    """
    Decorator for logging ML operations with minimal overhead.
    
    Args:
        operation_name: Name of the operation (defaults to function name)
        span_type: Type of span for tracing
        log_inputs: Whether to log input parameters
        log_outputs: Whether to log outputs
        log_performance: Whether to log performance metrics
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get logger (create if doesn't exist)
            if not hasattr(self, 'logger'):
                self.logger = get_logger(f"ml.{self.__class__.__name__.lower()}")
            
            op_name = operation_name or func.__name__
            
            with self.logger.start_span(op_name, span_type):
                # Log operation start
                log_kwargs = {
                    "operation": op_name,
                    "component": self.__class__.__name__,
                }
                
                if log_inputs:
                    log_kwargs["attributes"] = {
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    }
                
                self.logger.info(f"Starting {op_name}", **log_kwargs)
                
                # Execute operation with timing
                start_time = time.perf_counter()
                try:
                    result = func(self, *args, **kwargs)
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    
                    # Log successful completion
                    log_kwargs = {
                        "operation": op_name,
                        "component": self.__class__.__name__,
                        "duration_ms": duration_ms,
                        "success": True
                    }
                    
                    if log_outputs and result is not None:
                        if isinstance(result, (np.ndarray, pd.DataFrame)):
                            log_kwargs["attributes"] = {
                                "output_type": type(result).__name__,
                                "output_shape": getattr(result, 'shape', None)
                            }
                        elif isinstance(result, (dict, list)):
                            log_kwargs["attributes"] = {
                                "output_type": type(result).__name__,
                                "output_size": len(result)
                            }
                        else:
                            log_kwargs["attributes"] = {
                                "output_type": type(result).__name__,
                                "output_value": str(result)[:100]
                            }
                    
                    if log_performance:
                        # Log performance warning if slow
                        if duration_ms > 1000:
                            self.logger.warning(
                                f"Slow operation {op_name} took {duration_ms:.2f}ms",
                                **log_kwargs
                            )
                        else:
                            self.logger.info(f"Completed {op_name}", **log_kwargs)
                    
                    return result
                
                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    
                    # Log error
                    self.logger.error(
                        f"Error in {op_name}: {e}",
                        operation=op_name,
                        component=self.__class__.__name__,
                        duration_ms=duration_ms,
                        success=False,
                        attributes={"error_type": type(e).__name__}
                    )
                    raise
        
        return wrapper
    return decorator


class LoggedFeatureEngineer:
    """Example integration with feature engineering component."""
    
    def __init__(self):
        self.logger = get_logger("ml.feature_engineer")
    
    @log_ml_operation("feature_engineering", SpanType.ML_TRAINING, log_performance=True)
    def engineer_features(self, data: pd.DataFrame, feature_config: Dict[str, Any]) -> pd.DataFrame:
        """Engineer features with comprehensive logging."""
        self.logger.info(
            "Feature engineering started",
            operation="feature_engineering",
            attributes={
                "input_shape": data.shape,
                "input_columns": len(data.columns),
                "feature_types": list(feature_config.keys())
            }
        )
        
        # Simulate feature engineering
        time.sleep(0.01)  # Simulate computation
        
        # Create dummy features
        features = data.copy()
        for feature_type, config in feature_config.items():
            if feature_type == "technical_indicators":
                features[f"sma_{config.get('window', 20)}"] = data['close'].rolling(
                    window=config.get('window', 20)
                ).mean()
            elif feature_type == "volatility":
                features[f"volatility_{config.get('window', 20)}"] = data['close'].pct_change().rolling(
                    window=config.get('window', 20)
                ).std()
        
        self.logger.info(
            "Feature engineering completed",
            operation="feature_engineering",
            attributes={
                "output_shape": features.shape,
                "features_created": len(features.columns) - len(data.columns),
                "total_features": len(features.columns)
            }
        )
        
        return features


class LoggedModelTrainer:
    """Example integration with model training component."""
    
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.logger = get_logger("ml.model_trainer")
        self.model = None
    
    @log_ml_operation("model_training", SpanType.ML_TRAINING, log_performance=True)
    def train(self, X: pd.DataFrame, y: pd.Series, **training_params) -> Dict[str, float]:
        """Train model with comprehensive logging."""
        with self.logger.correlation_context() as corr_id:
            self.logger.info(
                "Model training started",
                operation="model_training",
                model_type=self.model_type,
                attributes={
                    "training_samples": len(X),
                    "features": len(X.columns),
                    "target_distribution": {
                        "mean": float(y.mean()),
                        "std": float(y.std()),
                        "min": float(y.min()),
                        "max": float(y.max())
                    },
                    "training_params": training_params
                }
            )
            
            # Simulate training phases
            phases = ["data_validation", "hyperparameter_tuning", "model_fitting", "validation"]
            
            for phase in phases:
                with self.logger.start_span(f"training_{phase}", SpanType.ML_TRAINING):
                    self.logger.info(
                        f"Training phase: {phase}",
                        operation=f"training_{phase}",
                        model_type=self.model_type
                    )
                    time.sleep(0.02)  # Simulate phase computation
            
            # Simulate model performance
            metrics = {
                "accuracy": 0.68 + np.random.normal(0, 0.05),
                "precision": 0.70 + np.random.normal(0, 0.03),
                "recall": 0.65 + np.random.normal(0, 0.04),
                "f1_score": 0.67 + np.random.normal(0, 0.03),
                "auc_roc": 0.75 + np.random.normal(0, 0.02)
            }
            
            # Log individual metrics
            for metric_name, value in metrics.items():
                self.logger.metric(
                    f"Training metric: {metric_name}",
                    value=value,
                    metric_name=metric_name,
                    model_type=self.model_type,
                    tags={"phase": "training", "dataset": "train"}
                )
            
            self.logger.audit(
                "Model training completed",
                operation="model_training",
                model_type=self.model_type,
                correlation_id=corr_id,
                attributes={
                    "final_metrics": metrics,
                    "model_size": "2.5MB",
                    "training_duration": "45.2s"
                }
            )
            
            return metrics


class LoggedRiskCalculator:
    """Example integration with risk calculation component."""
    
    def __init__(self):
        self.logger = get_logger("ml.risk_calculator")
    
    @log_ml_operation("var_calculation", SpanType.RISK_CALCULATION, log_performance=True)
    def calculate_var(
        self,
        portfolio_returns: np.ndarray,
        confidence: float = 0.95,
        method: str = "historical"
    ) -> Dict[str, float]:
        """Calculate VaR with logging."""
        self.logger.info(
            "VaR calculation started",
            operation="var_calculation",
            attributes={
                "method": method,
                "confidence": confidence,
                "data_points": len(portfolio_returns),
                "return_stats": {
                    "mean": float(np.mean(portfolio_returns)),
                    "std": float(np.std(portfolio_returns)),
                    "min": float(np.min(portfolio_returns)),
                    "max": float(np.max(portfolio_returns))
                }
            }
        )
        
        # Calculate VaR
        if method == "historical":
            var_value = np.percentile(portfolio_returns, (1 - confidence) * 100)
        else:
            # Parametric method
            mean_return = np.mean(portfolio_returns)
            std_return = np.std(portfolio_returns)
            from scipy.stats import norm
            var_value = mean_return + norm.ppf(1 - confidence) * std_return
        
        # Calculate CVaR (Expected Shortfall)
        cvar_value = np.mean(portfolio_returns[portfolio_returns <= var_value])
        
        results = {
            "var": float(var_value),
            "cvar": float(cvar_value),
            "confidence": confidence
        }
        
        # Log risk metrics
        for metric_name, value in results.items():
            self.logger.metric(
                f"Risk metric: {metric_name}",
                value=value,
                metric_name=metric_name,
                tags={"method": method, "confidence": str(confidence)}
            )
        
        return results


class LoggedBacktestEngine:
    """Example integration with backtest engine."""
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.logger = get_logger("ml.backtest_engine")
    
    @log_ml_operation("backtest_run", SpanType.BACKTEST, log_performance=True)
    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        symbols: list,
        initial_capital: float = 100000
    ) -> Dict[str, Any]:
        """Run backtest with comprehensive logging."""
        backtest_id = f"BT-{int(time.time())}"
        
        with self.logger.correlation_context() as corr_id:
            self.logger.audit(
                "Backtest started",
                operation="backtest_run",
                backtest_id=backtest_id,
                strategy=self.strategy_name,
                attributes={
                    "start_date": start_date,
                    "end_date": end_date,
                    "symbols": symbols,
                    "initial_capital": initial_capital
                }
            )
            
            # Simulate backtest phases
            phases = [
                ("data_loading", 0.1),
                ("signal_generation", 0.2),
                ("trade_execution", 0.15),
                ("performance_calculation", 0.05)
            ]
            
            total_trades = 0
            for phase_name, duration in phases:
                with self.logger.start_span(f"backtest_{phase_name}", SpanType.BACKTEST):
                    self.logger.info(
                        f"Backtest phase: {phase_name}",
                        operation=f"backtest_{phase_name}",
                        backtest_id=backtest_id,
                        strategy=self.strategy_name
                    )
                    time.sleep(duration)
                    
                    if phase_name == "trade_execution":
                        # Simulate trades
                        trades_this_phase = np.random.randint(10, 50)
                        total_trades += trades_this_phase
                        
                        for i in range(trades_this_phase):
                            self.logger.audit(
                                "Trade executed",
                                trade_id=f"TRD-{backtest_id}-{i:03d}",
                                backtest_id=backtest_id,
                                symbol=np.random.choice(symbols),
                                side=np.random.choice(["BUY", "SELL"]),
                                quantity=np.random.randint(1, 1000),
                                price=np.random.uniform(50, 500)
                            )
            
            # Generate final results
            results = {
                "backtest_id": backtest_id,
                "total_return": np.random.uniform(0.05, 0.25),
                "sharpe_ratio": np.random.uniform(0.8, 2.0),
                "max_drawdown": np.random.uniform(0.05, 0.20),
                "total_trades": total_trades,
                "win_rate": np.random.uniform(0.45, 0.75),
                "profit_factor": np.random.uniform(1.1, 2.5)
            }
            
            # Log performance metrics
            for metric_name, value in results.items():
                if isinstance(value, (int, float)) and metric_name != "total_trades":
                    self.logger.metric(
                        f"Backtest metric: {metric_name}",
                        value=value,
                        metric_name=metric_name,
                        strategy=self.strategy_name,
                        tags={"backtest_id": backtest_id}
                    )
            
            self.logger.audit(
                "Backtest completed",
                operation="backtest_run",
                backtest_id=backtest_id,
                strategy=self.strategy_name,
                correlation_id=corr_id,
                attributes=results
            )
            
            return results


# Utility functions for common ML logging patterns
def log_data_quality_check(data: pd.DataFrame, component_name: str = "data_quality"):
    """Log data quality metrics."""
    logger = get_logger(f"ml.{component_name}")
    
    quality_metrics = {
        "total_rows": len(data),
        "total_columns": len(data.columns),
        "missing_values": data.isnull().sum().sum(),
        "duplicate_rows": data.duplicated().sum(),
        "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    # Calculate missing value percentage by column
    missing_by_column = (data.isnull().sum() / len(data) * 100).to_dict()
    
    logger.info(
        "Data quality check completed",
        operation="data_quality_check",
        attributes={
            **quality_metrics,
            "missing_by_column": {k: v for k, v in missing_by_column.items() if v > 0}
        }
    )
    
    # Log warnings for data quality issues
    if quality_metrics["missing_values"] > 0:
        logger.warning(
            f"Found {quality_metrics['missing_values']} missing values",
            operation="data_quality_check",
            attributes={"missing_percentage": quality_metrics["missing_values"] / (len(data) * len(data.columns)) * 100}
        )
    
    if quality_metrics["duplicate_rows"] > 0:
        logger.warning(
            f"Found {quality_metrics['duplicate_rows']} duplicate rows",
            operation="data_quality_check",
            attributes={"duplicate_percentage": quality_metrics["duplicate_rows"] / len(data) * 100}
        )
    
    return quality_metrics


def log_model_deployment(model_id: str, model_metadata: Dict[str, Any], component_name: str = "model_deployment"):
    """Log model deployment event."""
    logger = get_logger(f"ml.{component_name}")
    
    logger.audit(
        "Model deployed to production",
        operation="model_deployment",
        model_id=model_id,
        attributes=model_metadata,
        tags={"environment": "production", "deployment_type": "ml_model"}
    )


def log_prediction_batch(
    predictions: np.ndarray,
    model_id: str,
    batch_id: str,
    component_name: str = "prediction_service"
):
    """Log batch prediction results."""
    logger = get_logger(f"ml.{component_name}")
    
    prediction_stats = {
        "batch_size": len(predictions),
        "mean_prediction": float(np.mean(predictions)),
        "std_prediction": float(np.std(predictions)),
        "min_prediction": float(np.min(predictions)),
        "max_prediction": float(np.max(predictions))
    }
    
    logger.info(
        "Batch predictions completed",
        operation="batch_prediction",
        model_id=model_id,
        batch_id=batch_id,
        attributes=prediction_stats
    )
    
    # Log metric for monitoring
    logger.metric(
        "Batch prediction throughput",
        value=len(predictions),
        metric_name="batch_size",
        tags={"model_id": model_id, "batch_id": batch_id}
    )


# Export public interface
__all__ = [
    'MLLoggingMixin',
    'log_ml_operation',
    'LoggedFeatureEngineer',
    'LoggedModelTrainer',
    'LoggedRiskCalculator',
    'LoggedBacktestEngine',
    'log_data_quality_check',
    'log_model_deployment',
    'log_prediction_batch'
]