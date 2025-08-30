"""
ML Confidence Filtering System
Phase 5: ML Integration - High Confidence Trading

Implements advanced confidence scoring and filtering to reduce false signals 
and overtrading, targeting:
- Reduce trades from 150/year to 30-50/year  
- Increase win rate from 28% to 55%+
- Reduce transaction costs by 70%
- Improve Sharpe ratio by 0.5+
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore")

# Import existing components
try:
    from src.bot.ml.ensemble_manager import EnsembleManager
    from src.bot.ml.model_calibrator import ModelCalibrator, CalibrationConfig
    from src.bot.ml.features.market_regime import MarketRegimeDetector
except ImportError:
    # Fallback for testing
    EnsembleManager = None
    ModelCalibrator = None
    MarketRegimeDetector = None

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceConfig:
    """Configuration for confidence filtering system"""
    
    # Base confidence thresholds
    base_confidence_threshold: float = 0.7
    min_confidence_threshold: float = 0.6
    max_confidence_threshold: float = 0.9
    
    # Multi-level filtering
    enable_model_confidence: bool = True
    enable_ensemble_agreement: bool = True
    enable_regime_confidence: bool = True
    enable_performance_confidence: bool = True
    
    # Trade frequency control
    target_trades_per_year: int = 40
    max_trades_per_year: int = 50
    min_trades_per_year: int = 20
    
    # Performance tracking
    lookback_periods: int = 50
    adaptation_rate: float = 0.1
    performance_window: int = 20
    
    # Risk parameters
    max_position_size: float = 0.15
    kelly_fraction: float = 0.25
    transaction_cost: float = 0.002  # 0.2% per trade
    
    # Calibration settings
    calibration_method: str = "isotonic"
    recalibration_frequency: int = 100  # trades


@dataclass
class ConfidenceMetrics:
    """Confidence metrics for a prediction"""
    
    model_confidence: float = 0.0
    ensemble_agreement: float = 0.0
    regime_confidence: float = 0.0
    performance_confidence: float = 0.0
    
    overall_confidence: float = 0.0
    should_trade: bool = False
    position_size_multiplier: float = 0.0
    
    # Supporting data
    prediction_variance: float = 0.0
    market_regime: str = "unknown"
    recent_accuracy: float = 0.0
    trade_frequency_adjustment: float = 1.0


class ModelConfidenceCalculator:
    """Calculate confidence scores for individual model predictions"""
    
    def __init__(self):
        self.prediction_history = []
        self.accuracy_history = []
        
    def calculate_prediction_confidence(
        self, 
        model: BaseEstimator, 
        features: np.ndarray, 
        method: str = 'entropy'
    ) -> np.ndarray:
        """
        Calculate confidence scores for predictions
        
        Args:
            model: Trained model
            features: Input features
            method: Confidence calculation method
            
        Returns:
            Confidence scores for each prediction
        """
        if hasattr(model, 'predict_proba'):
            # Classification model
            probabilities = model.predict_proba(features)
            
            if method == 'entropy':
                # Shannon entropy (lower = more confident)
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
                max_entropy = np.log(probabilities.shape[1])
                confidence = 1 - (entropy / max_entropy)
                # Ensure confidence is in [0, 1] range
                confidence = np.clip(confidence, 0, 1)
                
            elif method == 'max_prob':
                # Maximum probability
                confidence = np.max(probabilities, axis=1)
                
            elif method == 'margin':
                # Difference between top 2 probabilities
                sorted_probs = np.sort(probabilities, axis=1)
                if probabilities.shape[1] >= 2:
                    confidence = sorted_probs[:, -1] - sorted_probs[:, -2]
                else:
                    confidence = sorted_probs[:, -1]
                    
        else:
            # Regression model or models without predict_proba
            if hasattr(model, 'estimators_'):  # Random Forest or similar
                predictions = np.array([tree.predict(features) for tree in model.estimators_])
                # Confidence = inverse of prediction variance
                pred_std = np.std(predictions, axis=0)
                confidence = 1 / (1 + pred_std)
            else:
                # Default: use absolute prediction magnitude
                predictions = model.predict(features)
                pred_mean = np.mean(np.abs(predictions))
                confidence = np.abs(predictions) / (pred_mean + 1e-10)
                confidence = np.clip(confidence, 0, 1)
        
        return confidence
    
    def calculate_temporal_confidence(
        self, 
        recent_predictions: List[float], 
        recent_outcomes: List[bool]
    ) -> float:
        """
        Calculate confidence based on recent prediction accuracy
        
        Args:
            recent_predictions: Recent model predictions
            recent_outcomes: Whether predictions were correct
            
        Returns:
            Temporal confidence score
        """
        if len(recent_outcomes) < 5:
            return 0.5  # Neutral confidence for insufficient data
            
        # Calculate recent accuracy
        recent_accuracy = np.mean(recent_outcomes)
        
        # Calculate trend in accuracy
        if len(recent_outcomes) >= 10:
            half_point = len(recent_outcomes) // 2
            early_accuracy = np.mean(recent_outcomes[:half_point])
            late_accuracy = np.mean(recent_outcomes[half_point:])
            trend = late_accuracy - early_accuracy
        else:
            trend = 0
            
        # Combine accuracy and trend
        base_confidence = recent_accuracy
        trend_adjustment = trend * 0.5  # Moderate trend impact
        
        temporal_confidence = base_confidence + trend_adjustment
        return np.clip(temporal_confidence, 0, 1)


class EnsembleConfidenceCalculator:
    """Calculate confidence from ensemble model agreement"""
    
    def __init__(self):
        self.model_weights = {}
        
    def calculate_ensemble_agreement(
        self, 
        model_predictions: Dict[str, np.ndarray],
        model_confidences: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Calculate confidence from model agreement
        
        Args:
            model_predictions: Predictions from each model
            model_confidences: Individual model confidence scores
            
        Returns:
            Ensemble agreement scores
        """
        if not model_predictions:
            return np.array([0.5])
            
        # Stack predictions
        pred_matrix = np.column_stack(list(model_predictions.values()))
        n_samples, n_models = pred_matrix.shape
        
        if n_models == 1:
            return np.ones(n_samples) * 0.5
            
        # Method 1: Prediction variance (inverse relationship with confidence)
        pred_variance = np.var(pred_matrix, axis=1)
        variance_confidence = 1 / (1 + pred_variance)
        
        # Method 2: Pairwise correlation
        correlations = []
        for i in range(n_samples):
            sample_preds = pred_matrix[i, :]
            if np.std(sample_preds) > 1e-10:  # Avoid division by zero
                # Calculate coefficient of variation (lower = more agreement)
                cv = np.std(sample_preds) / (np.mean(np.abs(sample_preds)) + 1e-10)
                correlations.append(1 / (1 + cv))
            else:
                correlations.append(1.0)  # Perfect agreement
                
        correlation_confidence = np.array(correlations)
        
        # Method 3: Majority vote strength
        # For classification: strength of majority
        # For regression: clustering around median
        majority_confidence = []
        for i in range(n_samples):
            sample_preds = pred_matrix[i, :]
            if len(np.unique(sample_preds)) == 1:
                # All models agree exactly
                majority_confidence.append(1.0)
            else:
                # Calculate how close predictions are to median
                median_pred = np.median(sample_preds)
                deviations = np.abs(sample_preds - median_pred)
                avg_deviation = np.mean(deviations)
                conf = 1 / (1 + avg_deviation)
                majority_confidence.append(conf)
                
        majority_confidence = np.array(majority_confidence)
        
        # Combine methods
        if model_confidences:
            # Weight by individual model confidences
            conf_matrix = np.column_stack(list(model_confidences.values()))
            avg_model_confidence = np.mean(conf_matrix, axis=1)
            
            # Weighted combination
            ensemble_agreement = (
                0.3 * variance_confidence +
                0.3 * correlation_confidence +
                0.2 * majority_confidence +
                0.2 * avg_model_confidence
            )
        else:
            # Equal weighting
            ensemble_agreement = (
                0.4 * variance_confidence +
                0.3 * correlation_confidence +
                0.3 * majority_confidence
            )
            
        return np.clip(ensemble_agreement, 0, 1)
    
    def update_model_weights(
        self, 
        model_performances: Dict[str, float],
        decay_factor: float = 0.95
    ):
        """
        Update model weights based on recent performance
        
        Args:
            model_performances: Recent performance for each model
            decay_factor: Decay factor for historical weights
        """
        # Decay existing weights
        for model_id in self.model_weights:
            self.model_weights[model_id] *= decay_factor
            
        # Update with new performances
        total_performance = sum(model_performances.values())
        if total_performance > 0:
            for model_id, performance in model_performances.items():
                normalized_perf = performance / total_performance
                if model_id in self.model_weights:
                    self.model_weights[model_id] += normalized_perf * (1 - decay_factor)
                else:
                    self.model_weights[model_id] = normalized_perf
                    
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for model_id in self.model_weights:
                self.model_weights[model_id] /= total_weight


class RegimeConfidenceCalculator:
    """Calculate confidence based on market regime analysis"""
    
    def __init__(self):
        self.regime_performance = {}
        self.current_regime = "unknown"
        
    def calculate_regime_confidence(
        self, 
        market_data: pd.DataFrame,
        model_type: str = "trend_following"
    ) -> Tuple[float, str]:
        """
        Calculate confidence based on market regime favorability
        
        Args:
            market_data: Recent market data
            model_type: Type of trading model
            
        Returns:
            Tuple of (regime_confidence, regime_name)
        """
        # Calculate market regime indicators
        regime_indicators = self._calculate_regime_indicators(market_data)
        regime = self._classify_regime(regime_indicators)
        
        # Get historical performance in this regime
        regime_confidence = self._get_regime_performance(regime, model_type)
        
        self.current_regime = regime
        return regime_confidence, regime
    
    def _calculate_regime_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate market regime indicators"""
        if 'close' not in data.columns:
            # Handle case where columns might be capitalized
            data.columns = [c.lower() for c in data.columns]
            
        close_prices = data['close']
        
        # Volatility regime
        returns = close_prices.pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        # Trend regime
        sma_20 = close_prices.rolling(20).mean().iloc[-1]
        sma_50 = close_prices.rolling(50).mean().iloc[-1]
        current_price = close_prices.iloc[-1]
        
        trend_strength = (current_price - sma_50) / sma_50
        trend_direction = 1 if current_price > sma_20 > sma_50 else -1 if current_price < sma_20 < sma_50 else 0
        
        # Volume regime
        if 'volume' in data.columns:
            volume_ratio = data['volume'].rolling(20).mean().iloc[-1] / data['volume'].rolling(50).mean().iloc[-1]
        else:
            volume_ratio = 1.0
            
        # Momentum regime
        momentum_5 = (close_prices.iloc[-1] / close_prices.iloc[-6] - 1) if len(close_prices) >= 6 else 0
        momentum_20 = (close_prices.iloc[-1] / close_prices.iloc[-21] - 1) if len(close_prices) >= 21 else 0
        
        return {
            'volatility': volatility,
            'trend_strength': abs(trend_strength),
            'trend_direction': trend_direction,
            'volume_ratio': volume_ratio,
            'momentum_5d': momentum_5,
            'momentum_20d': momentum_20
        }
    
    def _classify_regime(self, indicators: Dict[str, float]) -> str:
        """Classify market regime based on indicators"""
        vol = indicators['volatility']
        trend = indicators['trend_strength']
        momentum = indicators['momentum_20d']
        
        # Volatility thresholds
        if vol > 0.25:
            vol_regime = "high_vol"
        elif vol > 0.15:
            vol_regime = "medium_vol"
        else:
            vol_regime = "low_vol"
            
        # Trend thresholds
        if trend > 0.05:
            trend_regime = "trending"
        elif trend > 0.02:
            trend_regime = "weak_trend"
        else:
            trend_regime = "ranging"
            
        # Momentum thresholds
        if abs(momentum) > 0.05:
            momentum_regime = "momentum"
        else:
            momentum_regime = "mean_reversion"
            
        # Combine into regime classification
        regime = f"{vol_regime}_{trend_regime}_{momentum_regime}"
        
        # Simplify to major regimes
        if "trending" in regime and "momentum" in regime:
            return "strong_trend"
        elif "trending" in regime:
            return "weak_trend"
        elif "ranging" in regime and "low_vol" in regime:
            return "quiet_range"
        elif "ranging" in regime and "high_vol" in regime:
            return "choppy_range"
        else:
            return "mixed"
    
    def _get_regime_performance(self, regime: str, model_type: str) -> float:
        """Get historical performance for regime-model combination"""
        # Default regime confidences based on model type
        regime_matrix = {
            "trend_following": {
                "strong_trend": 0.8,
                "weak_trend": 0.6,
                "quiet_range": 0.3,
                "choppy_range": 0.2,
                "mixed": 0.5
            },
            "mean_reversion": {
                "strong_trend": 0.2,
                "weak_trend": 0.4,
                "quiet_range": 0.8,
                "choppy_range": 0.6,
                "mixed": 0.5
            },
            "momentum": {
                "strong_trend": 0.9,
                "weak_trend": 0.7,
                "quiet_range": 0.2,
                "choppy_range": 0.3,
                "mixed": 0.5
            }
        }
        
        # Use historical performance if available
        key = f"{regime}_{model_type}"
        if key in self.regime_performance:
            return self.regime_performance[key]
            
        # Fallback to default matrix
        return regime_matrix.get(model_type, {}).get(regime, 0.5)
    
    def update_regime_performance(
        self, 
        regime: str, 
        model_type: str, 
        performance: float,
        alpha: float = 0.1
    ):
        """Update regime performance with exponential smoothing"""
        key = f"{regime}_{model_type}"
        if key in self.regime_performance:
            self.regime_performance[key] = (
                (1 - alpha) * self.regime_performance[key] + 
                alpha * performance
            )
        else:
            self.regime_performance[key] = performance


class PerformanceConfidenceCalculator:
    """Calculate confidence based on recent model performance"""
    
    def __init__(self, lookback_window: int = 20):
        self.lookback_window = lookback_window
        self.performance_history = []
        self.prediction_history = []
        self.outcome_history = []
        
    def calculate_performance_confidence(
        self, 
        recent_accuracy: float,
        recent_trades: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate confidence based on recent performance
        
        Args:
            recent_accuracy: Recent prediction accuracy
            recent_trades: Recent trading performance data
            
        Returns:
            Performance-based confidence score
        """
        # Base confidence from accuracy
        base_confidence = self._accuracy_to_confidence(recent_accuracy)
        
        # Trend in performance
        trend_confidence = self._calculate_performance_trend()
        
        # Consistency of performance
        consistency_confidence = self._calculate_performance_consistency()
        
        # Combine components
        performance_confidence = (
            0.5 * base_confidence +
            0.3 * trend_confidence +
            0.2 * consistency_confidence
        )
        
        return np.clip(performance_confidence, 0, 1)
    
    def _accuracy_to_confidence(self, accuracy: float) -> float:
        """Convert accuracy to confidence score"""
        # Sigmoid transformation with inflection at 0.55
        # High confidence requires accuracy > 0.55
        x = (accuracy - 0.55) * 10  # Scale and center
        confidence = 1 / (1 + np.exp(-x))
        return confidence
    
    def _calculate_performance_trend(self) -> float:
        """Calculate trend in recent performance"""
        if len(self.performance_history) < 10:
            return 0.5
            
        # Use linear regression to find trend
        x = np.arange(len(self.performance_history))
        y = self.performance_history
        
        slope, _, r_value, _, _ = stats.linregress(x, y)
        
        # Convert slope to confidence
        # Positive slope = increasing confidence
        trend_confidence = 0.5 + slope * 10  # Scale slope
        trend_confidence *= abs(r_value)  # Weight by correlation strength
        
        return np.clip(trend_confidence, 0, 1)
    
    def _calculate_performance_consistency(self) -> float:
        """Calculate consistency of performance"""
        if len(self.performance_history) < 5:
            return 0.5
            
        # Lower variance = higher consistency = higher confidence
        variance = np.var(self.performance_history)
        consistency = 1 / (1 + variance * 10)
        
        return consistency
    
    def update_performance(self, accuracy: float, trade_result: float):
        """Update performance tracking"""
        self.performance_history.append(accuracy)
        
        # Keep only recent history
        if len(self.performance_history) > self.lookback_window:
            self.performance_history = self.performance_history[-self.lookback_window:]


class AdaptiveThresholdOptimizer:
    """Adaptively optimize confidence thresholds"""
    
    def __init__(self, config: ConfidenceConfig):
        self.config = config
        self.trade_count = 0
        self.performance_history = []
        self.threshold_history = []
        
    def adaptive_confidence_threshold(
        self, 
        recent_performance: List[float],
        current_trade_frequency: float,
        base_threshold: float = 0.7
    ) -> float:
        """
        Adjust confidence threshold based on recent performance and trade frequency
        
        Args:
            recent_performance: Recent trading returns
            current_trade_frequency: Current trades per year
            base_threshold: Base confidence threshold
            
        Returns:
            Adapted confidence threshold
        """
        # Performance-based adjustment
        if len(recent_performance) >= 10:
            win_rate = np.mean(np.array(recent_performance) > 0)
            avg_return = np.mean(recent_performance)
            
            if win_rate < 0.4:  # Poor performance - require higher confidence
                performance_adjustment = min(0.2, (0.4 - win_rate) * 0.5)
            elif win_rate > 0.6:  # Good performance - can accept lower confidence
                performance_adjustment = max(-0.1, (win_rate - 0.6) * -0.25)
            else:
                performance_adjustment = 0
        else:
            performance_adjustment = 0
            
        # Frequency-based adjustment
        target_frequency = self.config.target_trades_per_year
        if current_trade_frequency > self.config.max_trades_per_year:
            # Too many trades - increase threshold
            frequency_adjustment = min(0.15, (current_trade_frequency - target_frequency) / target_frequency * 0.3)
        elif current_trade_frequency < self.config.min_trades_per_year:
            # Too few trades - decrease threshold
            frequency_adjustment = max(-0.1, (current_trade_frequency - target_frequency) / target_frequency * 0.2)
        else:
            frequency_adjustment = 0
            
        # Combine adjustments
        adapted_threshold = base_threshold + performance_adjustment + frequency_adjustment
        
        # Ensure within bounds
        adapted_threshold = np.clip(
            adapted_threshold,
            self.config.min_confidence_threshold,
            self.config.max_confidence_threshold
        )
        
        return adapted_threshold
    
    def optimize_threshold_from_history(
        self, 
        confidence_scores: np.ndarray,
        returns: np.ndarray,
        transaction_cost: float = 0.002
    ) -> float:
        """
        Optimize threshold to maximize risk-adjusted returns
        
        Args:
            confidence_scores: Historical confidence scores
            returns: Historical returns when trades were made
            transaction_cost: Transaction cost per trade
            
        Returns:
            Optimal threshold
        """
        thresholds = np.linspace(0.5, 0.9, 41)
        best_sharpe = -np.inf
        best_threshold = 0.7
        
        for threshold in thresholds:
            # Simulate trades with this threshold
            trade_mask = confidence_scores >= threshold
            
            if np.sum(trade_mask) < 10:  # Too few trades
                continue
                
            trade_returns = returns[trade_mask] - transaction_cost
            
            if len(trade_returns) == 0:
                continue
                
            # Calculate Sharpe ratio
            mean_return = np.mean(trade_returns)
            std_return = np.std(trade_returns)
            
            if std_return > 0:
                sharpe = mean_return / std_return * np.sqrt(252)
                
                # Penalize for extreme trade frequencies
                trade_frequency = np.sum(trade_mask) / len(trade_mask) * 252  # Annualized
                
                if trade_frequency < 20 or trade_frequency > 60:
                    sharpe *= 0.8  # Penalty for suboptimal frequency
                    
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_threshold = threshold
                    
        return best_threshold


class MLConfidenceFilter:
    """Main confidence filtering system"""
    
    def __init__(self, config: Optional[ConfidenceConfig] = None):
        """
        Initialize ML confidence filtering system
        
        Args:
            config: Configuration for confidence filtering
        """
        self.config = config or ConfidenceConfig()
        
        # Initialize calculators
        self.model_confidence_calc = ModelConfidenceCalculator()
        self.ensemble_confidence_calc = EnsembleConfidenceCalculator()
        self.regime_confidence_calc = RegimeConfidenceCalculator()
        self.performance_confidence_calc = PerformanceConfidenceCalculator()
        self.threshold_optimizer = AdaptiveThresholdOptimizer(self.config)
        
        # Initialize calibrator if available
        if ModelCalibrator:
            calibration_config = CalibrationConfig(
                method=self.config.calibration_method,
                optimize_threshold=True,
                threshold_metric="profit"
            )
            self.calibrator = ModelCalibrator(calibration_config)
        else:
            self.calibrator = None
            
        # Tracking
        self.trade_history = []
        self.confidence_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_return': 0.0,
            'total_fees': 0.0
        }
        
        logger.info("MLConfidenceFilter initialized with multi-level filtering")
    
    def calculate_prediction_confidence(
        self,
        models: Dict[str, BaseEstimator],
        features: np.ndarray,
        market_data: Optional[pd.DataFrame] = None,
        model_type: str = "trend_following"
    ) -> ConfidenceMetrics:
        """
        Calculate comprehensive confidence metrics for predictions
        
        Args:
            models: Dictionary of trained models
            features: Input features for prediction
            market_data: Recent market data for regime analysis
            model_type: Type of trading model
            
        Returns:
            Comprehensive confidence metrics
        """
        metrics = ConfidenceMetrics()
        
        # Level 1: Individual model confidence
        if self.config.enable_model_confidence:
            model_confidences = {}
            model_predictions = {}
            
            for model_id, model in models.items():
                confidence = self.model_confidence_calc.calculate_prediction_confidence(
                    model, features, method='entropy'
                )
                pred = model.predict(features)
                
                model_confidences[model_id] = confidence
                model_predictions[model_id] = pred
                
            # Average model confidence
            all_confidences = np.array(list(model_confidences.values()))
            metrics.model_confidence = np.mean(all_confidences) if len(all_confidences) > 0 else 0.5
            
            # Calculate prediction variance
            all_predictions = np.array(list(model_predictions.values()))
            metrics.prediction_variance = np.var(all_predictions) if len(all_predictions) > 1 else 0.0
            
        # Level 2: Ensemble agreement
        if self.config.enable_ensemble_agreement and len(models) > 1:
            ensemble_agreement = self.ensemble_confidence_calc.calculate_ensemble_agreement(
                model_predictions, model_confidences
            )
            metrics.ensemble_agreement = np.mean(ensemble_agreement)
        else:
            metrics.ensemble_agreement = 0.5
            
        # Level 3: Market regime confidence
        if self.config.enable_regime_confidence and market_data is not None:
            regime_conf, regime = self.regime_confidence_calc.calculate_regime_confidence(
                market_data, model_type
            )
            metrics.regime_confidence = regime_conf
            metrics.market_regime = regime
        else:
            metrics.regime_confidence = 0.5
            
        # Level 4: Recent performance confidence
        if self.config.enable_performance_confidence:
            recent_accuracy = self._calculate_recent_accuracy()
            metrics.performance_confidence = self.performance_confidence_calc.calculate_performance_confidence(
                recent_accuracy, self.trade_history[-20:] if len(self.trade_history) >= 20 else []
            )
            metrics.recent_accuracy = recent_accuracy
        else:
            metrics.performance_confidence = 0.5
            
        # Calculate overall confidence
        metrics.overall_confidence = self._calculate_overall_confidence(metrics)
        
        # Determine if should trade
        current_threshold = self._get_current_threshold()
        metrics.should_trade = metrics.overall_confidence >= current_threshold
        
        # Calculate position size multiplier
        if metrics.should_trade:
            metrics.position_size_multiplier = self._calculate_position_size_multiplier(metrics)
        
        # Calculate trade frequency adjustment
        metrics.trade_frequency_adjustment = self._calculate_frequency_adjustment()
        
        return metrics
    
    def apply_confidence_filter(
        self, 
        signals: np.ndarray, 
        confidence_scores: np.ndarray, 
        min_confidence: float = 0.7
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply confidence filtering to trading signals
        
        Args:
            signals: Original trading signals
            confidence_scores: Confidence scores for each signal
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (filtered_signals, high_confidence_mask)
        """
        # Create filtered signals
        filtered_signals = np.zeros_like(signals)
        
        # Only keep high confidence signals
        high_conf_mask = confidence_scores >= min_confidence
        filtered_signals[high_conf_mask] = signals[high_conf_mask]
        
        # Log filtering results
        original_trades = np.sum(signals != 0)
        filtered_trades = np.sum(filtered_signals != 0)
        
        if original_trades > 0:
            reduction_pct = (1 - filtered_trades / original_trades) * 100
            logger.info(
                f"Confidence filter: {original_trades} → {filtered_trades} signals "
                f"({reduction_pct:.1f}% reduction)"
            )
        
        return filtered_signals, high_conf_mask
    
    def calibrate_confidence_scores(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> BaseEstimator:
        """
        Calibrate model to improve confidence score reliability
        
        Args:
            model: Model to calibrate
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Calibrated model
        """
        if self.calibrator is None:
            logger.warning("Model calibrator not available")
            return model
            
        try:
            calibrated_model = self.calibrator.calibrate(
                model, X_train, y_train, X_val, y_val
            )
            
            logger.info(
                f"Model calibrated. ECE: {self.calibrator.calibration_metrics.ece:.4f}, "
                f"Optimal threshold: {self.calibrator.optimal_threshold:.3f}"
            )
            
            return calibrated_model
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return model
    
    def analyze_trade_frequency(
        self, 
        confidence_scores: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Analyze trade frequency at different confidence thresholds
        
        Args:
            confidence_scores: Historical confidence scores
            thresholds: Confidence thresholds to analyze
            
        Returns:
            DataFrame with frequency analysis
        """
        if thresholds is None:
            thresholds = np.linspace(0.5, 0.9, 21)
            
        results = []
        
        for threshold in thresholds:
            trades_mask = confidence_scores >= threshold
            trades_count = np.sum(trades_mask)
            
            # Annualized frequency (assuming daily data)
            trades_per_year = trades_count / len(confidence_scores) * 252
            
            # Estimated transaction costs
            annual_cost = trades_per_year * self.config.transaction_cost
            
            results.append({
                'threshold': threshold,
                'trades_count': trades_count,
                'trades_per_year': trades_per_year,
                'trade_frequency': trades_count / len(confidence_scores),
                'annual_transaction_cost': annual_cost
            })
            
        return pd.DataFrame(results)
    
    def find_optimal_confidence_threshold(
        self,
        confidence_scores: np.ndarray,
        returns: np.ndarray,
        target_trades_per_year: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Find optimal confidence threshold balancing returns and frequency
        
        Args:
            confidence_scores: Historical confidence scores
            returns: Historical returns
            target_trades_per_year: Target number of trades per year
            
        Returns:
            Dictionary with optimization results
        """
        if target_trades_per_year is None:
            target_trades_per_year = self.config.target_trades_per_year
            
        # Optimize for maximum Sharpe ratio
        optimal_sharpe_threshold = self.threshold_optimizer.optimize_threshold_from_history(
            confidence_scores, returns, self.config.transaction_cost
        )
        
        # Find threshold that achieves target frequency
        frequency_analysis = self.analyze_trade_frequency(confidence_scores)
        target_row = frequency_analysis.iloc[
            (frequency_analysis['trades_per_year'] - target_trades_per_year).abs().idxmin()
        ]
        frequency_threshold = target_row['threshold']
        
        # Adaptive threshold based on recent performance
        recent_returns = returns[-50:] if len(returns) >= 50 else returns
        current_frequency = len(returns) / 252  # Approximate current frequency
        adaptive_threshold = self.threshold_optimizer.adaptive_confidence_threshold(
            recent_returns, current_frequency * 252
        )
        
        return {
            'optimal_sharpe_threshold': optimal_sharpe_threshold,
            'frequency_target_threshold': frequency_threshold,
            'adaptive_threshold': adaptive_threshold,
            'target_frequency': target_trades_per_year,
            'estimated_frequency_at_optimal': frequency_analysis[
                frequency_analysis['threshold'] == optimal_sharpe_threshold
            ]['trades_per_year'].iloc[0] if len(frequency_analysis) > 0 else 0
        }
    
    def _calculate_overall_confidence(self, metrics: ConfidenceMetrics) -> float:
        """Calculate weighted overall confidence score"""
        weights = {
            'model': 0.3,
            'ensemble': 0.25,
            'regime': 0.25,
            'performance': 0.2
        }
        
        overall = (
            weights['model'] * metrics.model_confidence +
            weights['ensemble'] * metrics.ensemble_agreement +
            weights['regime'] * metrics.regime_confidence +
            weights['performance'] * metrics.performance_confidence
        )
        
        return np.clip(overall, 0, 1)
    
    def _get_current_threshold(self) -> float:
        """Get current adaptive threshold"""
        if len(self.trade_history) < 20:
            return self.config.base_confidence_threshold
            
        recent_performance = [trade.get('return', 0) for trade in self.trade_history[-20:]]
        current_frequency = self._calculate_current_frequency()
        
        return self.threshold_optimizer.adaptive_confidence_threshold(
            recent_performance, current_frequency
        )
    
    def _calculate_position_size_multiplier(self, metrics: ConfidenceMetrics) -> float:
        """Calculate position size multiplier based on confidence"""
        # Higher confidence = larger position (up to max)
        base_multiplier = metrics.overall_confidence
        
        # Adjust for regime confidence
        regime_adjustment = 0.5 + 0.5 * metrics.regime_confidence
        
        # Adjust for ensemble agreement
        agreement_adjustment = 0.8 + 0.4 * metrics.ensemble_agreement
        
        multiplier = base_multiplier * regime_adjustment * agreement_adjustment
        
        # Cap at maximum position size
        max_multiplier = self.config.max_position_size / 0.1  # Assuming 10% base position
        return min(multiplier, max_multiplier)
    
    def _calculate_frequency_adjustment(self) -> float:
        """Calculate adjustment factor for trade frequency"""
        if len(self.trade_history) < 50:
            return 1.0
            
        recent_frequency = self._calculate_current_frequency()
        target_frequency = self.config.target_trades_per_year
        
        if recent_frequency > self.config.max_trades_per_year:
            return 0.7  # Reduce trading
        elif recent_frequency < self.config.min_trades_per_year:
            return 1.3  # Increase trading
        else:
            return 1.0  # Maintain current level
    
    def _calculate_current_frequency(self) -> float:
        """Calculate current annualized trade frequency"""
        if len(self.trade_history) < 10:
            return self.config.target_trades_per_year
            
        # Use last 60 days
        recent_trades = self.trade_history[-60:]
        return len(recent_trades) / 60 * 365
    
    def _calculate_recent_accuracy(self) -> float:
        """Calculate recent prediction accuracy"""
        if len(self.trade_history) < 10:
            return 0.5
            
        recent_trades = self.trade_history[-20:]
        correct_predictions = sum(1 for trade in recent_trades if trade.get('return', 0) > 0)
        return correct_predictions / len(recent_trades)
    
    def update_trade_performance(
        self, 
        prediction: float, 
        actual_return: float, 
        confidence: float,
        regime: str = "unknown"
    ):
        """
        Update performance tracking with new trade result
        
        Args:
            prediction: Model prediction
            actual_return: Actual trade return
            confidence: Confidence score
            regime: Market regime
        """
        trade_record = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'return': actual_return,
            'confidence': confidence,
            'regime': regime,
            'correct': (prediction > 0) == (actual_return > 0)
        }
        
        self.trade_history.append(trade_record)
        
        # Update performance metrics
        self.performance_metrics['total_trades'] += 1
        if actual_return > 0:
            self.performance_metrics['winning_trades'] += 1
        self.performance_metrics['total_return'] += actual_return
        self.performance_metrics['total_fees'] += self.config.transaction_cost
        
        # Update component calculators
        self.performance_confidence_calc.update_performance(
            float(trade_record['correct']), actual_return
        )
        
        # Update regime performance
        model_type = "trend_following"  # Default, could be parameter
        self.regime_confidence_calc.update_regime_performance(
            regime, model_type, float(trade_record['correct'])
        )
        
        # Keep history manageable
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-500:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        if len(self.trade_history) == 0:
            return {"status": "No trades recorded"}
            
        recent_trades = self.trade_history[-50:] if len(self.trade_history) >= 50 else self.trade_history
        
        # Calculate metrics
        total_trades = len(self.trade_history)
        win_rate = np.mean([trade['correct'] for trade in recent_trades])
        avg_return = np.mean([trade['return'] for trade in recent_trades])
        total_return = sum([trade['return'] for trade in self.trade_history])
        
        # Annualized metrics
        if len(self.trade_history) > 1:
            days_elapsed = (self.trade_history[-1]['timestamp'] - self.trade_history[0]['timestamp']).days
            if days_elapsed > 0:
                trades_per_year = total_trades / days_elapsed * 365
                annual_return = total_return / days_elapsed * 365
            else:
                trades_per_year = 0
                annual_return = 0
        else:
            trades_per_year = 0
            annual_return = 0
            
        # Sharpe ratio (simplified)
        returns = [trade['return'] for trade in recent_trades]
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
            
        return {
            'total_trades': total_trades,
            'recent_win_rate': win_rate,
            'recent_avg_return': avg_return,
            'total_return': total_return,
            'trades_per_year': trades_per_year,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'confidence_threshold': self._get_current_threshold(),
            'total_fees': self.performance_metrics['total_fees'],
            'net_return': total_return - self.performance_metrics['total_fees']
        }


def create_confidence_filter(config: Optional[ConfidenceConfig] = None) -> MLConfidenceFilter:
    """
    Create ML confidence filter instance
    
    Args:
        config: Optional configuration
        
    Returns:
        Configured confidence filter
    """
    return MLConfidenceFilter(config)


if __name__ == "__main__":
    # Example usage and testing
    import yfinance as yf
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    
    # Get sample data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="2y")
    data.columns = [c.lower() for c in data.columns]
    
    # Create features
    data['returns'] = data['close'].pct_change()
    data['sma_20'] = data['close'].rolling(20).mean()
    data['rsi'] = 100 - (100 / (1 + data['returns'].rolling(14).apply(
        lambda x: np.mean(x[x > 0]) / np.mean(np.abs(x[x < 0])) if len(x[x < 0]) > 0 else 1
    )))
    
    X = pd.DataFrame(index=data.index)
    X['returns'] = data['returns']
    X['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
    X['price_to_sma'] = data['close'] / data['sma_20']
    X['rsi'] = data['rsi']
    X = X.dropna()
    
    # Create target (next day return direction)
    y = (data['close'].shift(-1) > data['close']).astype(int)
    y = y.loc[X.index]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=False
    )
    
    # Train multiple models
    models = {}
    models['rf'] = RandomForestClassifier(n_estimators=100, random_state=42)
    models['gb'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
    # Create confidence filter
    config = ConfidenceConfig(
        base_confidence_threshold=0.65,
        target_trades_per_year=40,
        enable_regime_confidence=True
    )
    
    confidence_filter = create_confidence_filter(config)
    
    # Test confidence calculation
    test_features = X_test.values[:10]
    test_market_data = data.loc[X_test.index[:10]]
    
    confidence_metrics = confidence_filter.calculate_prediction_confidence(
        models, test_features, test_market_data
    )
    
    print("\nConfidence Metrics:")
    print(f"  Model Confidence: {confidence_metrics.model_confidence:.3f}")
    print(f"  Ensemble Agreement: {confidence_metrics.ensemble_agreement:.3f}")
    print(f"  Regime Confidence: {confidence_metrics.regime_confidence:.3f}")
    print(f"  Overall Confidence: {confidence_metrics.overall_confidence:.3f}")
    print(f"  Should Trade: {confidence_metrics.should_trade}")
    print(f"  Market Regime: {confidence_metrics.market_regime}")
    
    # Test signal filtering
    dummy_signals = np.random.choice([-1, 0, 1], size=100)
    confidence_scores = np.random.beta(2, 3, size=100)  # Realistic confidence distribution
    
    filtered_signals, mask = confidence_filter.apply_confidence_filter(
        dummy_signals, confidence_scores, min_confidence=0.7
    )
    
    print(f"\nSignal Filtering:")
    print(f"  Original signals: {np.sum(dummy_signals != 0)}")
    print(f"  Filtered signals: {np.sum(filtered_signals != 0)}")
    print(f"  Reduction: {(1 - np.sum(filtered_signals != 0) / np.sum(dummy_signals != 0)) * 100:.1f}%")
    
    # Test frequency analysis
    freq_analysis = confidence_filter.analyze_trade_frequency(confidence_scores)
    print(f"\nFrequency Analysis (sample):")
    print(freq_analysis.head(10).round(3))
    
    # Test threshold optimization
    dummy_returns = np.random.normal(0.001, 0.02, size=100)  # 0.1% daily return, 2% volatility
    threshold_results = confidence_filter.find_optimal_confidence_threshold(
        confidence_scores, dummy_returns
    )
    
    print(f"\nThreshold Optimization:")
    for key, value in threshold_results.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")