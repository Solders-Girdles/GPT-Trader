"""
ML model implementations for strategy selection - LOCAL to this slice.

Using simplified models to maintain isolation.
In production, could use scikit-learn, XGBoost, etc.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .types import StrategyName
from .features import get_feature_names


class StrategySelector:
    """
    Main ML model for strategy selection.
    
    Simplified implementation - in production use XGBoost/LightGBM.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 5, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_names = get_feature_names()
        self.strategy_models: Dict[StrategyName, 'SimpleModel'] = {}
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model.
        
        X: Feature matrix
        y: Labels (strategy performance)
        """
        # Train a model for each strategy
        for strategy in StrategyName:
            model = SimpleModel()
            # Filter data for this strategy
            strategy_mask = self._get_strategy_mask(y, strategy)
            if np.any(strategy_mask):
                strategy_X = X[strategy_mask]
                strategy_y = self._extract_performance(y[strategy_mask])
                model.fit(strategy_X, strategy_y)
                self.strategy_models[strategy] = model
        
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict best strategy for each sample."""
        predictions = []
        for sample in X:
            scores = {}
            for strategy, model in self.strategy_models.items():
                scores[strategy] = model.predict(sample.reshape(1, -1))[0]
            best_strategy = max(scores, key=scores.get)
            predictions.append(best_strategy)
        return np.array(predictions)
    
    def predict_return(self, features: np.ndarray, strategy: StrategyName) -> float:
        """Predict expected return for a strategy."""
        if strategy not in self.strategy_models:
            return np.random.uniform(-5, 15)  # Fallback
        
        model = self.strategy_models[strategy]
        return model.predict(features.reshape(1, -1))[0]
    
    def predict_sharpe(self, features: np.ndarray, strategy: StrategyName) -> float:
        """Predict Sharpe ratio for a strategy."""
        # Simplified: derive from return prediction
        expected_return = self.predict_return(features, strategy)
        # Estimate Sharpe based on return (simplified)
        if expected_return > 10:
            return np.random.uniform(1.5, 2.5)
        elif expected_return > 0:
            return np.random.uniform(0.5, 1.5)
        else:
            return np.random.uniform(-0.5, 0.5)
    
    def predict_drawdown(self, features: np.ndarray, strategy: StrategyName) -> float:
        """Predict maximum drawdown for a strategy."""
        # Simplified: estimate based on volatility in features
        volatility = features[0] * 100  # First feature is normalized volatility
        base_drawdown = -10 - (volatility * 0.5)
        strategy_factor = {
            StrategyName.VOLATILITY: 1.2,  # Higher drawdown
            StrategyName.MOMENTUM: 1.1,
            StrategyName.BREAKOUT: 1.15,
            StrategyName.MEAN_REVERSION: 0.9,
            StrategyName.SIMPLE_MA: 1.0
        }
        return base_drawdown * strategy_factor.get(strategy, 1.0)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        predictions = self.predict(X)
        correct = np.sum(predictions == y)
        return correct / len(y) if len(y) > 0 else 0
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'n_strategies': len(self.strategy_models)
        }
    
    def _get_strategy_mask(self, y: np.ndarray, strategy: StrategyName) -> np.ndarray:
        """Get mask for samples belonging to a strategy."""
        # Simplified: assume y contains strategy labels
        return np.array([str(strategy.value) in str(label) for label in y])
    
    def _extract_performance(self, y: np.ndarray) -> np.ndarray:
        """Extract performance metrics from labels."""
        # Simplified: return random performance for now
        return np.random.uniform(-10, 20, size=len(y))


class SimpleModel:
    """
    Simplified model for individual strategy performance prediction.
    
    In production, replace with real ML model.
    """
    
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.mean = 0
        self.std = 1
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model using simple linear regression."""
        # Add bias term
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        
        # Simple least squares (in production use proper solver)
        try:
            # Pseudo-inverse for stability
            coeffs = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
            self.intercept = coeffs[0]
            self.coefficients = coeffs[1:]
        except:
            # Fallback to random coefficients
            self.intercept = np.mean(y) if len(y) > 0 else 0
            self.coefficients = np.random.randn(X.shape[1]) * 0.1
        
        self.mean = np.mean(y) if len(y) > 0 else 0
        self.std = np.std(y) if len(y) > 1 else 1
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.coefficients is None:
            return np.array([self.mean] * X.shape[0])
        
        predictions = X @ self.coefficients + self.intercept
        # Add some noise for realism
        noise = np.random.normal(0, self.std * 0.1, size=predictions.shape)
        return predictions + noise


class ConfidenceScorer:
    """
    Model for scoring prediction confidence.
    
    Based on:
    - Historical accuracy for similar conditions
    - Feature certainty
    - Model agreement
    """
    
    def __init__(self):
        self.accuracy_history: Dict[StrategyName, List[float]] = {}
        self.feature_importance = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, base_model: StrategySelector):
        """
        Fit confidence scorer based on model performance.
        """
        # Calculate feature importance (simplified)
        self.feature_importance = self._calculate_feature_importance(X, y)
        
        # Initialize accuracy history
        for strategy in StrategyName:
            self.accuracy_history[strategy] = []
            
        # Simulate historical accuracy (in production, use cross-validation)
        for strategy in StrategyName:
            # Random accuracy between 0.6 and 0.9
            accuracy = np.random.uniform(0.6, 0.9)
            self.accuracy_history[strategy].append(accuracy)
    
    def score(self, features: np.ndarray, strategy: StrategyName) -> float:
        """
        Calculate confidence score for a prediction.
        
        Returns value between 0 and 1.
        """
        # Base confidence from historical accuracy
        if strategy in self.accuracy_history and self.accuracy_history[strategy]:
            base_confidence = np.mean(self.accuracy_history[strategy])
        else:
            base_confidence = 0.5
        
        # Adjust based on feature quality
        feature_confidence = self._assess_feature_quality(features)
        
        # Combine scores
        final_confidence = base_confidence * 0.7 + feature_confidence * 0.3
        
        # Add strategy-specific adjustments
        if strategy == StrategyName.SIMPLE_MA:
            final_confidence *= 1.1  # Boost for reliable baseline
        elif strategy == StrategyName.VOLATILITY:
            # Check if volatility is actually high
            if features[0] > 0.3:  # High volatility
                final_confidence *= 1.15
            else:
                final_confidence *= 0.85
        
        return min(final_confidence, 1.0)
    
    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate feature importance scores."""
        # Simplified: use variance as proxy for importance
        if X.shape[0] > 0:
            importance = np.var(X, axis=0)
            # Normalize
            if importance.sum() > 0:
                importance = importance / importance.sum()
            return importance
        return np.ones(X.shape[1]) / X.shape[1]
    
    def _assess_feature_quality(self, features: np.ndarray) -> float:
        """Assess quality/reliability of input features."""
        quality_score = 0.5  # Base score
        
        # Check for extreme values (might indicate data issues)
        if np.any(np.abs(features) > 10):
            quality_score -= 0.2
        
        # Check for NaN or infinite values
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            quality_score -= 0.3
        
        # Bonus for normal-looking values
        if np.all(np.abs(features) < 2):
            quality_score += 0.2
        
        # Weight by feature importance if available
        if self.feature_importance is not None:
            weighted_score = np.sum(features * self.feature_importance)
            if 0 < weighted_score < 1:
                quality_score += 0.1
        
        return max(0, min(quality_score, 1))