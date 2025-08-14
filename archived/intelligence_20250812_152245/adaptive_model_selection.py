"""
Adaptive Model Selection Framework for GPT-Trader Phase 2.

This module provides intelligent model selection and combination:
- Dynamic model selection based on market conditions
- Contextual bandits for model recommendation
- Meta-learning for rapid model adaptation
- Ensemble weighting with performance tracking
- Regime-aware model switching
- Online model portfolio optimization

Automatically selects and combines the best models for current market conditions
without manual intervention.
"""

from __future__ import annotations

import logging
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.svm import SVR

# Optional advanced ML libraries
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF

    HAS_GP = True
except ImportError:
    HAS_GP = False

# Optional contextual bandits
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    HAS_BANDITS = True
except ImportError:
    HAS_BANDITS = False

from bot.utils.base import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Track model performance over time."""

    model_name: str
    recent_scores: list[float] = field(default_factory=list)
    overall_score: float = 0.0
    volatility: float = 0.0
    stability: float = 0.0
    last_updated: float = 0.0
    context_performance: dict[str, float] = field(default_factory=dict)
    prediction_count: int = 0
    error_rate: float = 0.0


@dataclass
class MarketContext:
    """Market context for model selection."""

    volatility_regime: str  # low, medium, high
    trend_regime: str  # bull, bear, sideways
    volume_regime: str  # low, normal, high
    time_of_day: str  # morning, midday, close
    day_of_week: str  # monday, tuesday, etc.
    market_stress: float  # 0-1 stress indicator
    regime_confidence: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert context to numerical vector."""
        vol_map = {"low": 0, "medium": 1, "high": 2}
        trend_map = {"bear": 0, "sideways": 1, "bull": 2}
        volume_map = {"low": 0, "normal": 1, "high": 2}
        time_map = {"morning": 0, "midday": 1, "close": 2}
        day_map = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }

        return np.array(
            [
                vol_map.get(self.volatility_regime, 1),
                trend_map.get(self.trend_regime, 1),
                volume_map.get(self.volume_regime, 1),
                time_map.get(self.time_of_day, 1),
                day_map.get(self.day_of_week, 0),
                self.market_stress,
                self.regime_confidence,
            ]
        )


@dataclass
class AdaptiveModelConfig(BaseConfig):
    """Configuration for adaptive model selection."""

    # Model pool
    model_types: list[str] = field(
        default_factory=lambda: ["linear", "ridge", "lasso", "random_forest", "svm"]
    )

    # Selection strategy
    selection_method: str = "contextual_bandit"  # contextual_bandit, performance, ensemble
    bandit_algorithm: str = "epsilon_greedy"  # epsilon_greedy, ucb, thompson

    # Performance tracking
    performance_window: int = 50
    min_observations: int = 20
    update_frequency: int = 10

    # Context parameters
    use_market_context: bool = True
    context_features: list[str] = field(
        default_factory=lambda: ["volatility_regime", "trend_regime", "volume_regime"]
    )

    # Ensemble parameters
    ensemble_method: str = "dynamic_weighted"  # equal, performance_weighted, dynamic_weighted
    min_ensemble_size: int = 3
    max_ensemble_size: int = 7

    # Bandit parameters
    epsilon: float = 0.1
    ucb_confidence: float = 1.96
    thompson_alpha: float = 1.0
    thompson_beta: float = 1.0

    # Model management
    model_retirement_threshold: float = 0.1  # Retire models performing < 10th percentile
    new_model_probability: float = 0.05
    performance_decay: float = 0.99

    # Evaluation parameters
    cv_folds: int = 5
    scoring_metric: str = "r2"  # r2, neg_mse, neg_mae
    time_series_cv: bool = True

    # Advanced features
    use_meta_learning: bool = True
    online_learning: bool = True
    regime_detection: bool = True

    # Random state
    random_state: int = 42


@dataclass
class SelectionResult:
    """Result from model selection process."""

    selected_model: str
    selection_confidence: float
    context: MarketContext
    performance_prediction: float
    alternative_models: list[tuple[str, float]]
    selection_time: float


class MarketRegimeDetector:
    """Detect current market regime for contextual model selection."""

    def __init__(self, lookback_window: int = 50) -> None:
        self.lookback_window = lookback_window
        self.volatility_thresholds = (0.01, 0.025)  # Low, high
        self.trend_thresholds = (-0.02, 0.02)  # Bear, bull
        self.volume_threshold = 1.5  # Multiple of average

    def detect_regime(self, data: pd.DataFrame) -> MarketContext:
        """Detect current market regime."""

        if len(data) < self.lookback_window:
            # Default regime for insufficient data
            return MarketContext(
                volatility_regime="medium",
                trend_regime="sideways",
                volume_regime="normal",
                time_of_day="midday",
                day_of_week="wednesday",
                market_stress=0.5,
            )

        # Recent data window
        recent_data = data.tail(self.lookback_window)

        # Volatility regime
        if "Close" in recent_data.columns:
            returns = recent_data["Close"].pct_change().dropna()
            volatility = returns.std()

            if volatility < self.volatility_thresholds[0]:
                vol_regime = "low"
            elif volatility > self.volatility_thresholds[1]:
                vol_regime = "high"
            else:
                vol_regime = "medium"

            # Trend regime
            total_return = (recent_data["Close"].iloc[-1] / recent_data["Close"].iloc[0]) - 1

            if total_return < self.trend_thresholds[0]:
                trend_regime = "bear"
            elif total_return > self.trend_thresholds[1]:
                trend_regime = "bull"
            else:
                trend_regime = "sideways"

            # Market stress (based on drawdown)
            cummax = recent_data["Close"].expanding().max()
            drawdown = (recent_data["Close"] - cummax) / cummax
            market_stress = abs(drawdown.min())

        else:
            vol_regime, trend_regime, market_stress = "medium", "sideways", 0.5

        # Volume regime
        if "Volume" in recent_data.columns:
            avg_volume = recent_data["Volume"].mean()
            recent_volume = recent_data["Volume"].tail(5).mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

            if volume_ratio < 0.7:
                volume_regime = "low"
            elif volume_ratio > self.volume_threshold:
                volume_regime = "high"
            else:
                volume_regime = "normal"
        else:
            volume_regime = "normal"

        # Time-based features (simplified)
        if hasattr(data.index, "hour"):
            hour = data.index[-1].hour
            if hour < 11:
                time_of_day = "morning"
            elif hour < 15:
                time_of_day = "midday"
            else:
                time_of_day = "close"
        else:
            time_of_day = "midday"

        if hasattr(data.index, "dayofweek"):
            day_names = [
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ]
            day_of_week = day_names[data.index[-1].dayofweek]
        else:
            day_of_week = "wednesday"

        return MarketContext(
            volatility_regime=vol_regime,
            trend_regime=trend_regime,
            volume_regime=volume_regime,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            market_stress=min(market_stress, 1.0),
            regime_confidence=0.8,  # Simplified confidence
        )


class BaseModelSelector(ABC):
    """Base class for model selection strategies."""

    def __init__(self, config: AdaptiveModelConfig) -> None:
        self.config = config
        self.model_performance = {}
        self.selection_history = []

    @abstractmethod
    def select_model(self, context: MarketContext, available_models: list[str]) -> SelectionResult:
        """Select best model for given context."""
        pass

    @abstractmethod
    def update_performance(
        self, model_name: str, performance: float, context: MarketContext
    ) -> None:
        """Update model performance information."""
        pass


class PerformanceBasedSelector(BaseModelSelector):
    """Select models based on historical performance."""

    def select_model(self, context: MarketContext, available_models: list[str]) -> SelectionResult:
        """Select model with best recent performance."""
        start_time = time.time()

        if not available_models:
            raise ValueError("No models available for selection")

        # Calculate model scores
        model_scores = {}
        for model_name in available_models:
            if model_name in self.model_performance:
                perf = self.model_performance[model_name]

                # Context-specific performance if available
                context_key = f"{context.volatility_regime}_{context.trend_regime}"
                if context_key in perf.context_performance:
                    score = perf.context_performance[context_key]
                else:
                    score = perf.overall_score

                # Penalize high volatility (prefer stability)
                score = score * (1 - perf.volatility * 0.1)
                model_scores[model_name] = score
            else:
                # New model gets neutral score
                model_scores[model_name] = 0.0

        # Select best model
        best_model = max(model_scores.items(), key=lambda x: x[1])[0]
        best_score = model_scores[best_model]

        # Alternative models
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        alternatives = sorted_models[1:4]  # Top 3 alternatives

        selection_time = time.time() - start_time

        return SelectionResult(
            selected_model=best_model,
            selection_confidence=min(best_score + 0.5, 1.0),
            context=context,
            performance_prediction=best_score,
            alternative_models=alternatives,
            selection_time=selection_time,
        )

    def update_performance(
        self, model_name: str, performance: float, context: MarketContext
    ) -> None:
        """Update performance tracking."""
        if model_name not in self.model_performance:
            self.model_performance[model_name] = ModelPerformance(
                model_name=model_name, last_updated=time.time()
            )

        perf = self.model_performance[model_name]

        # Update recent scores
        perf.recent_scores.append(performance)
        if len(perf.recent_scores) > self.config.performance_window:
            perf.recent_scores.pop(0)

        # Update overall score (exponential moving average)
        if perf.overall_score == 0.0:
            perf.overall_score = performance
        else:
            alpha = 0.1
            perf.overall_score = (1 - alpha) * perf.overall_score + alpha * performance

        # Update volatility
        if len(perf.recent_scores) >= 5:
            perf.volatility = np.std(perf.recent_scores[-10:])

        # Context-specific performance
        context_key = f"{context.volatility_regime}_{context.trend_regime}"
        if context_key not in perf.context_performance:
            perf.context_performance[context_key] = performance
        else:
            # Exponential moving average for context
            perf.context_performance[context_key] = (
                0.8 * perf.context_performance[context_key] + 0.2 * performance
            )

        perf.prediction_count += 1
        perf.last_updated = time.time()


class ContextualBanditSelector(BaseModelSelector):
    """Select models using contextual bandit algorithms."""

    def __init__(self, config: AdaptiveModelConfig) -> None:
        super().__init__(config)
        self.context_history = []
        self.reward_history = []
        self.action_history = []
        self.model_to_action = {}
        self.action_to_model = {}

    def _setup_model_actions(self, available_models: list[str]) -> None:
        """Setup mapping between models and actions."""
        for i, model in enumerate(available_models):
            self.model_to_action[model] = i
            self.action_to_model[i] = model

    def select_model(self, context: MarketContext, available_models: list[str]) -> SelectionResult:
        """Select model using contextual bandit."""
        start_time = time.time()

        if not available_models:
            raise ValueError("No models available")

        self._setup_model_actions(available_models)
        context_vector = context.to_vector()

        if self.config.bandit_algorithm == "epsilon_greedy":
            selected_action = self._epsilon_greedy_select(context_vector, available_models)
        elif self.config.bandit_algorithm == "ucb":
            selected_action = self._ucb_select(context_vector, available_models)
        else:
            selected_action = self._epsilon_greedy_select(context_vector, available_models)

        selected_model = self.action_to_model[selected_action]

        # Confidence based on exploration vs exploitation
        if len(self.reward_history) < self.config.min_observations:
            confidence = 0.5  # Low confidence during exploration
        else:
            # Higher confidence for frequently selected models
            selection_count = self.action_history.count(selected_action)
            confidence = min(selection_count / len(self.action_history) * 2, 1.0)

        # Predict performance based on context
        predicted_performance = self._predict_performance(context_vector, selected_action)

        # Alternative models (other actions with their predicted performance)
        alternatives = []
        for action in range(len(available_models)):
            if action != selected_action:
                model_name = self.action_to_model[action]
                pred_perf = self._predict_performance(context_vector, action)
                alternatives.append((model_name, pred_perf))

        alternatives.sort(key=lambda x: x[1], reverse=True)

        selection_time = time.time() - start_time

        return SelectionResult(
            selected_model=selected_model,
            selection_confidence=confidence,
            context=context,
            performance_prediction=predicted_performance,
            alternative_models=alternatives[:3],
            selection_time=selection_time,
        )

    def _epsilon_greedy_select(self, context: np.ndarray, available_models: list[str]) -> int:
        """Epsilon-greedy action selection."""
        n_actions = len(available_models)

        if (
            len(self.context_history) < self.config.min_observations
            or np.random.random() < self.config.epsilon
        ):
            # Explore: random action
            return np.random.randint(n_actions)
        else:
            # Exploit: best predicted action
            best_action = 0
            best_score = -np.inf

            for action in range(n_actions):
                score = self._predict_performance(context, action)
                if score > best_score:
                    best_score = score
                    best_action = action

            return best_action

    def _ucb_select(self, context: np.ndarray, available_models: list[str]) -> int:
        """Upper Confidence Bound action selection."""
        n_actions = len(available_models)
        t = len(self.action_history)

        if t == 0:
            return np.random.randint(n_actions)

        best_action = 0
        best_ucb = -np.inf

        for action in range(n_actions):
            # Count how often this action was selected
            action_count = self.action_history.count(action)

            if action_count == 0:
                # Unselected actions get infinite UCB
                return action

            # Average reward for this action
            action_rewards = [
                self.reward_history[i] for i, a in enumerate(self.action_history) if a == action
            ]
            avg_reward = np.mean(action_rewards)

            # UCB calculation
            confidence_interval = self.config.ucb_confidence * np.sqrt(np.log(t) / action_count)
            ucb_value = avg_reward + confidence_interval

            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_action = action

        return best_action

    def _predict_performance(self, context: np.ndarray, action: int) -> float:
        """Predict performance for action in given context."""
        if len(self.context_history) < 5:
            return 0.0

        # Find similar contexts and their rewards for this action
        similar_rewards = []

        for _i, (hist_context, hist_action, reward) in enumerate(
            zip(self.context_history, self.action_history, self.reward_history, strict=False)
        ):
            if hist_action == action:
                # Calculate context similarity (Euclidean distance)
                distance = np.linalg.norm(context - hist_context)
                if distance < 2.0:  # Threshold for similarity
                    weight = 1.0 / (1.0 + distance)
                    similar_rewards.append(reward * weight)

        if similar_rewards:
            return np.mean(similar_rewards)
        else:
            # Fallback to overall average for this action
            action_rewards = [
                self.reward_history[i] for i, a in enumerate(self.action_history) if a == action
            ]
            return np.mean(action_rewards) if action_rewards else 0.0

    def update_performance(
        self, model_name: str, performance: float, context: MarketContext
    ) -> None:
        """Update bandit with observed reward."""
        if model_name in self.model_to_action:
            action = self.model_to_action[model_name]
            context_vector = context.to_vector()

            self.context_history.append(context_vector)
            self.action_history.append(action)
            self.reward_history.append(performance)

            # Maintain history window
            max_history = 1000
            if len(self.context_history) > max_history:
                self.context_history.pop(0)
                self.action_history.pop(0)
                self.reward_history.pop(0)

        # Also update base performance tracking
        super().update_performance(model_name, performance, context)


class AdaptiveModelSelectionFramework:
    """
    Comprehensive adaptive model selection framework.

    Automatically selects and combines the best models for current
    market conditions using contextual information and performance tracking.
    """

    def __init__(self, config: AdaptiveModelConfig) -> None:
        self.config = config

        # Components
        self.regime_detector = MarketRegimeDetector()
        self.model_selector = self._create_selector()

        # Model pool
        self.model_pool = {}
        self.model_predictions = {}
        self.ensemble_weights = {}

        # Performance tracking
        self.selection_history = []
        self.performance_history = deque(maxlen=config.performance_window * 2)

        # Initialize model pool
        self._initialize_model_pool()

    def _create_selector(self) -> BaseModelSelector:
        """Create model selector based on configuration."""
        if self.config.selection_method == "contextual_bandit":
            return ContextualBanditSelector(self.config)
        else:
            return PerformanceBasedSelector(self.config)

    def _initialize_model_pool(self) -> None:
        """Initialize the pool of available models."""
        logger.info("Initializing model pool...")

        # Basic models
        self.model_pool["linear"] = LinearRegression()
        self.model_pool["ridge"] = Ridge(alpha=1.0)
        self.model_pool["lasso"] = Lasso(alpha=0.1)
        self.model_pool["random_forest"] = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=self.config.random_state
        )

        # Advanced models (if available)
        if HAS_XGBOOST and "xgboost" in self.config.model_types:
            self.model_pool["xgboost"] = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, random_state=self.config.random_state
            )

        if HAS_GP and "gaussian_process" in self.config.model_types:
            self.model_pool["gaussian_process"] = GaussianProcessRegressor(
                kernel=RBF(), random_state=self.config.random_state
            )

        # SVM (can be slow, so smaller parameters)
        if "svm" in self.config.model_types:
            self.model_pool["svm"] = SVR(kernel="rbf", C=1.0, gamma="scale")

        # Filter models based on config
        self.model_pool = {
            name: model
            for name, model in self.model_pool.items()
            if name in self.config.model_types
        }

        logger.info(f"Initialized {len(self.model_pool)} models: {list(self.model_pool.keys())}")

    def select_model(self, data: pd.DataFrame, target: pd.Series) -> SelectionResult:
        """Select best model for current market conditions."""

        # Detect current market regime
        context = self.regime_detector.detect_regime(data)

        # Get available models
        available_models = list(self.model_pool.keys())

        # Select model using configured strategy
        result = self.model_selector.select_model(context, available_models)

        # Store selection
        self.selection_history.append(
            {
                "timestamp": time.time(),
                "context": context,
                "selected_model": result.selected_model,
                "confidence": result.selection_confidence,
                "alternatives": result.alternative_models,
            }
        )

        logger.info(
            f"Selected model: {result.selected_model} "
            f"(confidence: {result.selection_confidence:.3f}) "
            f"for regime: {context.volatility_regime}/{context.trend_regime}"
        )

        return result

    def fit_selected_model(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Any:
        """Fit the selected model on provided data."""
        if model_name not in self.model_pool:
            raise ValueError(f"Model {model_name} not in pool")

        model = self.model_pool[model_name]

        try:
            # Handle different model types
            X_clean = X.fillna(0)

            # Fit model
            fitted_model = model.fit(X_clean, y)

            logger.info(f"Successfully fitted {model_name}")
            return fitted_model

        except Exception as e:
            logger.error(f"Failed to fit {model_name}: {e}")
            # Fallback to linear model
            fallback_model = LinearRegression()
            return fallback_model.fit(X.fillna(0), y)

    def create_ensemble(
        self, data: pd.DataFrame, target: pd.Series, top_k: int | None = None
    ) -> dict[str, Any]:
        """Create ensemble from top-performing models."""

        if top_k is None:
            top_k = self.config.max_ensemble_size

        # Evaluate all models
        model_scores = {}
        model_predictions = {}

        # Use time series cross-validation
        if self.config.time_series_cv:
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        else:
            cv = self.config.cv_folds

        X_clean = data.fillna(0)

        for model_name, model in self.model_pool.items():
            try:
                scores = cross_val_score(
                    model, X_clean, target, cv=cv, scoring=self.config.scoring_metric, n_jobs=-1
                )
                model_scores[model_name] = np.mean(scores)

                # Get predictions for ensemble weighting
                model.fit(X_clean, target)
                predictions = model.predict(X_clean)
                model_predictions[model_name] = predictions

            except Exception as e:
                logger.warning(f"Failed to evaluate {model_name}: {e}")
                model_scores[model_name] = -1.0  # Low score

        # Select top k models
        top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_models = [name for name, score in top_models if score > -0.5]  # Filter failed models

        if len(top_models) < self.config.min_ensemble_size:
            logger.warning(f"Only {len(top_models)} models available for ensemble")

        # Calculate ensemble weights
        if self.config.ensemble_method == "equal":
            weights = {name: 1.0 / len(top_models) for name in top_models}
        elif self.config.ensemble_method == "performance_weighted":
            total_score = sum(model_scores[name] for name in top_models)
            weights = {name: max(model_scores[name], 0) / total_score for name in top_models}
        else:  # dynamic_weighted
            weights = self._calculate_dynamic_weights(top_models, model_predictions, target)

        self.ensemble_weights = weights

        return {
            "ensemble_models": top_models,
            "model_weights": weights,
            "model_scores": {name: model_scores[name] for name in top_models},
            "ensemble_size": len(top_models),
        }

    def _calculate_dynamic_weights(
        self, models: list[str], predictions: dict[str, np.ndarray], target: pd.Series
    ) -> dict[str, float]:
        """Calculate dynamic ensemble weights based on prediction diversity and performance."""

        weights = {}

        # Calculate individual performance and diversity
        for model_name in models:
            if model_name not in predictions:
                weights[model_name] = 0.0
                continue

            pred = predictions[model_name]

            # Individual performance (R²)
            individual_r2 = r2_score(target, pred)

            # Diversity score (negative correlation with ensemble mean)
            other_preds = [predictions[other] for other in models if other != model_name]
            if other_preds:
                ensemble_mean = np.mean(other_preds, axis=0)
                diversity = 1.0 - abs(np.corrcoef(pred, ensemble_mean)[0, 1])
            else:
                diversity = 0.5

            # Combined weight (performance + diversity)
            weights[model_name] = 0.7 * max(individual_r2, 0) + 0.3 * diversity

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: w / total_weight for name, w in weights.items()}
        else:
            # Equal weights fallback
            weights = {name: 1.0 / len(models) for name in models}

        return weights

    def predict_with_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using current ensemble."""
        if not self.ensemble_weights:
            raise ValueError("Ensemble not created. Call create_ensemble() first.")

        X_clean = X.fillna(0)
        ensemble_prediction = np.zeros(len(X))

        for model_name, weight in self.ensemble_weights.items():
            if weight > 0 and model_name in self.model_pool:
                model = self.model_pool[model_name]
                try:
                    pred = model.predict(X_clean)
                    ensemble_prediction += weight * pred
                except Exception as e:
                    logger.warning(f"Failed prediction from {model_name}: {e}")

        return ensemble_prediction

    def update_model_performance(
        self,
        model_name: str,
        X: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray,
        context: MarketContext | None = None,
    ) -> None:
        """Update model performance metrics."""

        if context is None:
            context = self.regime_detector.detect_regime(
                pd.DataFrame(
                    {
                        "Close": (
                            y_true.index.to_series()
                            if hasattr(y_true.index, "to_series")
                            else range(len(y_true))
                        )
                    }
                )
            )

        # Calculate performance metrics
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        # Composite performance score
        performance_score = r2 - 0.1 * np.sqrt(mse)  # Penalize high error

        # Update selector
        self.model_selector.update_performance(model_name, performance_score, context)

        # Store in history
        self.performance_history.append(
            {
                "timestamp": time.time(),
                "model": model_name,
                "r2": r2,
                "mse": mse,
                "mae": mae,
                "performance_score": performance_score,
                "context": context,
            }
        )

    def get_model_rankings(self) -> list[tuple[str, float]]:
        """Get current model rankings based on recent performance."""

        model_scores = {}

        for model_name in self.model_pool.keys():
            if model_name in self.model_selector.model_performance:
                perf = self.model_selector.model_performance[model_name]
                model_scores[model_name] = perf.overall_score
            else:
                model_scores[model_name] = 0.0

        # Sort by score
        rankings = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

        return rankings

    def get_selection_summary(self) -> dict[str, Any]:
        """Get summary of adaptive selection performance."""

        if not self.selection_history:
            return {"status": "no_selections_made"}

        # Selection statistics
        recent_selections = self.selection_history[-50:]
        model_selection_counts = defaultdict(int)
        avg_confidence = 0.0

        for selection in recent_selections:
            model_selection_counts[selection["selected_model"]] += 1
            avg_confidence += selection["confidence"]

        avg_confidence /= len(recent_selections)

        # Performance statistics
        recent_performance = list(self.performance_history)[-100:]
        if recent_performance:
            avg_r2 = np.mean([p["r2"] for p in recent_performance])
            avg_mse = np.mean([p["mse"] for p in recent_performance])
        else:
            avg_r2 = avg_mse = 0.0

        # Model rankings
        rankings = self.get_model_rankings()

        return {
            "total_selections": len(self.selection_history),
            "recent_avg_confidence": avg_confidence,
            "model_selection_frequency": dict(model_selection_counts),
            "current_rankings": rankings[:5],  # Top 5
            "recent_performance": {"avg_r2": avg_r2, "avg_mse": avg_mse},
            "active_models": len(self.model_pool),
            "ensemble_size": len(self.ensemble_weights) if self.ensemble_weights else 0,
        }


def create_adaptive_model_selection(
    config: AdaptiveModelConfig | None = None,
) -> AdaptiveModelSelectionFramework:
    """Create default adaptive model selection framework."""

    if config is None:
        config = AdaptiveModelConfig(
            model_types=(
                ["linear", "ridge", "random_forest", "xgboost"]
                if HAS_XGBOOST
                else ["linear", "ridge", "random_forest"]
            ),
            selection_method="contextual_bandit",
            bandit_algorithm="epsilon_greedy",
            ensemble_method="dynamic_weighted",
            use_market_context=True,
            online_learning=True,
        )

    return AdaptiveModelSelectionFramework(config)


# Integration example with trading strategy
class AdaptiveStrategySelector:
    """
    Adaptive strategy selector using model selection principles.

    Applies adaptive model selection to trading strategy selection,
    automatically choosing the best strategy for current market conditions.
    """

    def __init__(self, framework: AdaptiveModelSelectionFramework) -> None:
        self.framework = framework
        self.strategy_performance = defaultdict(list)
        self.current_strategy = None

    def register_strategy_performance(
        self, strategy_name: str, performance_metrics: dict[str, float], market_data: pd.DataFrame
    ) -> None:
        """Register strategy performance for selection learning."""

        # Extract market context
        context = self.framework.regime_detector.detect_regime(market_data)

        # Use Sharpe ratio as performance measure
        performance = performance_metrics.get("sharpe_ratio", 0.0)

        # Update framework (treating strategies as models)
        self.framework.model_selector.update_performance(strategy_name, performance, context)

        self.strategy_performance[strategy_name].append(
            {
                "performance": performance,
                "context": context,
                "timestamp": time.time(),
                "metrics": performance_metrics,
            }
        )

    def select_strategy(self, current_market_data: pd.DataFrame) -> dict[str, Any]:
        """Select best strategy for current market conditions."""

        # Get available strategies (registered in performance history)
        available_strategies = list(self.strategy_performance.keys())

        if not available_strategies:
            return {"error": "No strategies registered"}

        # Use framework to select strategy
        context = self.framework.regime_detector.detect_regime(current_market_data)

        try:
            result = self.framework.model_selector.select_model(context, available_strategies)

            self.current_strategy = result.selected_model

            return {
                "selected_strategy": result.selected_model,
                "confidence": result.selection_confidence,
                "market_regime": {
                    "volatility": context.volatility_regime,
                    "trend": context.trend_regime,
                    "volume": context.volume_regime,
                },
                "alternatives": result.alternative_models,
                "selection_time": result.selection_time,
            }

        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            return {"error": str(e), "fallback_strategy": available_strategies[0]}

    def get_strategy_performance_summary(self) -> dict[str, Any]:
        """Get summary of strategy selection performance."""

        summary = {}

        for strategy_name, performance_history in self.strategy_performance.items():
            if performance_history:
                performances = [p["performance"] for p in performance_history]
                summary[strategy_name] = {
                    "avg_performance": np.mean(performances),
                    "std_performance": np.std(performances),
                    "best_performance": np.max(performances),
                    "total_selections": len(performances),
                    "recent_performance": (
                        np.mean(performances[-10:])
                        if len(performances) >= 10
                        else np.mean(performances)
                    ),
                }

        return summary


# Example usage function
def demonstrate_adaptive_selection(
    market_data: pd.DataFrame, target_returns: pd.Series
) -> dict[str, Any]:
    """Demonstrate adaptive model selection on market data."""

    try:
        # Create adaptive framework
        framework = create_adaptive_model_selection()

        # Split data for training/testing
        split_idx = int(len(market_data) * 0.8)
        train_data = market_data.iloc[:split_idx]
        test_data = market_data.iloc[split_idx:]
        train_target = target_returns.iloc[:split_idx]
        test_target = target_returns.iloc[split_idx:]

        # Select model for test period
        selection_result = framework.select_model(test_data, test_target)

        # Fit selected model
        fitted_model = framework.fit_selected_model(
            selection_result.selected_model, train_data, train_target
        )

        # Create ensemble
        ensemble_info = framework.create_ensemble(train_data, train_target)

        # Make predictions
        selected_predictions = fitted_model.predict(test_data.fillna(0))
        ensemble_predictions = framework.predict_with_ensemble(test_data)

        # Evaluate performance
        selected_r2 = r2_score(test_target, selected_predictions)
        ensemble_r2 = r2_score(test_target, ensemble_predictions)

        # Update performance
        framework.update_model_performance(
            selection_result.selected_model, test_data, test_target, selected_predictions
        )

        # Get summary
        summary = framework.get_selection_summary()

        return {
            "demo_completed": True,
            "selected_model": selection_result.selected_model,
            "selection_confidence": selection_result.selection_confidence,
            "selected_model_r2": selected_r2,
            "ensemble_r2": ensemble_r2,
            "ensemble_info": ensemble_info,
            "framework_summary": summary,
        }

    except Exception as e:
        logger.error(f"Adaptive selection demo failed: {e}")
        return {"demo_completed": False, "error": str(e), "framework_created": True}
