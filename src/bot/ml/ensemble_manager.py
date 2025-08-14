"""
Ensemble Model Management System
Phase 3, Week 6: ADAPT-017 to ADAPT-024
Dynamic weighting, Bayesian averaging, and ensemble optimization
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


class EnsembleMethod(Enum):
    """Ensemble combination methods"""

    VOTING = "voting"
    WEIGHTED = "weighted"
    STACKING = "stacking"
    BAYESIAN = "bayesian"
    DYNAMIC = "dynamic"
    BLENDING = "blending"


class WeightingStrategy(Enum):
    """Strategies for model weighting"""

    EQUAL = "equal"
    PERFORMANCE = "performance"
    INVERSE_ERROR = "inverse_error"
    BAYESIAN = "bayesian"
    OPTIMAL = "optimal"
    ADAPTIVE = "adaptive"


class DiversityMetric(Enum):
    """Metrics for measuring ensemble diversity"""

    DISAGREEMENT = "disagreement"
    CORRELATION = "correlation"
    Q_STATISTIC = "q_statistic"
    ENTROPY = "entropy"
    KL_DIVERGENCE = "kl_divergence"


@dataclass
class ModelMetadata:
    """Metadata for individual models in ensemble"""

    model_id: str
    model_type: str
    created_at: datetime

    # Performance metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    sharpe_ratio: float = 0.0

    # Weight and contribution
    weight: float = 1.0
    contribution: float = 0.0
    reliability: float = 1.0

    # Training info
    training_samples: int = 0
    feature_set: list[str] = field(default_factory=list)
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    # Performance history
    performance_history: list[float] = field(default_factory=list)
    prediction_history: deque = field(default_factory=lambda: deque(maxlen=1000))


@dataclass
class EnsemblePerformance:
    """Performance metrics for ensemble"""

    timestamp: datetime
    method: EnsembleMethod

    # Overall metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    # Financial metrics
    sharpe_ratio: float
    max_drawdown: float
    total_return: float

    # Ensemble specific
    diversity_score: float
    agreement_ratio: float
    effective_models: int

    # Model contributions
    model_weights: dict[str, float]
    model_contributions: dict[str, float]


class DynamicWeightOptimizer:
    """Optimize ensemble weights dynamically"""

    def __init__(self, optimization_method: str = "scipy", regularization: float = 0.01):
        """
        Initialize weight optimizer.

        Args:
            optimization_method: Method for optimization
            regularization: L2 regularization strength
        """
        self.optimization_method = optimization_method
        self.regularization = regularization
        self.weight_history = []

    def optimize_weights(
        self, predictions: np.ndarray, targets: np.ndarray, constraints: dict | None = None
    ) -> np.ndarray:
        """
        Optimize ensemble weights.

        Args:
            predictions: Model predictions (n_samples, n_models)
            targets: True targets
            constraints: Weight constraints

        Returns:
            Optimal weights
        """
        n_models = predictions.shape[1]

        # Define objective function
        def objective(weights):
            ensemble_pred = np.average(predictions, weights=weights, axis=1)
            mse = np.mean((ensemble_pred - targets) ** 2)
            # Add L2 regularization
            reg_term = self.regularization * np.sum(weights**2)
            return mse + reg_term

        # Constraints
        constraints_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]  # Sum to 1

        if constraints:
            if "min_weight" in constraints:
                constraints_list.append(
                    {"type": "ineq", "fun": lambda w: w - constraints["min_weight"]}
                )
            if "max_weight" in constraints:
                constraints_list.append(
                    {"type": "ineq", "fun": lambda w: constraints["max_weight"] - w}
                )

        # Bounds
        bounds = [(0, 1) for _ in range(n_models)]

        # Initial weights
        initial_weights = np.ones(n_models) / n_models

        # Optimize
        result = minimize(
            objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints_list
        )

        if result.success:
            optimal_weights = result.x
        else:
            logger.warning("Optimization failed, using equal weights")
            optimal_weights = initial_weights

        self.weight_history.append(optimal_weights)
        return optimal_weights

    def adaptive_weighting(
        self, performance_history: dict[str, list[float]], lookback: int = 20
    ) -> dict[str, float]:
        """
        Adaptive weighting based on recent performance.

        Args:
            performance_history: Historical performance by model
            lookback: Number of periods to consider

        Returns:
            Adaptive weights
        """
        weights = {}

        for model_id, history in performance_history.items():
            if len(history) >= lookback:
                recent_performance = history[-lookback:]
            else:
                recent_performance = history

            if recent_performance:
                # Weight based on Sharpe ratio of recent performance
                returns = np.diff(recent_performance) / recent_performance[:-1]
                if len(returns) > 0:
                    sharpe = np.mean(returns) / (np.std(returns) + 1e-6)
                    weights[model_id] = max(0, sharpe)  # Only positive weights
                else:
                    weights[model_id] = 1.0
            else:
                weights[model_id] = 1.0

        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # Equal weights if all negative
            n_models = len(weights)
            weights = {k: 1 / n_models for k in weights.keys()}

        return weights


class BayesianModelAveraging:
    """Bayesian Model Averaging for ensemble"""

    def __init__(self, prior_strength: float = 1.0):
        """
        Initialize BMA.

        Args:
            prior_strength: Strength of prior beliefs
        """
        self.prior_strength = prior_strength
        self.posterior_weights = {}
        self.evidence_history = defaultdict(list)

    def update_posterior(
        self, model_id: str, likelihood: float, prior: float | None = None
    ) -> float:
        """
        Update posterior probability for model.

        Args:
            model_id: Model identifier
            likelihood: Likelihood of data given model
            prior: Prior probability

        Returns:
            Posterior probability
        """
        if prior is None:
            prior = self.posterior_weights.get(model_id, 1.0)

        # Bayes rule: P(M|D) ∝ P(D|M) * P(M)
        posterior = likelihood * prior

        self.evidence_history[model_id].append(likelihood)
        self.posterior_weights[model_id] = posterior

        return posterior

    def calculate_bma_weights(self, model_likelihoods: dict[str, float]) -> dict[str, float]:
        """
        Calculate BMA weights from likelihoods.

        Args:
            model_likelihoods: Likelihood for each model

        Returns:
            BMA weights
        """
        # Update posteriors
        for model_id, likelihood in model_likelihoods.items():
            self.update_posterior(model_id, likelihood)

        # Normalize to get weights
        total_posterior = sum(self.posterior_weights.values())

        if total_posterior > 0:
            weights = {
                model_id: post / total_posterior
                for model_id, post in self.posterior_weights.items()
            }
        else:
            # Equal weights fallback
            n_models = len(model_likelihoods)
            weights = {model_id: 1 / n_models for model_id in model_likelihoods}

        return weights

    def calculate_model_likelihood(
        self, predictions: np.ndarray, targets: np.ndarray, method: str = "gaussian"
    ) -> float:
        """
        Calculate likelihood of model predictions.

        Args:
            predictions: Model predictions
            targets: True values
            method: Likelihood calculation method

        Returns:
            Likelihood value
        """
        errors = predictions - targets

        if method == "gaussian":
            # Gaussian likelihood
            sigma = np.std(errors) + 1e-6
            likelihood = np.exp(-0.5 * np.sum(errors**2) / sigma**2)
        elif method == "laplace":
            # Laplace likelihood (robust to outliers)
            scale = np.median(np.abs(errors)) + 1e-6
            likelihood = np.exp(-np.sum(np.abs(errors)) / scale)
        else:
            # Binary likelihood for classification
            correct = (predictions == targets).astype(float)
            likelihood = np.mean(correct)

        return likelihood


class DiversityAnalyzer:
    """Analyze and optimize ensemble diversity"""

    def __init__(self):
        """Initialize diversity analyzer"""
        self.diversity_metrics = {}

    def calculate_disagreement(self, predictions: list[np.ndarray]) -> float:
        """
        Calculate disagreement measure.

        Args:
            predictions: List of model predictions

        Returns:
            Disagreement score
        """
        n_models = len(predictions)
        n_samples = len(predictions[0])

        disagreements = 0
        for i in range(n_samples):
            votes = [pred[i] for pred in predictions]
            # Count pairs that disagree
            for j in range(n_models):
                for k in range(j + 1, n_models):
                    if votes[j] != votes[k]:
                        disagreements += 1

        # Normalize
        max_disagreements = n_samples * n_models * (n_models - 1) / 2
        disagreement_score = disagreements / max_disagreements if max_disagreements > 0 else 0

        return disagreement_score

    def calculate_correlation_diversity(self, predictions: list[np.ndarray]) -> float:
        """
        Calculate diversity based on prediction correlations.

        Args:
            predictions: List of model predictions

        Returns:
            Correlation-based diversity
        """
        n_models = len(predictions)

        if n_models < 2:
            return 0.0

        # Calculate pairwise correlations
        correlations = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                correlations.append(abs(corr))

        # Diversity is inverse of average correlation
        avg_correlation = np.mean(correlations)
        diversity = 1.0 - avg_correlation

        return diversity

    def calculate_q_statistic(
        self, pred1: np.ndarray, pred2: np.ndarray, targets: np.ndarray
    ) -> float:
        """
        Calculate Q-statistic for two models.

        Args:
            pred1: First model predictions
            pred2: Second model predictions
            targets: True targets

        Returns:
            Q-statistic
        """
        # Create contingency table
        both_correct = np.sum((pred1 == targets) & (pred2 == targets))
        both_wrong = np.sum((pred1 != targets) & (pred2 != targets))
        first_only = np.sum((pred1 == targets) & (pred2 != targets))
        second_only = np.sum((pred1 != targets) & (pred2 == targets))

        # Q-statistic
        numerator = both_correct * both_wrong - first_only * second_only
        denominator = both_correct * both_wrong + first_only * second_only

        if denominator != 0:
            q_stat = numerator / denominator
        else:
            q_stat = 0

        return q_stat

    def calculate_entropy_diversity(self, predictions: list[np.ndarray]) -> float:
        """
        Calculate entropy-based diversity.

        Args:
            predictions: List of model predictions

        Returns:
            Entropy diversity score
        """
        n_models = len(predictions)
        n_samples = len(predictions[0])

        entropy_scores = []

        for i in range(n_samples):
            votes = [pred[i] for pred in predictions]
            # Calculate vote distribution
            unique, counts = np.unique(votes, return_counts=True)
            probabilities = counts / n_models

            # Calculate entropy
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            entropy_scores.append(entropy)

        # Average entropy
        avg_entropy = np.mean(entropy_scores)

        # Normalize by maximum possible entropy
        max_entropy = np.log2(n_models)
        normalized_entropy = avg_entropy / max_entropy if max_entropy > 0 else 0

        return normalized_entropy


class StackingMetaLearner:
    """Stacking ensemble with meta-learner"""

    def __init__(
        self,
        meta_model: BaseEstimator | None = None,
        use_probabilities: bool = True,
        cv_folds: int = 5,
    ):
        """
        Initialize stacking meta-learner.

        Args:
            meta_model: Meta-learning model
            use_probabilities: Use probabilities instead of predictions
            cv_folds: Number of CV folds for meta-training
        """
        self.meta_model = meta_model
        self.use_probabilities = use_probabilities
        self.cv_folds = cv_folds
        self.is_fitted = False

        # Default to logistic regression if not provided
        if self.meta_model is None:
            from sklearn.linear_model import LogisticRegression

            self.meta_model = LogisticRegression()

    def create_meta_features(
        self,
        base_predictions: list[np.ndarray],
        include_original: bool = False,
        original_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Create meta-features from base model predictions.

        Args:
            base_predictions: Predictions from base models
            include_original: Include original features
            original_features: Original feature matrix

        Returns:
            Meta-feature matrix
        """
        # Stack base predictions
        meta_features = np.column_stack(base_predictions)

        # Add statistical features
        stats_features = []
        stats_features.append(np.mean(meta_features, axis=1))
        stats_features.append(np.std(meta_features, axis=1))
        stats_features.append(np.max(meta_features, axis=1))
        stats_features.append(np.min(meta_features, axis=1))

        meta_features = np.column_stack([meta_features] + stats_features)

        # Include original features if requested
        if include_original and original_features is not None:
            meta_features = np.column_stack([meta_features, original_features])

        return meta_features

    def fit(
        self,
        base_predictions: list[np.ndarray],
        targets: np.ndarray,
        original_features: np.ndarray | None = None,
    ):
        """
        Fit meta-learner.

        Args:
            base_predictions: Predictions from base models
            targets: True targets
            original_features: Original features (optional)
        """
        # Create meta-features
        meta_features = self.create_meta_features(
            base_predictions,
            include_original=(original_features is not None),
            original_features=original_features,
        )

        # Fit meta-model
        self.meta_model.fit(meta_features, targets)
        self.is_fitted = True

    def predict(
        self, base_predictions: list[np.ndarray], original_features: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Make predictions using meta-learner.

        Args:
            base_predictions: Predictions from base models
            original_features: Original features (optional)

        Returns:
            Meta-learner predictions
        """
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted first")

        # Create meta-features
        meta_features = self.create_meta_features(
            base_predictions,
            include_original=(original_features is not None),
            original_features=original_features,
        )

        # Predict
        return self.meta_model.predict(meta_features)


class EnsembleManager:
    """Main ensemble management system"""

    def __init__(self, method: EnsembleMethod = EnsembleMethod.DYNAMIC, max_models: int = 10):
        """
        Initialize ensemble manager.

        Args:
            method: Ensemble combination method
            max_models: Maximum number of models in ensemble
        """
        self.method = method
        self.max_models = max_models

        # Model storage
        self.models: dict[str, BaseEstimator] = {}
        self.model_metadata: dict[str, ModelMetadata] = {}

        # Components
        self.weight_optimizer = DynamicWeightOptimizer()
        self.bayesian_averager = BayesianModelAveraging()
        self.diversity_analyzer = DiversityAnalyzer()
        self.meta_learner = StackingMetaLearner()

        # Performance tracking
        self.performance_history: list[EnsemblePerformance] = []
        self.current_weights: dict[str, float] = {}

        # Configuration
        self.min_weight = 0.01  # Minimum weight for any model
        self.diversity_threshold = 0.3  # Minimum diversity required
        self.prune_threshold = 0.05  # Prune models below this weight

    def add_model(
        self, model_id: str, model: BaseEstimator, metadata: ModelMetadata | None = None
    ) -> bool:
        """
        Add model to ensemble.

        Args:
            model_id: Unique model identifier
            model: Trained model
            metadata: Model metadata

        Returns:
            Success status
        """
        if len(self.models) >= self.max_models:
            logger.warning(f"Maximum models ({self.max_models}) reached")
            # Prune worst performing model
            self._prune_worst_model()

        self.models[model_id] = model

        if metadata is None:
            metadata = ModelMetadata(
                model_id=model_id, model_type=type(model).__name__, created_at=datetime.now()
            )

        self.model_metadata[model_id] = metadata

        # Initialize weight
        n_models = len(self.models)
        self.current_weights = {mid: 1 / n_models for mid in self.models.keys()}

        logger.info(f"Added model {model_id} to ensemble")
        return True

    def remove_model(self, model_id: str) -> bool:
        """
        Remove model from ensemble.

        Args:
            model_id: Model to remove

        Returns:
            Success status
        """
        if model_id in self.models:
            del self.models[model_id]
            del self.model_metadata[model_id]

            # Rebalance weights
            if self.models:
                total_weight = sum(w for mid, w in self.current_weights.items() if mid != model_id)
                self.current_weights = {
                    mid: w / total_weight
                    for mid, w in self.current_weights.items()
                    if mid != model_id
                }

            logger.info(f"Removed model {model_id} from ensemble")
            return True

        return False

    def predict(self, X: np.ndarray, method: EnsembleMethod | None = None) -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            X: Input features
            method: Override ensemble method

        Returns:
            Ensemble predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        method = method or self.method

        # Get base predictions
        base_predictions = []
        model_ids = []

        for model_id, model in self.models.items():
            try:
                pred = model.predict(X)
                base_predictions.append(pred)
                model_ids.append(model_id)
            except Exception as e:
                logger.error(f"Model {model_id} prediction failed: {e}")

        if not base_predictions:
            raise ValueError("All model predictions failed")

        # Combine predictions based on method
        if method == EnsembleMethod.VOTING:
            # Simple majority voting
            ensemble_pred = stats.mode(base_predictions, axis=0)[0][0]

        elif method == EnsembleMethod.WEIGHTED:
            # Weighted average
            weights = [self.current_weights.get(mid, 1.0) for mid in model_ids]
            ensemble_pred = np.average(base_predictions, weights=weights, axis=0)

        elif method == EnsembleMethod.BAYESIAN:
            # Bayesian model averaging
            ensemble_pred = self._bayesian_predict(base_predictions, model_ids)

        elif method == EnsembleMethod.STACKING:
            # Stacking with meta-learner
            if self.meta_learner.is_fitted:
                ensemble_pred = self.meta_learner.predict(base_predictions)
            else:
                # Fallback to weighted average
                weights = [self.current_weights.get(mid, 1.0) for mid in model_ids]
                ensemble_pred = np.average(base_predictions, weights=weights, axis=0)

        elif method == EnsembleMethod.DYNAMIC:
            # Dynamic weighting based on recent performance
            ensemble_pred = self._dynamic_predict(base_predictions, model_ids, X)

        else:  # BLENDING
            # Blend multiple methods
            ensemble_pred = self._blended_predict(base_predictions, model_ids, X)

        return ensemble_pred

    def _dynamic_predict(
        self, base_predictions: list[np.ndarray], model_ids: list[str], X: np.ndarray
    ) -> np.ndarray:
        """
        Dynamic prediction with adaptive weights.

        Args:
            base_predictions: Base model predictions
            model_ids: Model identifiers
            X: Input features

        Returns:
            Dynamic ensemble predictions
        """
        # Update weights based on recent performance
        performance_history = {
            mid: self.model_metadata[mid].performance_history for mid in model_ids
        }

        adaptive_weights = self.weight_optimizer.adaptive_weighting(
            performance_history, lookback=20
        )

        # Apply diversity bonus
        diversity = self.diversity_analyzer.calculate_correlation_diversity(base_predictions)

        if diversity < self.diversity_threshold:
            # Low diversity - increase weight variance
            weights = list(adaptive_weights.values())
            weights = np.array(weights)
            weights = weights**2  # Square to increase variance
            weights = weights / weights.sum()  # Renormalize
            adaptive_weights = dict(zip(adaptive_weights.keys(), weights, strict=False))

        # Create weighted prediction
        weights = [adaptive_weights.get(mid, 1.0) for mid in model_ids]
        ensemble_pred = np.average(base_predictions, weights=weights, axis=0)

        # Update current weights
        self.current_weights.update(adaptive_weights)

        return ensemble_pred

    def _bayesian_predict(
        self, base_predictions: list[np.ndarray], model_ids: list[str]
    ) -> np.ndarray:
        """
        Bayesian model averaging prediction.

        Args:
            base_predictions: Base model predictions
            model_ids: Model identifiers

        Returns:
            BMA predictions
        """
        # Calculate likelihoods (simplified - using recent accuracy)
        model_likelihoods = {}
        for mid in model_ids:
            metadata = self.model_metadata[mid]
            # Use recent accuracy as likelihood proxy
            likelihood = metadata.accuracy if metadata.accuracy > 0 else 0.5
            model_likelihoods[mid] = likelihood

        # Get BMA weights
        bma_weights = self.bayesian_averager.calculate_bma_weights(model_likelihoods)

        # Apply weights
        weights = [bma_weights.get(mid, 1.0) for mid in model_ids]
        ensemble_pred = np.average(base_predictions, weights=weights, axis=0)

        return ensemble_pred

    def _blended_predict(
        self, base_predictions: list[np.ndarray], model_ids: list[str], X: np.ndarray
    ) -> np.ndarray:
        """
        Blended prediction combining multiple methods.

        Args:
            base_predictions: Base model predictions
            model_ids: Model identifiers
            X: Input features

        Returns:
            Blended predictions
        """
        predictions = []

        # Weighted average
        weights = [self.current_weights.get(mid, 1.0) for mid in model_ids]
        weighted_pred = np.average(base_predictions, weights=weights, axis=0)
        predictions.append(weighted_pred)

        # Bayesian
        bayesian_pred = self._bayesian_predict(base_predictions, model_ids)
        predictions.append(bayesian_pred)

        # Dynamic
        dynamic_pred = self._dynamic_predict(base_predictions, model_ids, X)
        predictions.append(dynamic_pred)

        # Blend with equal weights
        ensemble_pred = np.mean(predictions, axis=0)

        return ensemble_pred

    def update_model_performance(
        self, model_id: str, predictions: np.ndarray, targets: np.ndarray
    ) -> None:
        """
        Update model performance metrics.

        Args:
            model_id: Model identifier
            predictions: Model predictions
            targets: True targets
        """
        if model_id not in self.model_metadata:
            return

        metadata = self.model_metadata[model_id]

        # Calculate metrics
        if len(np.unique(targets)) == 2:  # Binary classification
            metadata.accuracy = accuracy_score(targets, predictions > 0.5)
            metadata.precision = precision_score(targets, predictions > 0.5)
            metadata.recall = recall_score(targets, predictions > 0.5)
            metadata.f1_score = f1_score(targets, predictions > 0.5)
        else:  # Regression
            # Use correlation as accuracy proxy
            metadata.accuracy = np.corrcoef(predictions, targets)[0, 1]

        # Update history
        metadata.performance_history.append(metadata.accuracy)
        metadata.prediction_history.extend(predictions)

        # Update reliability based on consistency
        if len(metadata.performance_history) > 10:
            recent_performance = metadata.performance_history[-10:]
            metadata.reliability = 1.0 - np.std(recent_performance)

    def optimize_ensemble(self, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
        """
        Optimize ensemble configuration.

        Args:
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Optimized weights
        """
        # Get base predictions
        base_predictions = []
        model_ids = []

        for model_id, model in self.models.items():
            try:
                pred = model.predict(X_val)
                base_predictions.append(pred)
                model_ids.append(model_id)
            except Exception as e:
                logger.error(f"Model {model_id} prediction failed: {e}")

        if len(base_predictions) < 2:
            logger.warning("Not enough models for optimization")
            return self.current_weights

        # Optimize weights
        predictions_array = np.column_stack(base_predictions)
        optimal_weights = self.weight_optimizer.optimize_weights(
            predictions_array, y_val, constraints={"min_weight": self.min_weight}
        )

        # Update weights
        self.current_weights = dict(zip(model_ids, optimal_weights, strict=False))

        # Check diversity
        diversity = self.diversity_analyzer.calculate_correlation_diversity(base_predictions)

        if diversity < self.diversity_threshold:
            logger.warning(f"Low ensemble diversity: {diversity:.3f}")
            # Consider pruning similar models
            self._prune_similar_models(base_predictions, model_ids)

        return self.current_weights

    def _prune_worst_model(self) -> None:
        """Prune worst performing model"""
        if not self.model_metadata:
            return

        # Find worst model based on recent performance
        worst_model = None
        worst_score = float("inf")

        for model_id, metadata in self.model_metadata.items():
            if metadata.performance_history:
                recent_score = np.mean(metadata.performance_history[-10:])
                if recent_score < worst_score:
                    worst_score = recent_score
                    worst_model = model_id

        if worst_model:
            self.remove_model(worst_model)
            logger.info(f"Pruned worst model: {worst_model}")

    def _prune_similar_models(self, predictions: list[np.ndarray], model_ids: list[str]) -> None:
        """
        Prune highly correlated models.

        Args:
            predictions: Model predictions
            model_ids: Model identifiers
        """
        n_models = len(predictions)

        # Calculate correlation matrix
        corr_matrix = np.zeros((n_models, n_models))
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    corr_matrix[i, j] = np.corrcoef(predictions[i], predictions[j])[0, 1]

        # Find highly correlated pairs
        high_corr_threshold = 0.95
        pruned = set()

        for i in range(n_models):
            if model_ids[i] in pruned:
                continue

            for j in range(i + 1, n_models):
                if model_ids[j] in pruned:
                    continue

                if abs(corr_matrix[i, j]) > high_corr_threshold:
                    # Keep better performing model
                    perf_i = self.model_metadata[model_ids[i]].accuracy
                    perf_j = self.model_metadata[model_ids[j]].accuracy

                    if perf_i > perf_j:
                        self.remove_model(model_ids[j])
                        pruned.add(model_ids[j])
                        logger.info(f"Pruned similar model: {model_ids[j]}")
                    else:
                        self.remove_model(model_ids[i])
                        pruned.add(model_ids[i])
                        logger.info(f"Pruned similar model: {model_ids[i]}")
                        break

    def get_ensemble_report(self) -> str:
        """
        Generate ensemble performance report.

        Returns:
            Formatted report
        """
        report = []
        report.append("=" * 60)
        report.append("ENSEMBLE MANAGEMENT REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Ensemble Method: {self.method.value}")
        report.append(f"Number of Models: {len(self.models)}")
        report.append("")

        # Model details
        report.append("MODEL PERFORMANCE")
        report.append("-" * 40)

        for model_id, metadata in self.model_metadata.items():
            weight = self.current_weights.get(model_id, 0)
            report.append(f"  {model_id}:")
            report.append(f"    Type: {metadata.model_type}")
            report.append(f"    Weight: {weight:.3f}")
            report.append(f"    Accuracy: {metadata.accuracy:.3f}")
            report.append(f"    F1 Score: {metadata.f1_score:.3f}")
            report.append(f"    Reliability: {metadata.reliability:.3f}")
            report.append("")

        # Ensemble metrics
        if self.performance_history:
            latest = self.performance_history[-1]
            report.append("ENSEMBLE METRICS")
            report.append("-" * 40)
            report.append(f"  Overall Accuracy: {latest.accuracy:.3f}")
            report.append(f"  F1 Score: {latest.f1_score:.3f}")
            report.append(f"  Sharpe Ratio: {latest.sharpe_ratio:.3f}")
            report.append(f"  Diversity Score: {latest.diversity_score:.3f}")
            report.append(f"  Effective Models: {latest.effective_models}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    ensemble = EnsembleManager(method=EnsembleMethod.DYNAMIC)

    # Add some dummy models
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(random_state=42)

    # Simulate trained models
    ensemble.add_model("rf_model", rf)
    ensemble.add_model("gb_model", gb)
    ensemble.add_model("lr_model", lr)

    # Generate report
    report = ensemble.get_ensemble_report()
    print(report)
