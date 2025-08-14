"""
Automated Feature Selection Framework for GPT-Trader Phase 1.

This module provides sophisticated feature selection techniques:
- Mutual Information-based selection
- Recursive Feature Elimination
- Statistical significance testing
- Feature importance tracking
- Time-series aware feature selection

Integrates with existing feature engineering pipeline and ensemble models.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    RFECV,
    f_regression,
    mutual_info_regression,
)
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Optional advanced feature selection
try:
    from skfeature.feature_selection import CFS, FCBF

    HAS_SCIKIT_FEATURE = True
except ImportError:
    HAS_SCIKIT_FEATURE = False
    warnings.warn("scikit-feature not available. Install with: pip install scikit-feature")

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from bot.utils.base import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """Container for feature importance information."""

    feature_name: str
    importance_score: float
    importance_type: str  # mutual_info, correlation, statistical, model_based
    p_value: float | None = None
    confidence_interval: tuple[float, float] | None = None
    stability_score: float | None = None  # Stability across time periods


@dataclass
class FeatureSelectionResult:
    """Results of feature selection process."""

    selected_features: list[str]
    feature_scores: dict[str, float]
    selection_method: str
    n_original_features: int
    n_selected_features: int
    selection_ratio: float
    feature_importances: list[FeatureImportance]
    performance_improvement: float | None = None


@dataclass
class FeatureSelectionConfig(BaseConfig):
    """Configuration for automated feature selection."""

    # Selection methods to use
    use_mutual_information: bool = True
    use_correlation_filter: bool = True
    use_statistical_tests: bool = True
    use_rfe: bool = True
    use_lasso_selection: bool = True
    use_stability_selection: bool = True

    # Method parameters
    mutual_info_k_best: int = 50
    correlation_threshold: float = 0.8
    correlation_method: str = "pearson"  # pearson, spearman, kendall
    statistical_alpha: float = 0.05

    # RFE parameters
    rfe_estimator: str = "random_forest"  # random_forest, xgboost, linear
    rfe_step: float = 0.1
    rfe_cv_folds: int = 5

    # Regularization parameters
    lasso_cv_folds: int = 5
    lasso_max_iter: int = 1000

    # Stability selection
    stability_threshold: float = 0.6
    stability_iterations: int = 100
    stability_subsample_ratio: float = 0.8

    # Time series parameters
    time_series_cv: bool = True
    time_window_size: int = 252  # Trading days in a year
    min_time_periods: int = 3

    # Performance validation
    validate_performance: bool = True
    performance_metric: str = "r2"  # r2, mse, mae

    # Output parameters
    max_features: int | None = None
    min_features: int = 10
    target_feature_ratio: float = 0.3  # Target ratio of selected features

    # Advanced options
    ensemble_selection: bool = True
    remove_correlated_features: bool = True
    feature_interaction_detection: bool = False


class BaseFeatureSelector(ABC):
    """Base class for feature selection methods."""

    def __init__(self, config: FeatureSelectionConfig) -> None:
        self.config = config
        self.selected_features_: list[str] | None = None
        self.feature_scores_: dict[str, float] | None = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseFeatureSelector:
        """Fit the feature selector."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features based on selection."""
        pass

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit selector and transform features."""
        return self.fit(X, y).transform(X)

    def get_support(self) -> list[bool]:
        """Get boolean mask of selected features."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted first")
        return [col in self.selected_features_ for col in self.feature_names_]


class MutualInformationSelector(BaseFeatureSelector):
    """Feature selection based on mutual information."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MutualInformationSelector:
        """Fit mutual information selector."""
        self.feature_names_ = list(X.columns)

        # Calculate mutual information scores
        mi_scores = mutual_info_regression(X, y, random_state=42)

        # Create feature-score mapping
        self.feature_scores_ = dict(zip(self.feature_names_, mi_scores, strict=False))

        # Select top k features
        k_best = min(self.config.mutual_info_k_best, len(self.feature_names_))
        top_features = sorted(self.feature_scores_.items(), key=lambda x: x[1], reverse=True)[
            :k_best
        ]

        self.selected_features_ = [feat for feat, score in top_features]

        logger.info(f"Mutual Information: selected {len(self.selected_features_)} features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting features."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted first")
        return X[self.selected_features_]


class CorrelationSelector(BaseFeatureSelector):
    """Feature selection based on correlation analysis."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> CorrelationSelector:
        """Fit correlation selector."""
        self.feature_names_ = list(X.columns)

        # Calculate correlations with target
        if self.config.correlation_method == "pearson":
            correlations = X.corrwith(y)
        elif self.config.correlation_method == "spearman":
            correlations = X.corrwith(y, method="spearman")
        else:
            raise ValueError(f"Unsupported correlation method: {self.config.correlation_method}")

        # Calculate absolute correlations for selection
        abs_correlations = correlations.abs()
        self.feature_scores_ = abs_correlations.to_dict()

        # Remove highly correlated features among themselves
        if self.config.remove_correlated_features:
            selected_features = self._remove_correlated_features(X, abs_correlations)
        else:
            # Select features above threshold
            selected_features = abs_correlations[
                abs_correlations >= self.config.correlation_threshold
            ].index.tolist()

        self.selected_features_ = selected_features

        logger.info(f"Correlation: selected {len(self.selected_features_)} features")
        return self

    def _remove_correlated_features(
        self, X: pd.DataFrame, target_correlations: pd.Series
    ) -> list[str]:
        """Remove highly correlated features while keeping most predictive ones."""
        # Calculate feature-feature correlation matrix
        corr_matrix = X.corr().abs()

        # Start with features sorted by target correlation
        sorted_features = target_correlations.sort_values(ascending=False).index.tolist()
        selected = []

        for feature in sorted_features:
            # Check if this feature is highly correlated with any already selected
            is_correlated = False
            for selected_feature in selected:
                if corr_matrix.loc[feature, selected_feature] > self.config.correlation_threshold:
                    is_correlated = True
                    break

            if not is_correlated:
                selected.append(feature)

        return selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting features."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted first")
        return X[self.selected_features_]


class StatisticalSelector(BaseFeatureSelector):
    """Feature selection based on statistical significance tests."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> StatisticalSelector:
        """Fit statistical selector using various statistical tests."""
        self.feature_names_ = list(X.columns)

        # Use F-statistic for regression
        f_stats, p_values = f_regression(X, y)

        # Store results
        self.feature_scores_ = dict(zip(self.feature_names_, f_stats, strict=False))
        self.p_values_ = dict(zip(self.feature_names_, p_values, strict=False))

        # Select features with p-value below threshold
        significant_features = [
            feat for feat, p_val in self.p_values_.items() if p_val < self.config.statistical_alpha
        ]

        self.selected_features_ = significant_features

        logger.info(f"Statistical: selected {len(self.selected_features_)} features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting features."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted first")
        return X[self.selected_features_]


class RFESelector(BaseFeatureSelector):
    """Recursive Feature Elimination selector."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RFESelector:
        """Fit RFE selector."""
        self.feature_names_ = list(X.columns)

        # Choose base estimator
        if self.config.rfe_estimator == "random_forest":
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        elif self.config.rfe_estimator == "xgboost" and HAS_XGBOOST:
            estimator = xgb.XGBRegressor(n_estimators=50, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)

        # Use cross-validation for optimal number of features
        if self.config.time_series_cv:
            cv = TimeSeriesSplit(n_splits=self.config.rfe_cv_folds)
        else:
            cv = self.config.rfe_cv_folds

        rfe = RFECV(estimator=estimator, step=self.config.rfe_step, cv=cv, scoring="r2", n_jobs=-1)

        rfe.fit(X, y)

        # Get selected features
        self.selected_features_ = X.columns[rfe.support_].tolist()

        # Get feature rankings as scores (inverse ranking for consistency)
        max_rank = max(rfe.ranking_)
        self.feature_scores_ = {
            feat: max_rank - rank + 1
            for feat, rank in zip(self.feature_names_, rfe.ranking_, strict=False)
        }

        logger.info(f"RFE: selected {len(self.selected_features_)} features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting features."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted first")
        return X[self.selected_features_]


class LassoSelector(BaseFeatureSelector):
    """Feature selection using LASSO regularization."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LassoSelector:
        """Fit LASSO selector."""
        self.feature_names_ = list(X.columns)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Use LASSO with cross-validation
        if self.config.time_series_cv:
            cv = TimeSeriesSplit(n_splits=self.config.lasso_cv_folds)
        else:
            cv = self.config.lasso_cv_folds

        lasso = LassoCV(cv=cv, max_iter=self.config.lasso_max_iter, random_state=42, n_jobs=-1)

        lasso.fit(X_scaled, y)

        # Get features with non-zero coefficients
        non_zero_coefs = lasso.coef_ != 0
        self.selected_features_ = X.columns[non_zero_coefs].tolist()

        # Use absolute coefficients as feature scores
        self.feature_scores_ = dict(zip(self.feature_names_, np.abs(lasso.coef_), strict=False))

        logger.info(f"LASSO: selected {len(self.selected_features_)} features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting features."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted first")
        return X[self.selected_features_]


class StabilitySelector(BaseFeatureSelector):
    """Stability selection for robust feature selection."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> StabilitySelector:
        """Fit stability selector."""
        self.feature_names_ = list(X.columns)

        feature_selection_counts = {feat: 0 for feat in self.feature_names_}

        # Perform multiple iterations with subsampling
        for i in range(self.config.stability_iterations):
            # Subsample data
            n_samples = int(len(X) * self.config.stability_subsample_ratio)
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sub = X.iloc[indices]
            y_sub = y.iloc[indices]

            # Use LASSO for feature selection
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_sub)

            lasso = LassoCV(cv=3, random_state=42 + i)
            lasso.fit(X_scaled, y_sub)

            # Count selected features
            selected = X_sub.columns[lasso.coef_ != 0]
            for feat in selected:
                feature_selection_counts[feat] += 1

        # Calculate selection probabilities
        selection_probs = {
            feat: count / self.config.stability_iterations
            for feat, count in feature_selection_counts.items()
        }

        # Select features above stability threshold
        self.selected_features_ = [
            feat
            for feat, prob in selection_probs.items()
            if prob >= self.config.stability_threshold
        ]

        self.feature_scores_ = selection_probs

        logger.info(f"Stability: selected {len(self.selected_features_)} features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting features."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted first")
        return X[self.selected_features_]


class AutomatedFeatureSelector:
    """
    Automated feature selection framework combining multiple methods.

    This class integrates various feature selection techniques and provides
    ensemble-based selection for robust feature identification.
    """

    def __init__(self, config: FeatureSelectionConfig) -> None:
        self.config = config
        self.selectors: dict[str, BaseFeatureSelector] = {}
        self.final_features_: list[str] | None = None
        self.feature_importances_: list[FeatureImportance] | None = None
        self.selection_results_: dict[str, FeatureSelectionResult] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> AutomatedFeatureSelector:
        """
        Fit all configured feature selection methods.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Fitted selector
        """
        logger.info("Starting automated feature selection...")

        # Initialize selectors based on configuration
        if self.config.use_mutual_information:
            self.selectors["mutual_info"] = MutualInformationSelector(self.config)

        if self.config.use_correlation_filter:
            self.selectors["correlation"] = CorrelationSelector(self.config)

        if self.config.use_statistical_tests:
            self.selectors["statistical"] = StatisticalSelector(self.config)

        if self.config.use_rfe:
            self.selectors["rfe"] = RFESelector(self.config)

        if self.config.use_lasso_selection:
            self.selectors["lasso"] = LassoSelector(self.config)

        if self.config.use_stability_selection:
            self.selectors["stability"] = StabilitySelector(self.config)

        # Fit each selector
        self.selection_results_ = {}
        for name, selector in self.selectors.items():
            try:
                logger.info(f"Fitting {name} selector...")
                selector.fit(X, y)

                # Store results
                self.selection_results_[name] = FeatureSelectionResult(
                    selected_features=selector.selected_features_,
                    feature_scores=selector.feature_scores_,
                    selection_method=name,
                    n_original_features=len(X.columns),
                    n_selected_features=len(selector.selected_features_),
                    selection_ratio=len(selector.selected_features_) / len(X.columns),
                    feature_importances=[],
                )

            except Exception as e:
                logger.error(f"Failed to fit {name} selector: {e}")
                del self.selectors[name]

        # Combine results using ensemble approach
        if self.config.ensemble_selection:
            self._ensemble_selection(X.columns.tolist())
        else:
            # Use the best performing method
            self._select_best_method(X, y)

        logger.info(f"Feature selection completed. Selected {len(self.final_features_)} features")
        return self

    def _ensemble_selection(self, all_features: list[str]) -> None:
        """Combine selections from multiple methods using voting."""
        feature_votes = {feat: 0 for feat in all_features}
        feature_scores_combined = {feat: [] for feat in all_features}

        # Collect votes and scores from each method
        for _name, result in self.selection_results_.items():
            for feat in result.selected_features:
                feature_votes[feat] += 1

            # Normalize scores to 0-1 range for fair combination
            max_score = max(result.feature_scores.values())
            min_score = min(result.feature_scores.values())
            score_range = max_score - min_score

            for feat, score in result.feature_scores.items():
                if score_range > 0:
                    normalized_score = (score - min_score) / score_range
                else:
                    normalized_score = 1.0
                feature_scores_combined[feat].append(normalized_score)

        # Calculate ensemble scores
        ensemble_scores = {}
        for feat in all_features:
            if feature_scores_combined[feat]:
                ensemble_scores[feat] = np.mean(feature_scores_combined[feat])
            else:
                ensemble_scores[feat] = 0.0

        # Select features based on votes and scores
        min_votes = max(1, len(self.selectors) // 2)  # Majority voting

        candidate_features = [feat for feat, votes in feature_votes.items() if votes >= min_votes]

        # If we have too many features, select top by ensemble score
        if self.config.max_features and len(candidate_features) > self.config.max_features:
            candidate_features = sorted(
                candidate_features, key=lambda x: ensemble_scores[x], reverse=True
            )[: self.config.max_features]

        # Ensure minimum number of features
        if len(candidate_features) < self.config.min_features:
            # Add top features by ensemble score
            remaining_features = [feat for feat in all_features if feat not in candidate_features]
            remaining_features.sort(key=lambda x: ensemble_scores[x], reverse=True)

            needed = self.config.min_features - len(candidate_features)
            candidate_features.extend(remaining_features[:needed])

        self.final_features_ = candidate_features

        # Create feature importance objects
        self.feature_importances_ = [
            FeatureImportance(
                feature_name=feat,
                importance_score=ensemble_scores[feat],
                importance_type="ensemble",
                stability_score=feature_votes[feat] / len(self.selectors),
            )
            for feat in self.final_features_
        ]

    def _select_best_method(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Select features from the best performing method."""
        if not self.config.validate_performance:
            # Just use the first available method
            first_method = next(iter(self.selection_results_.values()))
            self.final_features_ = first_method.selected_features
            return

        best_performance = -np.inf
        best_features = None

        # Evaluate each method's performance
        for name, result in self.selection_results_.items():
            try:
                # Simple train-test validation
                split_idx = int(len(X) * 0.8)
                X_train = X.iloc[:split_idx][result.selected_features]
                X_test = X.iloc[split_idx:][result.selected_features]
                y_train = y.iloc[:split_idx]
                y_test = y.iloc[split_idx:]

                # Train simple model for evaluation
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                # Calculate performance
                if self.config.performance_metric == "r2":
                    performance = r2_score(y_test, predictions)
                elif self.config.performance_metric == "mse":
                    performance = -mean_squared_error(y_test, predictions)
                else:
                    performance = r2_score(y_test, predictions)

                # Update results
                result.performance_improvement = performance

                if performance > best_performance:
                    best_performance = performance
                    best_features = result.selected_features

            except Exception as e:
                logger.warning(f"Performance evaluation failed for {name}: {e}")

        self.final_features_ = best_features or list(X.columns[: self.config.min_features])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting final features."""
        if self.final_features_ is None:
            raise ValueError("Selector must be fitted first")
        return X[self.final_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit selector and transform features."""
        return self.fit(X, y).transform(X)

    def get_selected_features(self) -> list[str]:
        """Get list of selected features."""
        if self.final_features_ is None:
            raise ValueError("Selector must be fitted first")
        return self.final_features_.copy()

    def get_feature_importances(self) -> list[FeatureImportance]:
        """Get feature importance information."""
        if self.feature_importances_ is None:
            raise ValueError("Selector must be fitted first")
        return self.feature_importances_.copy()

    def get_selection_summary(self) -> dict[str, Any]:
        """Get summary of feature selection process."""
        if self.final_features_ is None:
            raise ValueError("Selector must be fitted first")

        summary = {
            "n_original_features": len(
                next(iter(self.selection_results_.values())).selected_features
            ),
            "n_final_features": len(self.final_features_),
            "selection_ratio": len(self.final_features_)
            / len(next(iter(self.selection_results_.values())).selected_features),
            "methods_used": list(self.selection_results_.keys()),
            "final_features": self.final_features_.copy(),
        }

        # Add method-specific results
        for name, result in self.selection_results_.items():
            summary[f"{name}_n_features"] = result.n_selected_features
            summary[f"{name}_selection_ratio"] = result.selection_ratio
            if result.performance_improvement is not None:
                summary[f"{name}_performance"] = result.performance_improvement

        return summary


def create_default_feature_selector() -> AutomatedFeatureSelector:
    """Create a default feature selector configuration."""
    config = FeatureSelectionConfig(
        use_mutual_information=True,
        use_correlation_filter=True,
        use_statistical_tests=True,
        use_rfe=True,
        use_lasso_selection=True,
        use_stability_selection=True,
        ensemble_selection=True,
        validate_performance=True,
    )

    return AutomatedFeatureSelector(config)
