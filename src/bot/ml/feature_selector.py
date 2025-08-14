"""
Advanced Feature Selection Module
Phase 2.5 - Day 6

Implements multiple feature selection methods with aggressive reduction.
Integrates with existing feature_engineering_v2.py
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import warnings
from pathlib import Path
import json

from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, mutual_info_regression,
    RFE, RFECV, VarianceThreshold, SelectFromModel
)
from sklearn.linear_model import Lasso, LassoCV, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SelectionMethod(Enum):
    """Feature selection methods"""
    MUTUAL_INFORMATION = "mutual_information"
    LASSO = "lasso"
    RECURSIVE_ELIMINATION = "recursive_elimination"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    ENSEMBLE = "ensemble"


@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection"""
    # Target number of features
    n_features_target: int = 50
    n_features_min: int = 30
    n_features_max: int = 70
    
    # Correlation threshold (more aggressive than before)
    correlation_threshold: float = 0.7  # Was 0.95 in v1
    
    # Selection methods
    primary_method: SelectionMethod = SelectionMethod.ENSEMBLE
    ensemble_methods: List[SelectionMethod] = None
    
    # Lasso parameters
    lasso_alpha: float = 0.01
    lasso_cv_folds: int = 5
    
    # RFE parameters
    rfe_step: float = 0.1  # Remove 10% features each step
    rfe_cv: bool = True
    
    # Mutual information parameters
    mi_neighbors: int = 3
    mi_random_state: int = 42
    
    # Validation
    cv_folds: int = 5
    min_importance_threshold: float = 0.001
    
    def __post_init__(self):
        if self.ensemble_methods is None:
            self.ensemble_methods = [
                SelectionMethod.MUTUAL_INFORMATION,
                SelectionMethod.LASSO,
                SelectionMethod.RANDOM_FOREST
            ]


class AdvancedFeatureSelector:
    """
    Advanced feature selection with multiple methods.
    Integrates with existing feature engineering pipeline.
    
    Key improvements over v1:
    - More aggressive correlation removal (0.7 vs 0.95)
    - Multiple selection methods with ensemble voting
    - Feature stability tracking across methods
    - Automatic hyperparameter tuning
    """
    
    def __init__(self, config: Optional[FeatureSelectionConfig] = None):
        self.config = config or FeatureSelectionConfig()
        self.selected_features: Dict[str, List[str]] = {}
        self.feature_scores: Dict[str, pd.DataFrame] = {}
        self.selection_history: List[Dict] = []
        
        logger.info(f"AdvancedFeatureSelector initialized with target {self.config.n_features_target} features")
    
    def select_features(self, 
                        X: pd.DataFrame, 
                        y: pd.Series,
                        method: Optional[SelectionMethod] = None) -> Tuple[List[str], pd.DataFrame]:
        """
        Select features using specified method.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Selection method to use
            
        Returns:
            Tuple of (selected_features, importance_scores)
        """
        if method is None:
            method = self.config.primary_method
        
        logger.info(f"Selecting features using {method.value} method")
        
        # Remove low variance features first
        X_filtered = self._remove_low_variance(X)
        
        # Remove highly correlated features
        X_filtered = self._remove_correlated_features(X_filtered)
        
        # Apply selection method
        if method == SelectionMethod.MUTUAL_INFORMATION:
            features, scores = self._select_by_mutual_information(X_filtered, y)
        elif method == SelectionMethod.LASSO:
            features, scores = self._select_by_lasso(X_filtered, y)
        elif method == SelectionMethod.RECURSIVE_ELIMINATION:
            features, scores = self._select_by_rfe(X_filtered, y)
        elif method == SelectionMethod.RANDOM_FOREST:
            features, scores = self._select_by_random_forest(X_filtered, y)
        elif method == SelectionMethod.XGBOOST:
            features, scores = self._select_by_xgboost(X_filtered, y)
        elif method == SelectionMethod.ENSEMBLE:
            features, scores = self._select_by_ensemble(X_filtered, y)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # Store results
        self.selected_features[method.value] = features
        self.feature_scores[method.value] = scores
        
        # Log selection
        self.selection_history.append({
            'method': method.value,
            'n_features_input': len(X.columns),
            'n_features_filtered': len(X_filtered.columns),
            'n_features_selected': len(features),
            'timestamp': pd.Timestamp.now()
        })
        
        logger.info(f"Selected {len(features)} features from {len(X.columns)} input features")
        
        return features, scores
    
    def _remove_low_variance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove features with low variance"""
        selector = VarianceThreshold(threshold=self.config.min_importance_threshold)
        X_transformed = selector.fit_transform(X.fillna(0))
        
        selected_features = X.columns[selector.get_support()].tolist()
        
        logger.debug(f"Removed {len(X.columns) - len(selected_features)} low variance features")
        
        return X[selected_features]
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features (more aggressive than v1)"""
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find pairs above threshold
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Track features to drop
        to_drop = set()
        
        # For each column, check if highly correlated with others
        for column in upper_tri.columns:
            # Find correlated features
            correlated = upper_tri[column][upper_tri[column] > self.config.correlation_threshold].index.tolist()
            
            if correlated and column not in to_drop:
                # Keep the feature with lower average correlation with others
                candidates = [column] + correlated
                avg_correlations = {}
                
                for candidate in candidates:
                    if candidate not in to_drop:
                        avg_corr = corr_matrix[candidate].mean()
                        avg_correlations[candidate] = avg_corr
                
                # Keep feature with lowest average correlation
                if avg_correlations:
                    keep_feature = min(avg_correlations, key=avg_correlations.get)
                    for feature in candidates:
                        if feature != keep_feature:
                            to_drop.add(feature)
        
        logger.info(f"Removing {len(to_drop)} correlated features (threshold={self.config.correlation_threshold})")
        
        return X.drop(columns=list(to_drop))
    
    def _select_by_mutual_information(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], pd.DataFrame]:
        """Select features using mutual information"""
        # Determine if classification or regression
        is_classification = len(np.unique(y)) <= 10
        
        if is_classification:
            mi_scores = mutual_info_classif(
                X.fillna(0), y,
                n_neighbors=self.config.mi_neighbors,
                random_state=self.config.mi_random_state
            )
        else:
            mi_scores = mutual_info_regression(
                X.fillna(0), y,
                n_neighbors=self.config.mi_neighbors,
                random_state=self.config.mi_random_state
            )
        
        # Create scores DataFrame
        scores_df = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        # Select top features
        n_features = min(self.config.n_features_target, len(scores_df))
        selected = scores_df.head(n_features)['feature'].tolist()
        
        return selected, scores_df
    
    def _select_by_lasso(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], pd.DataFrame]:
        """Select features using Lasso regularization"""
        # Scale features for Lasso
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.fillna(0))
        
        # Use LassoCV to find optimal alpha
        lasso_cv = LassoCV(
            cv=self.config.lasso_cv_folds,
            random_state=42,
            max_iter=1000
        )
        lasso_cv.fit(X_scaled, y)
        
        # Get coefficients
        coefficients = np.abs(lasso_cv.coef_)
        
        # Create scores DataFrame
        scores_df = pd.DataFrame({
            'feature': X.columns,
            'lasso_coef': coefficients
        }).sort_values('lasso_coef', ascending=False)
        
        # Select features with non-zero coefficients
        selected = scores_df[scores_df['lasso_coef'] > 0]['feature'].tolist()
        
        # If too many features, take top N
        if len(selected) > self.config.n_features_max:
            selected = scores_df.head(self.config.n_features_target)['feature'].tolist()
        # If too few, add more
        elif len(selected) < self.config.n_features_min:
            selected = scores_df.head(self.config.n_features_min)['feature'].tolist()
        
        logger.info(f"Lasso selected {len(selected)} features with alpha={lasso_cv.alpha_:.4f}")
        
        return selected, scores_df
    
    def _select_by_rfe(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], pd.DataFrame]:
        """Select features using Recursive Feature Elimination"""
        # Use XGBoost as base estimator
        estimator = XGBRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        if self.config.rfe_cv:
            # Use cross-validation to find optimal number
            selector = RFECV(
                estimator=estimator,
                step=self.config.rfe_step,
                cv=TimeSeriesSplit(n_splits=self.config.cv_folds),
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
        else:
            # Use fixed number of features
            selector = RFE(
                estimator=estimator,
                n_features_to_select=self.config.n_features_target,
                step=self.config.rfe_step
            )
        
        selector.fit(X.fillna(0), y)
        
        # Get rankings
        scores_df = pd.DataFrame({
            'feature': X.columns,
            'rfe_ranking': selector.ranking_,
            'selected': selector.support_
        }).sort_values('rfe_ranking')
        
        # Get selected features
        selected = X.columns[selector.support_].tolist()
        
        logger.info(f"RFE selected {len(selected)} features")
        
        return selected, scores_df
    
    def _select_by_random_forest(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], pd.DataFrame]:
        """Select features using Random Forest importance"""
        # Determine task type
        is_classification = len(np.unique(y)) <= 10
        
        if is_classification:
            rf = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                max_depth=10
            )
        else:
            rf = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                max_depth=10
            )
        
        rf.fit(X.fillna(0), y)
        
        # Get feature importance
        scores_df = pd.DataFrame({
            'feature': X.columns,
            'rf_importance': rf.feature_importances_
        }).sort_values('rf_importance', ascending=False)
        
        # Select top features
        selected = scores_df.head(self.config.n_features_target)['feature'].tolist()
        
        return selected, scores_df
    
    def _select_by_xgboost(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], pd.DataFrame]:
        """Select features using XGBoost importance"""
        # Determine task type
        is_classification = len(np.unique(y)) <= 10
        
        if is_classification:
            xgb = XGBClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        else:
            xgb = XGBRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        
        xgb.fit(X.fillna(0), y)
        
        # Get feature importance
        importance_dict = xgb.get_booster().get_score(importance_type='gain')
        
        # Map to feature names
        scores_df = pd.DataFrame({
            'feature': X.columns,
            'xgb_importance': [importance_dict.get(f'f{i}', 0) for i in range(len(X.columns))]
        }).sort_values('xgb_importance', ascending=False)
        
        # Select top features
        selected = scores_df.head(self.config.n_features_target)['feature'].tolist()
        
        return selected, scores_df
    
    def _select_by_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], pd.DataFrame]:
        """Select features using ensemble of methods"""
        all_scores = {}
        all_selected = {}
        
        # Run each method
        for method in self.config.ensemble_methods:
            logger.info(f"Running {method.value} for ensemble")
            
            if method == SelectionMethod.MUTUAL_INFORMATION:
                selected, scores = self._select_by_mutual_information(X, y)
            elif method == SelectionMethod.LASSO:
                selected, scores = self._select_by_lasso(X, y)
            elif method == SelectionMethod.RANDOM_FOREST:
                selected, scores = self._select_by_random_forest(X, y)
            elif method == SelectionMethod.XGBOOST:
                selected, scores = self._select_by_xgboost(X, y)
            else:
                continue
            
            all_selected[method.value] = selected
            all_scores[method.value] = scores
        
        # Combine scores using voting
        feature_votes = {}
        for feature in X.columns:
            votes = 0
            total_rank = 0
            
            for method, selected in all_selected.items():
                if feature in selected:
                    votes += 1
                    # Add rank-based score (1 for top feature, 0 for last)
                    rank = selected.index(feature) if feature in selected else len(selected)
                    total_rank += (len(selected) - rank) / len(selected)
            
            feature_votes[feature] = {
                'votes': votes,
                'avg_rank_score': total_rank / len(all_selected) if votes > 0 else 0,
                'ensemble_score': votes + total_rank / len(all_selected)
            }
        
        # Create ensemble scores DataFrame
        ensemble_df = pd.DataFrame(feature_votes).T
        ensemble_df.index.name = 'feature'
        ensemble_df = ensemble_df.sort_values('ensemble_score', ascending=False)
        
        # Select features that appear in at least 2 methods or top by score
        min_votes = max(1, len(self.config.ensemble_methods) // 2)
        selected_by_votes = ensemble_df[ensemble_df['votes'] >= min_votes].index.tolist()
        
        # If not enough features, add by ensemble score
        if len(selected_by_votes) < self.config.n_features_target:
            additional = ensemble_df.head(self.config.n_features_target).index.tolist()
            selected_features = list(dict.fromkeys(selected_by_votes + additional))[:self.config.n_features_target]
        else:
            selected_features = selected_by_votes[:self.config.n_features_target]
        
        logger.info(f"Ensemble selected {len(selected_features)} features with {min_votes}+ votes")
        
        return selected_features, ensemble_df.reset_index()
    
    def analyze_feature_stability(self, X: pd.DataFrame, y: pd.Series, 
                                 n_iterations: int = 10) -> pd.DataFrame:
        """
        Analyze feature stability across multiple runs.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_iterations: Number of bootstrap iterations
            
        Returns:
            DataFrame with stability scores
        """
        feature_selections = {feature: [] for feature in X.columns}
        
        for i in range(n_iterations):
            # Bootstrap sample
            sample_idx = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
            
            # Select features
            selected, _ = self.select_features(X_sample, y_sample)
            
            # Track selections
            for feature in X.columns:
                feature_selections[feature].append(1 if feature in selected else 0)
        
        # Calculate stability metrics
        stability_df = pd.DataFrame({
            'feature': X.columns,
            'selection_frequency': [np.mean(feature_selections[f]) for f in X.columns],
            'stability_score': [1 - np.std(feature_selections[f]) for f in X.columns]
        }).sort_values('selection_frequency', ascending=False)
        
        return stability_df
    
    def get_feature_importance_report(self) -> pd.DataFrame:
        """Get comprehensive feature importance report"""
        if not self.feature_scores:
            return pd.DataFrame()
        
        # Combine all scores
        combined_scores = None
        
        for method, scores in self.feature_scores.items():
            scores_renamed = scores.copy()
            
            # Rename score columns to include method
            score_cols = [col for col in scores.columns if col != 'feature']
            for col in score_cols:
                scores_renamed[f'{method}_{col}'] = scores_renamed[col]
                scores_renamed.drop(col, axis=1, inplace=True)
            
            if combined_scores is None:
                combined_scores = scores_renamed
            else:
                combined_scores = pd.merge(
                    combined_scores, scores_renamed, 
                    on='feature', how='outer'
                )
        
        # Add selection status
        for method, features in self.selected_features.items():
            combined_scores[f'{method}_selected'] = combined_scores['feature'].isin(features)
        
        return combined_scores
    
    def save_selection_config(self, filepath: str):
        """Save feature selection configuration and results"""
        config_dict = {
            'config': {
                'n_features_target': self.config.n_features_target,
                'correlation_threshold': self.config.correlation_threshold,
                'primary_method': self.config.primary_method.value,
                'ensemble_methods': [m.value for m in self.config.ensemble_methods]
            },
            'selected_features': self.selected_features,
            'selection_history': self.selection_history,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"Feature selection config saved to {filepath}")
    
    def load_selection_config(self, filepath: str) -> Dict:
        """Load feature selection configuration"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        self.selected_features = config_dict.get('selected_features', {})
        self.selection_history = config_dict.get('selection_history', [])
        
        logger.info(f"Feature selection config loaded from {filepath}")
        
        return config_dict


def integrate_with_feature_engineering(selector: AdvancedFeatureSelector,
                                      engineer: Any) -> List[str]:
    """
    Integrate feature selector with existing feature engineering pipeline.
    
    Args:
        selector: Feature selector instance
        engineer: Feature engineer instance from feature_engineering_v2.py
        
    Returns:
        List of selected features
    """
    # Get the selected features from engineer
    if hasattr(engineer, 'selected_features'):
        current_features = engineer.selected_features
        
        # Apply additional selection
        logger.info(f"Refining {len(current_features)} features from engineering pipeline")
        
        # Update engineer's selected features
        if hasattr(engineer, 'feature_importance'):
            # Use importance scores for better selection
            importance_df = pd.DataFrame(
                engineer.feature_importance.items(),
                columns=['feature', 'importance']
            )
            
            # Apply correlation threshold
            refined_features = selector._remove_correlated_features(
                pd.DataFrame(columns=current_features)
            ).columns.tolist()
            
            engineer.selected_features = refined_features
            
            return refined_features
    
    return engineer.selected_features if hasattr(engineer, 'selected_features') else []


def create_feature_selector(config: Optional[FeatureSelectionConfig] = None) -> AdvancedFeatureSelector:
    """Create feature selector instance"""
    return AdvancedFeatureSelector(config)


if __name__ == "__main__":
    # Test with sample data
    from sklearn.datasets import make_regression
    
    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=200, n_informative=50, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(200)])
    y = pd.Series(y)
    
    # Create selector with aggressive settings
    config = FeatureSelectionConfig(
        n_features_target=50,
        correlation_threshold=0.7,  # More aggressive
        primary_method=SelectionMethod.ENSEMBLE
    )
    
    selector = create_feature_selector(config)
    
    # Select features
    selected_features, scores = selector.select_features(X, y)
    
    print(f"Selected {len(selected_features)} features from {len(X.columns)}")
    print(f"Top 10 features: {selected_features[:10]}")
    
    # Analyze stability
    stability = selector.analyze_feature_stability(X, y, n_iterations=5)
    print("\nTop 10 most stable features:")
    print(stability.head(10))
    
    # Get importance report
    report = selector.get_feature_importance_report()
    print("\nFeature importance report shape:", report.shape)
    
    # Save configuration
    selector.save_selection_config('feature_selection_config.json')