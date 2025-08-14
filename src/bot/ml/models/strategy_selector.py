"""
XGBoost-based strategy selection model
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import optuna

from ..base import MLModel
from ...core.base import ComponentConfig


class StrategyMetaSelector(MLModel):
    """XGBoost model for selecting optimal trading strategies"""
    
    def __init__(self,
                 config: Optional[ComponentConfig] = None,
                 db_manager=None,
                 strategies: Optional[List[str]] = None):
        """Initialize strategy selector
        
        Args:
            config: Component configuration
            db_manager: Database manager
            strategies: List of available strategy names
        """
        if config is None:
            config = ComponentConfig(
                component_id='strategy_selector',
                component_type='ml_model'
            )
        super().__init__(config, db_manager)
        
        self.logger = logging.getLogger(__name__)
        
        # Available strategies
        self.strategies = strategies or [
            'momentum',
            'mean_reversion',
            'trend_following',
            'breakout',
            'pairs_trading',
            'volatility_arbitrage'
        ]
        
        # Label encoder for strategies
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.strategies)
        
        # Initialize XGBoost model
        self.model = None
        self.best_params = None
        
        # Feature importance tracking
        self.feature_importance = {}
        
        # Performance tracking
        self.strategy_performance = {}
        
    def train(self, X: pd.DataFrame, y: pd.Series, 
             optimize_hyperparams: bool = False) -> Dict[str, float]:
        """Train XGBoost strategy selector
        
        Args:
            X: Feature matrix
            y: Strategy labels (best performing strategy for each period)
            optimize_hyperparams: Whether to optimize hyperparameters
            
        Returns:
            Dictionary of training metrics
        """
        self.logger.info(f"Training strategy selector with {len(X)} samples")
        
        # Encode strategy labels
        y_encoded = self.label_encoder.transform(y)
        
        # Optimize hyperparameters if requested
        if optimize_hyperparams:
            self.logger.info("Optimizing hyperparameters with Optuna...")
            self.best_params = self._optimize_hyperparameters(X, y_encoded)
        else:
            # Use default parameters
            self.best_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            }
        
        # Train model with best parameters
        self.model = xgb.XGBClassifier(
            **self.best_params,
            objective='multi:softprob',
            num_class=len(self.strategies),
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False
        )
        
        # Fit model
        self.model.fit(
            X, y_encoded,
            eval_set=[(X, y_encoded)],
            eval_metric='mlogloss',
            early_stopping_rounds=10,
            verbose=False
        )
        
        self.is_trained = True
        
        # Calculate metrics
        train_score = self.model.score(X, y_encoded)
        predictions = self.model.predict(X)
        
        # Calculate per-strategy accuracy
        strategy_accuracy = {}
        for strategy_idx, strategy in enumerate(self.strategies):
            mask = y_encoded == strategy_idx
            if mask.sum() > 0:
                strategy_acc = (predictions[mask] == strategy_idx).mean()
                strategy_accuracy[strategy] = float(strategy_acc)
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_names = X.columns.tolist()
            self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        
        self.metrics = {
            'accuracy': float(train_score),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_strategies': len(self.strategies),
            'strategy_accuracy': strategy_accuracy,
            'best_iteration': self.model.best_iteration
        }
        
        self.logger.info(f"Training complete. Accuracy: {train_score:.3f}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict best strategy
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of strategy names
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Get predictions
        y_pred = self.model.predict(X)
        
        # Decode to strategy names
        strategies = self.label_encoder.inverse_transform(y_pred)
        
        # Store predictions if database available
        if self.db_manager:
            self._store_predictions(X.index, strategies)
        
        return strategies
    
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get strategy selection probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with probabilities for each strategy
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Get probabilities
        probs = self.model.predict_proba(X)
        
        # Create DataFrame with strategy names as columns
        prob_df = pd.DataFrame(
            probs,
            index=X.index,
            columns=self.strategies
        )
        
        return prob_df
    
    def select_strategy_with_confidence(self, X: pd.DataFrame) -> Tuple[str, float, Dict[str, float]]:
        """Select best strategy with confidence score
        
        Args:
            X: Feature matrix (single row expected)
            
        Returns:
            Tuple of (best_strategy, confidence, all_probabilities)
        """
        # Get probabilities
        probs_df = self.predict_proba(X)
        
        if len(probs_df) > 1:
            # Use last row if multiple rows provided
            probs = probs_df.iloc[-1]
        else:
            probs = probs_df.iloc[0]
        
        # Get best strategy and confidence
        best_strategy = probs.idxmax()
        confidence = probs[best_strategy]
        
        # Get all probabilities as dict
        all_probs = probs.to_dict()
        
        return best_strategy, float(confidence), all_probs
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters using Optuna
        
        Args:
            X: Feature matrix
            y: Encoded labels
            
        Returns:
            Dictionary of best hyperparameters
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
            }
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = xgb.XGBClassifier(
                    **params,
                    objective='multi:softprob',
                    num_class=len(self.strategies),
                    random_state=42,
                    n_jobs=-1,
                    use_label_encoder=False
                )
                
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='mlogloss',
                    early_stopping_rounds=10,
                    verbose=False
                )
                
                score = model.score(X_val, y_val)
                scores.append(score)
            
            return np.mean(scores)
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        
        self.logger.info(f"Best hyperparameters score: {study.best_value:.3f}")
        
        return study.best_params
    
    def analyze_strategy_performance(self, X: pd.DataFrame, y: pd.Series, 
                                    returns: pd.Series) -> Dict[str, Dict[str, float]]:
        """Analyze historical performance of each strategy
        
        Args:
            X: Feature matrix
            y: Actual best strategies used
            returns: Returns achieved
            
        Returns:
            Dictionary of performance metrics per strategy
        """
        performance = {}
        
        for strategy in self.strategies:
            mask = y == strategy
            if mask.sum() > 0:
                strategy_returns = returns[mask]
                
                performance[strategy] = {
                    'count': int(mask.sum()),
                    'frequency': float(mask.mean()),
                    'mean_return': float(strategy_returns.mean()),
                    'std_return': float(strategy_returns.std()),
                    'sharpe_ratio': float(strategy_returns.mean() / (strategy_returns.std() + 1e-10)),
                    'total_return': float((1 + strategy_returns).prod() - 1),
                    'win_rate': float((strategy_returns > 0).mean()),
                    'max_return': float(strategy_returns.max()),
                    'min_return': float(strategy_returns.min())
                }
            else:
                performance[strategy] = {
                    'count': 0,
                    'frequency': 0.0,
                    'mean_return': 0.0,
                    'std_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'total_return': 0.0,
                    'win_rate': 0.0,
                    'max_return': 0.0,
                    'min_return': 0.0
                }
        
        self.strategy_performance = performance
        return performance
    
    def get_feature_importance_analysis(self) -> Dict[str, Any]:
        """Analyze feature importance for strategy selection
        
        Returns:
            Dictionary with feature importance analysis
        """
        if not self.feature_importance:
            return {}
        
        # Sort features by importance
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Group features by type
        feature_groups = {
            'momentum': [],
            'volatility': [],
            'volume': [],
            'regime': [],
            'technical': [],
            'other': []
        }
        
        for feature, importance in sorted_features:
            if 'momentum' in feature or 'rsi' in feature or 'macd' in feature:
                group = 'momentum'
            elif 'volatility' in feature or 'atr' in feature or 'vol' in feature:
                group = 'volatility'
            elif 'volume' in feature:
                group = 'volume'
            elif 'regime' in feature:
                group = 'regime'
            elif any(ind in feature for ind in ['sma', 'ema', 'bb', 'adx']):
                group = 'technical'
            else:
                group = 'other'
            
            feature_groups[group].append({
                'name': feature,
                'importance': float(importance)
            })
        
        # Calculate group importance
        group_importance = {}
        for group, features in feature_groups.items():
            if features:
                group_importance[group] = sum(f['importance'] for f in features)
        
        analysis = {
            'top_features': sorted_features[:20],
            'feature_groups': feature_groups,
            'group_importance': group_importance,
            'total_features': len(self.feature_importance)
        }
        
        return analysis
    
    def create_ensemble_strategy(self, X: pd.DataFrame, 
                                threshold: float = 0.6) -> Dict[str, Any]:
        """Create ensemble strategy based on probability distribution
        
        Args:
            X: Feature matrix
            threshold: Minimum confidence threshold for single strategy
            
        Returns:
            Dictionary with ensemble strategy details
        """
        # Get probabilities
        probs_df = self.predict_proba(X)
        
        if len(probs_df) > 1:
            probs = probs_df.iloc[-1]
        else:
            probs = probs_df.iloc[0]
        
        # Check if any strategy has high confidence
        max_prob = probs.max()
        
        if max_prob >= threshold:
            # Use single strategy
            best_strategy = probs.idxmax()
            ensemble = {
                'type': 'single',
                'strategies': {best_strategy: 1.0},
                'confidence': float(max_prob),
                'reason': f'High confidence in {best_strategy}'
            }
        else:
            # Create weighted ensemble
            # Use strategies with probability > 0.2
            significant_strategies = probs[probs > 0.2]
            
            if len(significant_strategies) == 0:
                # Fallback to top 2 strategies
                top_2 = probs.nlargest(2)
                weights = (top_2 / top_2.sum()).to_dict()
            else:
                # Normalize to sum to 1
                weights = (significant_strategies / significant_strategies.sum()).to_dict()
            
            ensemble = {
                'type': 'ensemble',
                'strategies': weights,
                'confidence': float(max_prob),
                'reason': 'No single strategy has high confidence'
            }
        
        return ensemble
    
    def _store_predictions(self, timestamps, strategies: np.ndarray):
        """Store strategy predictions in database
        
        Args:
            timestamps: Index with timestamps
            strategies: Strategy predictions
        """
        if not self.db_manager:
            return
        
        for ts, strategy in zip(timestamps, strategies):
            try:
                self.db_manager.execute(
                    """INSERT OR REPLACE INTO ml_predictions 
                       (model_id, prediction_date, prediction_type, prediction_value, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (self.model_id, ts, 'strategy_selection', strategy, datetime.now())
                )
            except Exception as e:
                self.logger.error(f"Error storing prediction: {e}")