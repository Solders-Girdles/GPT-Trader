"""
Walk-Forward Validation Framework
Phase 2.5 - Day 7

Implements proper walk-forward analysis with backtesting on each fold
and model degradation tracking.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_percentage_error
)
from sklearn.base import BaseEstimator, clone

# Import our backtesting framework
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest.realistic_backtester import (
    RealisticBacktester, BacktestConfig, BacktestResults
)

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation"""
    # Window sizes (in trading days)
    train_window: int = 504  # 2 years
    test_window: int = 126   # 6 months
    step_size: int = 21      # 1 month
    min_train_size: int = 252  # Minimum 1 year of training data
    
    # Validation settings
    expanding_window: bool = True  # True: expanding, False: rolling
    purge_gap: int = 5  # Days between train and test to avoid lookahead
    
    # Backtesting on each fold
    backtest_each_fold: bool = True
    backtest_config: Optional[BacktestConfig] = None
    
    # Performance tracking
    track_degradation: bool = True
    degradation_threshold: float = 0.1  # 10% performance drop triggers alert
    
    # Model settings
    retrain_on_degradation: bool = True
    save_fold_models: bool = True
    model_save_path: str = "walk_forward_models"
    
    def __post_init__(self):
        if self.backtest_config is None:
            self.backtest_config = BacktestConfig(
                initial_capital=100000,
                position_size=0.1,
                stop_loss=0.02,
                take_profit=0.05
            )


@dataclass
class FoldResult:
    """Results for a single walk-forward fold"""
    fold_number: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Model performance
    train_accuracy: float
    test_accuracy: float
    train_f1: float
    test_f1: float
    
    # Additional metrics
    precision: float
    recall: float
    roc_auc: float
    
    # Backtesting results
    backtest_results: Optional[BacktestResults] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_return: Optional[float] = None
    
    # Degradation tracking
    performance_change: Optional[float] = None  # vs previous fold
    is_degraded: bool = False
    
    # Feature importance
    feature_importance: Optional[Dict[str, float]] = None
    
    # Model info
    model_params: Optional[Dict] = None
    training_time: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert dates to strings
        result['train_start'] = self.train_start.isoformat()
        result['train_end'] = self.train_end.isoformat()
        result['test_start'] = self.test_start.isoformat()
        result['test_end'] = self.test_end.isoformat()
        # Handle BacktestResults
        if self.backtest_results:
            result['backtest_results'] = {
                'total_return': self.backtest_results.total_return,
                'sharpe_ratio': self.backtest_results.sharpe_ratio,
                'max_drawdown': self.backtest_results.max_drawdown,
                'win_rate': self.backtest_results.win_rate,
                'total_trades': self.backtest_results.total_trades
            }
        return result


@dataclass
class WalkForwardResults:
    """Complete walk-forward validation results"""
    model_name: str
    n_folds: int
    config: WalkForwardConfig
    fold_results: List[FoldResult]
    
    # Aggregate metrics
    mean_test_accuracy: float
    std_test_accuracy: float
    mean_test_f1: float
    std_test_f1: float
    
    # Backtesting aggregate
    mean_sharpe: Optional[float] = None
    mean_return: Optional[float] = None
    mean_drawdown: Optional[float] = None
    
    # Degradation analysis
    degradation_detected: bool = False
    degradation_folds: List[int] = field(default_factory=list)
    stability_score: float = 0.0  # 0-1, higher is more stable
    
    # Feature stability
    feature_importance_stability: Optional[Dict[str, float]] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert fold results to DataFrame"""
        fold_data = []
        for fold in self.fold_results:
            fold_dict = {
                'fold': fold.fold_number,
                'train_start': fold.train_start,
                'train_end': fold.train_end,
                'test_start': fold.test_start,
                'test_end': fold.test_end,
                'train_accuracy': fold.train_accuracy,
                'test_accuracy': fold.test_accuracy,
                'train_f1': fold.train_f1,
                'test_f1': fold.test_f1,
                'precision': fold.precision,
                'recall': fold.recall,
                'roc_auc': fold.roc_auc,
                'sharpe_ratio': fold.sharpe_ratio,
                'max_drawdown': fold.max_drawdown,
                'total_return': fold.total_return,
                'is_degraded': fold.is_degraded
            }
            fold_data.append(fold_dict)
        
        return pd.DataFrame(fold_data)


class WalkForwardValidator:
    """
    Advanced walk-forward validation with backtesting and degradation tracking.
    
    Features:
    - Proper time series cross-validation with purging
    - Backtesting on each fold for realistic performance
    - Model degradation detection across folds
    - Feature importance stability tracking
    - Automatic retraining on degradation
    """
    
    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()
        self.fold_models = []
        self.fold_predictions = {}
        
        # Create model save directory if needed
        if self.config.save_fold_models:
            Path(self.config.model_save_path).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"WalkForwardValidator initialized with {self.config.train_window} day train window")
    
    def validate(self,
                model: BaseEstimator,
                X: pd.DataFrame,
                y: pd.Series,
                prices: Optional[pd.Series] = None,
                model_name: str = "model") -> WalkForwardResults:
        """
        Perform walk-forward validation.
        
        Args:
            model: Scikit-learn compatible model
            X: Feature matrix with datetime index
            y: Target variable with datetime index
            prices: Optional price series for backtesting
            model_name: Name for tracking
            
        Returns:
            Walk-forward validation results
        """
        logger.info(f"Starting walk-forward validation for {model_name}")
        
        # Ensure datetime index
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex")
        
        # Generate folds
        folds = self._generate_folds(X, y)
        logger.info(f"Generated {len(folds)} folds for validation")
        
        # Track results
        fold_results = []
        previous_accuracy = None
        
        # Validate each fold
        for i, (train_idx, test_idx) in enumerate(folds):
            logger.info(f"Processing fold {i+1}/{len(folds)}...")
            
            # Get fold data
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            
            # Validate fold
            fold_result = self._validate_fold(
                model=clone(model),  # Clone to avoid modifying original
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                prices=prices.iloc[test_idx] if prices is not None else None,
                fold_number=i,
                previous_accuracy=previous_accuracy
            )
            
            fold_results.append(fold_result)
            previous_accuracy = fold_result.test_accuracy
            
            # Check for degradation
            if fold_result.is_degraded and self.config.retrain_on_degradation:
                logger.warning(f"Degradation detected in fold {i}. Triggering retraining...")
                # In production, this would trigger model retraining
        
        # Calculate aggregate metrics
        results = self._aggregate_results(fold_results, model_name)
        
        # Analyze feature stability
        if any(f.feature_importance for f in fold_results):
            results.feature_importance_stability = self._analyze_feature_stability(fold_results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _generate_folds(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward validation folds"""
        folds = []
        n_samples = len(X)
        
        # Start position ensures minimum training size
        start_pos = self.config.min_train_size
        
        while start_pos + self.config.test_window <= n_samples:
            if self.config.expanding_window:
                # Expanding window: use all data up to this point
                train_start = 0
            else:
                # Rolling window: fixed size training window
                train_start = max(0, start_pos - self.config.train_window)
            
            # Training indices (with purge gap)
            train_end = start_pos - self.config.purge_gap
            train_idx = np.arange(train_start, train_end)
            
            # Test indices
            test_start = start_pos
            test_end = min(start_pos + self.config.test_window, n_samples)
            test_idx = np.arange(test_start, test_end)
            
            folds.append((train_idx, test_idx))
            
            # Move to next fold
            start_pos += self.config.step_size
        
        return folds
    
    def _validate_fold(self,
                      model: BaseEstimator,
                      X_train: pd.DataFrame,
                      y_train: pd.Series,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      prices: Optional[pd.Series],
                      fold_number: int,
                      previous_accuracy: Optional[float]) -> FoldResult:
        """Validate a single fold"""
        
        # Train model
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store model if configured
        if self.config.save_fold_models:
            self._save_fold_model(model, fold_number)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Store predictions for later analysis
        self.fold_predictions[fold_number] = {
            'y_test': y_test,
            'y_pred': y_test_pred
        }
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Handle both binary and multiclass
        avg_method = 'binary' if len(np.unique(y_train)) == 2 else 'weighted'
        
        train_f1 = f1_score(y_train, y_train_pred, average=avg_method)
        test_f1 = f1_score(y_test, y_test_pred, average=avg_method)
        precision = precision_score(y_test, y_test_pred, average=avg_method, zero_division=0)
        recall = recall_score(y_test, y_test_pred, average=avg_method, zero_division=0)
        
        # ROC-AUC for binary classification
        roc_auc = 0
        if len(np.unique(y_train)) == 2 and hasattr(model, 'predict_proba'):
            try:
                y_test_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_test_proba)
            except:
                pass
        
        # Check for degradation
        performance_change = None
        is_degraded = False
        if previous_accuracy is not None:
            performance_change = test_accuracy - previous_accuracy
            is_degraded = performance_change < -self.config.degradation_threshold
            
            if is_degraded:
                logger.warning(f"Performance degradation detected: {performance_change:.3f}")
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        
        # Create fold result
        fold_result = FoldResult(
            fold_number=fold_number,
            train_start=X_train.index[0],
            train_end=X_train.index[-1],
            test_start=X_test.index[0],
            test_end=X_test.index[-1],
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            train_f1=train_f1,
            test_f1=test_f1,
            precision=precision,
            recall=recall,
            roc_auc=roc_auc,
            performance_change=performance_change,
            is_degraded=is_degraded,
            feature_importance=feature_importance,
            training_time=training_time
        )
        
        # Run backtesting if configured
        if self.config.backtest_each_fold and prices is not None:
            fold_result = self._backtest_fold(model, X_test, y_test, prices, fold_result)
        
        return fold_result
    
    def _backtest_fold(self,
                      model: BaseEstimator,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      prices: pd.Series,
                      fold_result: FoldResult) -> FoldResult:
        """Run backtesting on a fold"""
        try:
            # Generate trading signals from predictions
            y_pred = model.predict(X_test)
            
            # Convert predictions to trading signals (-1, 0, 1)
            if len(np.unique(y_test)) == 2:
                # Binary classification: 1 for long, -1 for short
                signals = pd.Series(np.where(y_pred == 1, 1, -1), index=X_test.index)
            else:
                # Regression or multiclass: use as is
                signals = pd.Series(y_pred, index=X_test.index)
            
            # Create OHLCV data for backtester (using prices as close)
            backtest_data = pd.DataFrame(index=X_test.index)
            backtest_data['open'] = prices
            backtest_data['high'] = prices * 1.01  # Simulated high
            backtest_data['low'] = prices * 0.99   # Simulated low
            backtest_data['close'] = prices
            backtest_data['volume'] = 1000000  # Dummy volume
            
            # Run backtest
            backtester = RealisticBacktester(self.config.backtest_config)
            backtest_results = backtester.run(backtest_data, signals, prices)
            
            # Update fold result
            fold_result.backtest_results = backtest_results
            fold_result.sharpe_ratio = backtest_results.sharpe_ratio
            fold_result.max_drawdown = backtest_results.max_drawdown
            fold_result.total_return = backtest_results.total_return
            
            logger.info(f"Fold {fold_result.fold_number} backtest: "
                       f"Return={backtest_results.total_return:.2%}, "
                       f"Sharpe={backtest_results.sharpe_ratio:.2f}")
            
        except Exception as e:
            logger.error(f"Backtesting failed for fold {fold_result.fold_number}: {e}")
        
        return fold_result
    
    def _aggregate_results(self, 
                          fold_results: List[FoldResult],
                          model_name: str) -> WalkForwardResults:
        """Aggregate results across all folds"""
        
        # Calculate aggregate metrics
        test_accuracies = [f.test_accuracy for f in fold_results]
        test_f1_scores = [f.test_f1 for f in fold_results]
        
        # Backtesting aggregates
        sharpe_ratios = [f.sharpe_ratio for f in fold_results if f.sharpe_ratio is not None]
        returns = [f.total_return for f in fold_results if f.total_return is not None]
        drawdowns = [f.max_drawdown for f in fold_results if f.max_drawdown is not None]
        
        # Degradation analysis
        degraded_folds = [f.fold_number for f in fold_results if f.is_degraded]
        
        # Calculate stability score (inverse of coefficient of variation)
        if test_accuracies:
            cv = np.std(test_accuracies) / np.mean(test_accuracies) if np.mean(test_accuracies) > 0 else 1
            stability_score = 1 / (1 + cv)
        else:
            stability_score = 0
        
        results = WalkForwardResults(
            model_name=model_name,
            n_folds=len(fold_results),
            config=self.config,
            fold_results=fold_results,
            mean_test_accuracy=np.mean(test_accuracies),
            std_test_accuracy=np.std(test_accuracies),
            mean_test_f1=np.mean(test_f1_scores),
            std_test_f1=np.std(test_f1_scores),
            mean_sharpe=np.mean(sharpe_ratios) if sharpe_ratios else None,
            mean_return=np.mean(returns) if returns else None,
            mean_drawdown=np.mean(drawdowns) if drawdowns else None,
            degradation_detected=len(degraded_folds) > 0,
            degradation_folds=degraded_folds,
            stability_score=stability_score
        )
        
        return results
    
    def _analyze_feature_stability(self, fold_results: List[FoldResult]) -> Dict[str, float]:
        """Analyze feature importance stability across folds"""
        
        # Collect all features
        all_features = set()
        for fold in fold_results:
            if fold.feature_importance:
                all_features.update(fold.feature_importance.keys())
        
        # Calculate stability for each feature
        feature_stability = {}
        
        for feature in all_features:
            importances = []
            for fold in fold_results:
                if fold.feature_importance and feature in fold.feature_importance:
                    importances.append(fold.feature_importance[feature])
                else:
                    importances.append(0)
            
            if importances:
                # Stability = 1 - coefficient of variation
                mean_imp = np.mean(importances)
                if mean_imp > 0:
                    cv = np.std(importances) / mean_imp
                    feature_stability[feature] = 1 / (1 + cv)
                else:
                    feature_stability[feature] = 0
        
        return feature_stability
    
    def _save_fold_model(self, model: BaseEstimator, fold_number: int):
        """Save model for a specific fold"""
        import joblib
        model_path = Path(self.config.model_save_path) / f"fold_{fold_number}_model.joblib"
        joblib.dump(model, model_path)
        logger.debug(f"Saved model for fold {fold_number} to {model_path}")
    
    def _save_results(self, results: WalkForwardResults):
        """Save validation results"""
        results_path = Path(self.config.model_save_path) / f"{results.model_name}_results.json"
        
        # Convert to serializable format
        results_dict = {
            'model_name': results.model_name,
            'n_folds': results.n_folds,
            'mean_test_accuracy': results.mean_test_accuracy,
            'std_test_accuracy': results.std_test_accuracy,
            'mean_test_f1': results.mean_test_f1,
            'std_test_f1': results.std_test_f1,
            'mean_sharpe': results.mean_sharpe,
            'mean_return': results.mean_return,
            'mean_drawdown': results.mean_drawdown,
            'degradation_detected': results.degradation_detected,
            'degradation_folds': results.degradation_folds,
            'stability_score': results.stability_score,
            'fold_results': [f.to_dict() for f in results.fold_results]
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Saved results to {results_path}")
    
    def plot_results(self, results: WalkForwardResults):
        """Plot walk-forward validation results"""
        import matplotlib.pyplot as plt
        
        df = results.to_dataframe()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Accuracy over time
        ax1 = axes[0, 0]
        ax1.plot(df['fold'], df['train_accuracy'], 'b-', label='Train', marker='o')
        ax1.plot(df['fold'], df['test_accuracy'], 'r-', label='Test', marker='s')
        ax1.axhline(y=results.mean_test_accuracy, color='r', linestyle='--', alpha=0.5)
        ax1.fill_between(df['fold'], 
                         results.mean_test_accuracy - results.std_test_accuracy,
                         results.mean_test_accuracy + results.std_test_accuracy,
                         alpha=0.2, color='r')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Across Folds')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: F1 Score over time
        ax2 = axes[0, 1]
        ax2.plot(df['fold'], df['train_f1'], 'b-', label='Train', marker='o')
        ax2.plot(df['fold'], df['test_f1'], 'r-', label='Test', marker='s')
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score Across Folds')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Backtesting Performance
        if 'sharpe_ratio' in df.columns and df['sharpe_ratio'].notna().any():
            ax3 = axes[1, 0]
            ax3.bar(df['fold'], df['sharpe_ratio'].fillna(0))
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            if results.mean_sharpe:
                ax3.axhline(y=results.mean_sharpe, color='g', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Fold')
            ax3.set_ylabel('Sharpe Ratio')
            ax3.set_title('Sharpe Ratio by Fold')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Returns Distribution
        if 'total_return' in df.columns and df['total_return'].notna().any():
            ax4 = axes[1, 1]
            returns = df['total_return'].dropna() * 100  # Convert to percentage
            ax4.hist(returns, bins=min(len(returns), 20), edgecolor='black')
            ax4.axvline(x=returns.mean(), color='r', linestyle='--', label=f'Mean: {returns.mean():.1f}%')
            ax4.set_xlabel('Return (%)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Distribution of Fold Returns')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Mark degraded folds
        if results.degradation_folds:
            for fold in results.degradation_folds:
                for ax in axes.flat:
                    ax.axvspan(fold - 0.5, fold + 0.5, alpha=0.2, color='red')
        
        plt.suptitle(f'Walk-Forward Validation Results: {results.model_name}\n'
                    f'Mean Accuracy: {results.mean_test_accuracy:.3f} ± {results.std_test_accuracy:.3f}',
                    fontsize=14)
        plt.tight_layout()
        
        return fig


def create_walk_forward_validator(config: Optional[WalkForwardConfig] = None) -> WalkForwardValidator:
    """Create walk-forward validator instance"""
    return WalkForwardValidator(config)


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    
    # Get sample data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="5y")  # Need more data for walk-forward
    
    # Create features (simplified)
    data['returns'] = data['Close'].pct_change()
    data['sma_20'] = data['Close'].rolling(20).mean()
    data['rsi'] = 100 - (100 / (1 + data['Close'].diff().clip(lower=0).rolling(14).mean() / 
                                 data['Close'].diff().clip(upper=0).abs().rolling(14).mean()))
    
    # Create feature matrix
    X = pd.DataFrame(index=data.index)
    X['returns'] = data['returns']
    X['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
    X['price_to_sma'] = data['Close'] / data['sma_20']
    X['rsi'] = data['rsi']
    X = X.dropna()
    
    # Create target (next day direction)
    y = (data['Close'].shift(-1) > data['Close']).astype(int)
    y = y.loc[X.index]
    
    # Get prices for backtesting
    prices = data['Close'].loc[X.index]
    
    # Configure walk-forward validation
    config = WalkForwardConfig(
        train_window=252,  # 1 year
        test_window=63,    # 3 months
        step_size=21,      # 1 month
        expanding_window=True,
        backtest_each_fold=True
    )
    
    # Create validator
    validator = create_walk_forward_validator(config)
    
    # Test with different models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }
    
    for model_name, model in models.items():
        print(f"\nValidating {model_name}...")
        results = validator.validate(model, X, y, prices, model_name)
        
        print(f"\nResults for {model_name}:")
        print(f"  Mean Test Accuracy: {results.mean_test_accuracy:.3f} ± {results.std_test_accuracy:.3f}")
        print(f"  Mean Test F1: {results.mean_test_f1:.3f} ± {results.std_test_f1:.3f}")
        if results.mean_sharpe:
            print(f"  Mean Sharpe Ratio: {results.mean_sharpe:.2f}")
        if results.mean_return:
            print(f"  Mean Return: {results.mean_return:.2%}")
        print(f"  Stability Score: {results.stability_score:.3f}")
        print(f"  Degradation Detected: {results.degradation_detected}")
        if results.degradation_folds:
            print(f"  Degraded Folds: {results.degradation_folds}")
        
        # Plot results
        fig = validator.plot_results(results)
        fig.savefig(f'{model_name}_walk_forward.png')
        print(f"  Plot saved to {model_name}_walk_forward.png")