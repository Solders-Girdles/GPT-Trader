"""
Model Calibration Framework
Phase 2.5 - Day 8

Implements probability calibration and threshold optimization for trading models.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score,
    precision_recall_curve, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Configuration for model calibration"""
    # Calibration methods
    method: str = "isotonic"  # "isotonic", "sigmoid", or "ensemble"
    cv_folds: int = 3  # Cross-validation folds for calibration
    
    # Threshold optimization
    optimize_threshold: bool = True
    threshold_metric: str = "f1"  # "f1", "precision", "recall", "profit"
    min_precision: float = 0.6  # Minimum precision constraint
    
    # Confidence intervals
    confidence_level: float = 0.95  # 95% confidence intervals
    bootstrap_samples: int = 1000
    
    # Risk adjustment
    kelly_fraction: float = 0.25  # Kelly criterion fraction for position sizing
    max_position_size: float = 0.1  # Maximum position size as fraction of capital
    
    # Performance targets (realistic for trading)
    target_accuracy: float = 0.60  # 60% accuracy
    target_sharpe: float = 1.0  # Sharpe ratio > 1
    target_win_rate: float = 0.52  # 52% win rate
    max_drawdown: float = 0.20  # Maximum 20% drawdown


@dataclass
class CalibrationMetrics:
    """Metrics for evaluating calibration quality"""
    # Calibration metrics
    brier_score: float
    log_loss: float
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    
    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    
    # Trading metrics
    expected_return: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    
    # Reliability diagram data
    reliability_data: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'brier_score': self.brier_score,
            'log_loss': self.log_loss,
            'ece': self.ece,
            'mce': self.mce,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc,
            'expected_return': self.expected_return,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor
        }


@dataclass
class ThresholdOptimizationResult:
    """Results from threshold optimization"""
    optimal_threshold: float
    metric_value: float
    precision_at_threshold: float
    recall_at_threshold: float
    f1_at_threshold: float
    expected_profit: float
    
    # Threshold analysis
    threshold_curve: pd.DataFrame
    best_thresholds: Dict[str, float]  # Best threshold for each metric
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'optimal_threshold': self.optimal_threshold,
            'metric_value': self.metric_value,
            'precision': self.precision_at_threshold,
            'recall': self.recall_at_threshold,
            'f1': self.f1_at_threshold,
            'expected_profit': self.expected_profit,
            'best_thresholds': self.best_thresholds
        }


class ModelCalibrator:
    """
    Advanced model calibration for trading applications.
    
    Features:
    - Multiple calibration methods (isotonic, sigmoid, ensemble)
    - Threshold optimization for trading metrics
    - Confidence interval estimation
    - Kelly criterion position sizing
    - Realistic performance targeting
    """
    
    def __init__(self, config: Optional[CalibrationConfig] = None):
        """
        Initialize calibrator.
        
        Args:
            config: Calibration configuration
        """
        self.config = config or CalibrationConfig()
        self.calibrated_model = None
        self.optimal_threshold = 0.5
        self.calibration_metrics = None
        self.confidence_intervals = {}
        
        logger.info(f"ModelCalibrator initialized with {self.config.method} method")
    
    def calibrate(self,
                 model: BaseEstimator,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 X_val: pd.DataFrame,
                 y_val: pd.Series) -> BaseEstimator:
        """
        Calibrate model probabilities.
        
        Args:
            model: Trained model to calibrate
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Calibrated model
        """
        logger.info("Starting model calibration...")
        
        # Get base predictions
        if hasattr(model, 'predict_proba'):
            base_probs = model.predict_proba(X_val)[:, 1]
        else:
            # For models without predict_proba
            base_probs = model.decision_function(X_val)
            base_probs = 1 / (1 + np.exp(-base_probs))  # Sigmoid transformation
        
        # Apply calibration based on method
        if self.config.method == "isotonic":
            self.calibrated_model = self._isotonic_calibration(model, X_train, y_train)
        elif self.config.method == "sigmoid":
            self.calibrated_model = self._sigmoid_calibration(model, X_train, y_train)
        elif self.config.method == "ensemble":
            self.calibrated_model = self._ensemble_calibration(model, X_train, y_train)
        else:
            raise ValueError(f"Unknown calibration method: {self.config.method}")
        
        # Get calibrated predictions
        calibrated_probs = self.calibrated_model.predict_proba(X_val)[:, 1]
        
        # Evaluate calibration quality
        self.calibration_metrics = self._evaluate_calibration(
            y_val, base_probs, calibrated_probs
        )
        
        # Optimize threshold if requested
        if self.config.optimize_threshold:
            self.threshold_result = self._optimize_threshold(
                y_val, calibrated_probs
            )
            self.optimal_threshold = self.threshold_result.optimal_threshold
        
        # Estimate confidence intervals
        self.confidence_intervals = self._estimate_confidence_intervals(
            X_val, y_val
        )
        
        logger.info(f"Calibration complete. ECE: {self.calibration_metrics.ece:.4f}, "
                   f"Optimal threshold: {self.optimal_threshold:.3f}")
        
        return self.calibrated_model
    
    def _isotonic_calibration(self,
                             model: BaseEstimator,
                             X: pd.DataFrame,
                             y: pd.Series) -> BaseEstimator:
        """Apply isotonic regression calibration"""
        calibrated = CalibratedClassifierCV(
            model, method='isotonic', cv=self.config.cv_folds
        )
        calibrated.fit(X, y)
        return calibrated
    
    def _sigmoid_calibration(self,
                            model: BaseEstimator,
                            X: pd.DataFrame,
                            y: pd.Series) -> BaseEstimator:
        """Apply sigmoid (Platt) calibration"""
        calibrated = CalibratedClassifierCV(
            model, method='sigmoid', cv=self.config.cv_folds
        )
        calibrated.fit(X, y)
        return calibrated
    
    def _ensemble_calibration(self,
                             model: BaseEstimator,
                             X: pd.DataFrame,
                             y: pd.Series) -> BaseEstimator:
        """Apply ensemble calibration (combines isotonic and sigmoid)"""
        # Create both calibrators
        isotonic_cal = CalibratedClassifierCV(
            clone(model), method='isotonic', cv=self.config.cv_folds
        )
        sigmoid_cal = CalibratedClassifierCV(
            clone(model), method='sigmoid', cv=self.config.cv_folds
        )
        
        isotonic_cal.fit(X, y)
        sigmoid_cal.fit(X, y)
        
        # Create ensemble calibrator
        class EnsembleCalibratedClassifier:
            def __init__(self, isotonic, sigmoid):
                self.isotonic = isotonic
                self.sigmoid = sigmoid
            
            def predict_proba(self, X):
                iso_probs = self.isotonic.predict_proba(X)
                sig_probs = self.sigmoid.predict_proba(X)
                # Average the probabilities
                return (iso_probs + sig_probs) / 2
            
            def predict(self, X):
                probs = self.predict_proba(X)[:, 1]
                return (probs >= 0.5).astype(int)
        
        return EnsembleCalibratedClassifier(isotonic_cal, sigmoid_cal)
    
    def _evaluate_calibration(self,
                             y_true: np.ndarray,
                             base_probs: np.ndarray,
                             calibrated_probs: np.ndarray) -> CalibrationMetrics:
        """Evaluate calibration quality"""
        
        # Calculate calibration metrics
        brier_score = brier_score_loss(y_true, calibrated_probs)
        log_loss_val = log_loss(y_true, calibrated_probs)
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(y_true, calibrated_probs)
        
        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(y_true, calibrated_probs)
        
        # Performance metrics
        y_pred = (calibrated_probs >= self.optimal_threshold).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_true, calibrated_probs)
        except:
            roc_auc = 0.5
        
        # Trading metrics (simplified)
        returns = np.where(y_pred == y_true, 0.01, -0.01)  # 1% win/loss
        expected_return = np.mean(returns)
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        win_rate = np.mean(y_pred == y_true)
        
        # Profit factor
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        profit_factor = abs(np.sum(wins) / (np.sum(losses) + 1e-10))
        
        # Reliability diagram data
        reliability_data = self._calculate_reliability_diagram(y_true, calibrated_probs)
        
        return CalibrationMetrics(
            brier_score=brier_score,
            log_loss=log_loss_val,
            ece=ece,
            mce=mce,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            expected_return=expected_return,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            reliability_data=reliability_data
        )
    
    def _calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_mce(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def _calculate_reliability_diagram(self,
                                      y_true: np.ndarray,
                                      y_prob: np.ndarray,
                                      n_bins: int = 10) -> Dict:
        """Calculate data for reliability diagram"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(y_true[in_bin].mean())
                bin_confidences.append(y_prob[in_bin].mean())
                bin_counts.append(in_bin.sum())
        
        return {
            'bin_centers': bin_centers,
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts
        }
    
    def _optimize_threshold(self,
                           y_true: np.ndarray,
                           y_prob: np.ndarray) -> ThresholdOptimizationResult:
        """Optimize decision threshold for trading"""
        
        # Generate threshold candidates
        thresholds = np.linspace(0.3, 0.7, 41)  # Focus on realistic range
        
        results = []
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Calculate expected profit (simplified)
            # Assume 2% gain on correct predictions, 1% loss on incorrect
            profit = np.where(y_pred == 1,
                            np.where(y_true == 1, 0.02, -0.01),
                            0).sum()
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'profit': profit
            })
        
        results_df = pd.DataFrame(results)
        
        # Find optimal threshold based on metric
        if self.config.threshold_metric == "profit":
            # Filter by minimum precision constraint
            valid_results = results_df[results_df['precision'] >= self.config.min_precision]
            if len(valid_results) > 0:
                optimal_idx = valid_results['profit'].idxmax()
            else:
                optimal_idx = results_df['f1'].idxmax()
        else:
            metric_col = self.config.threshold_metric
            if metric_col in results_df.columns:
                optimal_idx = results_df[metric_col].idxmax()
            else:
                optimal_idx = results_df['f1'].idxmax()
        
        optimal_row = results_df.loc[optimal_idx]
        
        # Find best threshold for each metric
        best_thresholds = {
            'precision': results_df.loc[results_df['precision'].idxmax(), 'threshold'],
            'recall': results_df.loc[results_df['recall'].idxmax(), 'threshold'],
            'f1': results_df.loc[results_df['f1'].idxmax(), 'threshold'],
            'profit': results_df.loc[results_df['profit'].idxmax(), 'threshold']
        }
        
        return ThresholdOptimizationResult(
            optimal_threshold=optimal_row['threshold'],
            metric_value=optimal_row[self.config.threshold_metric],
            precision_at_threshold=optimal_row['precision'],
            recall_at_threshold=optimal_row['recall'],
            f1_at_threshold=optimal_row['f1'],
            expected_profit=optimal_row['profit'],
            threshold_curve=results_df,
            best_thresholds=best_thresholds
        )
    
    def _estimate_confidence_intervals(self,
                                      X: pd.DataFrame,
                                      y: pd.Series) -> Dict[str, Tuple[float, float]]:
        """Estimate confidence intervals using bootstrap"""
        
        n_samples = len(y)
        metrics_bootstrap = []
        
        for _ in range(self.config.bootstrap_samples):
            # Bootstrap sample
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X.iloc[idx]
            y_boot = y.iloc[idx]
            
            # Get predictions
            if self.calibrated_model:
                y_prob = self.calibrated_model.predict_proba(X_boot)[:, 1]
                y_pred = (y_prob >= self.optimal_threshold).astype(int)
            else:
                continue
            
            # Calculate metrics
            metrics_bootstrap.append({
                'accuracy': accuracy_score(y_boot, y_pred),
                'precision': precision_score(y_boot, y_pred, zero_division=0),
                'recall': recall_score(y_boot, y_pred, zero_division=0),
                'f1': f1_score(y_boot, y_pred, zero_division=0)
            })
        
        # Calculate confidence intervals
        confidence_intervals = {}
        alpha = 1 - self.config.confidence_level
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            values = [m[metric] for m in metrics_bootstrap]
            lower = np.percentile(values, alpha/2 * 100)
            upper = np.percentile(values, (1 - alpha/2) * 100)
            confidence_intervals[metric] = (lower, upper)
        
        return confidence_intervals
    
    def calculate_position_size(self,
                               probability: float,
                               current_capital: float = 100000) -> float:
        """
        Calculate position size using Kelly criterion.
        
        Args:
            probability: Calibrated probability of success
            current_capital: Current capital amount
            
        Returns:
            Recommended position size in dollars
        """
        # Kelly criterion: f = p - q/b
        # where p = probability of win, q = probability of loss, b = odds
        
        # Assume 2:1 reward/risk ratio
        win_amount = 0.02  # 2% gain
        loss_amount = 0.01  # 1% loss
        odds = win_amount / loss_amount
        
        # Kelly fraction
        kelly_fraction = probability - (1 - probability) / odds
        
        # Apply safety factor
        safe_kelly = kelly_fraction * self.config.kelly_fraction
        
        # Ensure within bounds
        position_fraction = min(
            max(safe_kelly, 0),
            self.config.max_position_size
        )
        
        return current_capital * position_fraction
    
    def plot_calibration_diagnostic(self,
                                   y_true: np.ndarray,
                                   y_prob_uncalibrated: np.ndarray,
                                   y_prob_calibrated: np.ndarray,
                                   save_path: Optional[str] = None):
        """Plot comprehensive calibration diagnostics"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Reliability Diagram
        ax = axes[0, 0]
        self._plot_reliability_diagram(y_true, y_prob_uncalibrated, y_prob_calibrated, ax)
        ax.set_title('Reliability Diagram')
        
        # 2. Histogram of Predicted Probabilities
        ax = axes[0, 1]
        ax.hist(y_prob_uncalibrated, bins=30, alpha=0.5, label='Uncalibrated', color='blue')
        ax.hist(y_prob_calibrated, bins=30, alpha=0.5, label='Calibrated', color='green')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Predictions')
        ax.legend()
        
        # 3. ROC Curves
        ax = axes[0, 2]
        fpr_uncal, tpr_uncal, _ = roc_curve(y_true, y_prob_uncalibrated)
        fpr_cal, tpr_cal, _ = roc_curve(y_true, y_prob_calibrated)
        
        auc_uncal = roc_auc_score(y_true, y_prob_uncalibrated)
        auc_cal = roc_auc_score(y_true, y_prob_calibrated)
        
        ax.plot(fpr_uncal, tpr_uncal, label=f'Uncalibrated (AUC={auc_uncal:.3f})', color='blue')
        ax.plot(fpr_cal, tpr_cal, label=f'Calibrated (AUC={auc_cal:.3f})', color='green')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        
        # 4. Threshold Optimization Curve
        ax = axes[1, 0]
        if hasattr(self, 'threshold_result'):
            df = self.threshold_result.threshold_curve
            ax.plot(df['threshold'], df['precision'], label='Precision', color='blue')
            ax.plot(df['threshold'], df['recall'], label='Recall', color='green')
            ax.plot(df['threshold'], df['f1'], label='F1', color='red')
            ax.axvline(x=self.optimal_threshold, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Score')
            ax.set_title('Threshold Optimization')
            ax.legend()
        
        # 5. Calibration Metrics
        ax = axes[1, 1]
        ax.axis('off')
        if self.calibration_metrics:
            metrics_text = f"""
Calibration Metrics:
  ECE: {self.calibration_metrics.ece:.4f}
  MCE: {self.calibration_metrics.mce:.4f}
  Brier Score: {self.calibration_metrics.brier_score:.4f}
  Log Loss: {self.calibration_metrics.log_loss:.4f}

Performance:
  Accuracy: {self.calibration_metrics.accuracy:.3f}
  Precision: {self.calibration_metrics.precision:.3f}
  Recall: {self.calibration_metrics.recall:.3f}
  F1 Score: {self.calibration_metrics.f1_score:.3f}

Trading:
  Sharpe Ratio: {self.calibration_metrics.sharpe_ratio:.2f}
  Win Rate: {self.calibration_metrics.win_rate:.3f}
  Profit Factor: {self.calibration_metrics.profit_factor:.2f}
"""
            ax.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center')
        ax.set_title('Metrics Summary')
        
        # 6. Confidence Intervals
        ax = axes[1, 2]
        if self.confidence_intervals:
            metrics = list(self.confidence_intervals.keys())
            centers = [np.mean(self.confidence_intervals[m]) for m in metrics]
            errors = [(centers[i] - self.confidence_intervals[m][0],
                      self.confidence_intervals[m][1] - centers[i])
                     for i, m in enumerate(metrics)]
            
            errors = list(zip(*errors))
            ax.barh(range(len(metrics)), centers, xerr=errors, capsize=5)
            ax.set_yticks(range(len(metrics)))
            ax.set_yticklabels(metrics)
            ax.set_xlabel('Score')
            ax.set_title(f'{self.config.confidence_level*100:.0f}% Confidence Intervals')
        
        plt.suptitle('Model Calibration Diagnostics', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Calibration diagnostic plot saved to {save_path}")
        
        return fig
    
    def _plot_reliability_diagram(self, y_true, y_prob_uncal, y_prob_cal, ax):
        """Plot reliability diagram"""
        
        # Calculate reliability for uncalibrated
        rel_uncal = self._calculate_reliability_diagram(y_true, y_prob_uncal)
        
        # Calculate reliability for calibrated
        rel_cal = self._calculate_reliability_diagram(y_true, y_prob_cal)
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        
        # Plot uncalibrated
        if rel_uncal['bin_centers']:
            ax.plot(rel_uncal['bin_confidences'], rel_uncal['bin_accuracies'],
                   'o-', color='blue', label='Uncalibrated', markersize=8)
        
        # Plot calibrated
        if rel_cal['bin_centers']:
            ax.plot(rel_cal['bin_confidences'], rel_cal['bin_accuracies'],
                   's-', color='green', label='Calibrated', markersize=8)
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def save_calibration_state(self, filepath: str):
        """Save calibration state"""
        state = {
            'config': {
                'method': self.config.method,
                'cv_folds': self.config.cv_folds,
                'threshold_metric': self.config.threshold_metric,
                'min_precision': self.config.min_precision
            },
            'optimal_threshold': self.optimal_threshold,
            'calibration_metrics': self.calibration_metrics.to_dict() if self.calibration_metrics else None,
            'confidence_intervals': self.confidence_intervals,
            'threshold_result': self.threshold_result.to_dict() if hasattr(self, 'threshold_result') else None,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Calibration state saved to {filepath}")


def create_model_calibrator(config: Optional[CalibrationConfig] = None) -> ModelCalibrator:
    """Create model calibrator instance"""
    return ModelCalibrator(config)


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Get sample data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="2y")
    
    # Create features
    data['returns'] = data['Close'].pct_change()
    data['sma_20'] = data['Close'].rolling(20).mean()
    
    X = pd.DataFrame(index=data.index)
    X['returns'] = data['returns']
    X['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
    X['price_to_sma'] = data['Close'] / data['sma_20']
    X = X.dropna()
    
    # Create target
    y = (data['Close'].shift(-1) > data['Close']).astype(int)
    y = y.loc[X.index]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Get uncalibrated predictions
    y_prob_uncal = model.predict_proba(X_test)[:, 1]
    
    # Create calibrator with realistic config
    config = CalibrationConfig(
        method="isotonic",
        optimize_threshold=True,
        threshold_metric="f1",
        min_precision=0.6,
        target_accuracy=0.60,
        target_sharpe=1.0
    )
    
    calibrator = create_model_calibrator(config)
    
    # Calibrate model
    calibrated_model = calibrator.calibrate(
        model, X_train, y_train, X_test, y_test
    )
    
    # Get calibrated predictions
    y_prob_cal = calibrated_model.predict_proba(X_test)[:, 1]
    
    # Display results
    print("\nCalibration Results:")
    print(f"  Method: {config.method}")
    print(f"  ECE: {calibrator.calibration_metrics.ece:.4f}")
    print(f"  Optimal Threshold: {calibrator.optimal_threshold:.3f}")
    print(f"  Accuracy: {calibrator.calibration_metrics.accuracy:.3f}")
    print(f"  Sharpe Ratio: {calibrator.calibration_metrics.sharpe_ratio:.2f}")
    
    # Position sizing example
    sample_prob = 0.65
    position = calibrator.calculate_position_size(sample_prob)
    print(f"\nPosition Sizing (p={sample_prob}):")
    print(f"  Recommended Position: ${position:,.2f}")
    
    # Plot diagnostics
    fig = calibrator.plot_calibration_diagnostic(
        y_test.values, y_prob_uncal, y_prob_cal
    )
    plt.show()