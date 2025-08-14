"""
Decision Threshold Optimizer for Trading Models
Phase 2.5 - Day 8

Optimizes decision thresholds for different trading objectives and constraints.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import minimize_scalar, differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, precision_recall_curve, roc_curve
)

logger = logging.getLogger(__name__)


@dataclass
class TradingConstraints:
    """Constraints for trading decisions"""
    min_precision: float = 0.6      # Minimum precision (avoid false signals)
    min_recall: float = 0.3          # Minimum recall (can miss some opportunities)
    max_positions: int = 10          # Maximum concurrent positions
    min_confidence: float = 0.55     # Minimum confidence for entry
    max_exposure: float = 0.3        # Maximum portfolio exposure
    
    # Risk constraints
    max_loss_per_trade: float = 0.02     # 2% max loss per trade
    max_daily_loss: float = 0.05         # 5% max daily loss
    max_consecutive_losses: int = 5       # Circuit breaker
    
    # Cost constraints
    commission_rate: float = 0.001       # 0.1% commission
    slippage_rate: float = 0.0005        # 0.05% slippage
    min_profit_threshold: float = 0.003  # 0.3% minimum profit to cover costs


@dataclass
class OptimizationObjective:
    """Optimization objective configuration"""
    primary_metric: str = "profit"  # "profit", "sharpe", "f1", "precision", "recall"
    secondary_metric: Optional[str] = "risk"
    
    # Weights for multi-objective optimization
    profit_weight: float = 0.4
    risk_weight: float = 0.3
    accuracy_weight: float = 0.2
    stability_weight: float = 0.1
    
    # Risk preferences
    risk_aversion: float = 2.0  # Higher = more risk averse
    
    def calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite optimization score"""
        score = 0
        
        # Normalize metrics to 0-1 scale
        profit = min(max(metrics.get('profit', 0) / 0.3, 0), 1)  # Normalize to 30% annual
        risk_adj = 1 - min(abs(metrics.get('max_drawdown', 0)) / 0.3, 1)  # Normalize drawdown
        accuracy = min(metrics.get('accuracy', 0.5) / 0.7, 1)  # Normalize to 70% max
        stability = metrics.get('stability', 0.5)
        
        # Apply weights
        score += self.profit_weight * profit
        score += self.risk_weight * risk_adj
        score += self.accuracy_weight * accuracy
        score += self.stability_weight * stability
        
        # Apply risk aversion penalty
        risk_penalty = self.risk_aversion * abs(metrics.get('max_drawdown', 0))
        score -= risk_penalty
        
        return score


@dataclass
class ThresholdOptimizationResult:
    """Results from threshold optimization"""
    # Optimal thresholds
    entry_threshold: float
    exit_threshold: float
    stop_loss_threshold: float
    take_profit_threshold: float
    
    # Performance at optimal thresholds
    expected_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    
    # Trade statistics
    avg_trades_per_period: float
    avg_holding_period: float
    precision: float
    recall: float
    
    # Optimization details
    objective_value: float
    convergence_achieved: bool
    iterations: int
    
    # Threshold curves
    threshold_performance: pd.DataFrame
    sensitivity_analysis: Dict[str, pd.DataFrame]


class ThresholdOptimizer:
    """
    Optimizes decision thresholds for trading models.
    
    Features:
    - Multi-objective optimization
    - Constraint satisfaction
    - Cost-aware optimization
    - Sensitivity analysis
    - Dynamic threshold adjustment
    """
    
    def __init__(self, 
                 constraints: Optional[TradingConstraints] = None,
                 objective: Optional[OptimizationObjective] = None):
        """
        Initialize optimizer.
        
        Args:
            constraints: Trading constraints
            objective: Optimization objective
        """
        self.constraints = constraints or TradingConstraints()
        self.objective = objective or OptimizationObjective()
        
        logger.info(f"ThresholdOptimizer initialized with {self.objective.primary_metric} objective")
    
    def optimize(self,
                y_true: np.ndarray,
                y_prob: np.ndarray,
                prices: Optional[pd.Series] = None,
                features: Optional[pd.DataFrame] = None) -> ThresholdOptimizationResult:
        """
        Optimize trading thresholds.
        
        Args:
            y_true: True labels (1 for profitable, 0 for not)
            y_prob: Predicted probabilities
            prices: Price series for backtesting
            features: Additional features for dynamic thresholds
            
        Returns:
            Optimization results with optimal thresholds
        """
        logger.info("Starting threshold optimization...")
        
        # Single threshold optimization
        entry_threshold = self._optimize_entry_threshold(y_true, y_prob)
        
        # Exit threshold optimization (if we have price data)
        if prices is not None:
            exit_threshold = self._optimize_exit_threshold(
                y_true, y_prob, prices, entry_threshold
            )
            stop_loss, take_profit = self._optimize_risk_thresholds(
                y_true, y_prob, prices, entry_threshold
            )
        else:
            exit_threshold = entry_threshold * 0.9  # Default: lower threshold for exit
            stop_loss = -self.constraints.max_loss_per_trade
            take_profit = self.constraints.max_loss_per_trade * 3  # 3:1 R:R ratio
        
        # Calculate performance at optimal thresholds
        performance = self._calculate_performance(
            y_true, y_prob, entry_threshold, prices
        )
        
        # Sensitivity analysis
        sensitivity = self._sensitivity_analysis(
            y_true, y_prob, entry_threshold, prices
        )
        
        # Generate threshold performance curve
        threshold_curve = self._generate_threshold_curve(y_true, y_prob, prices)
        
        return ThresholdOptimizationResult(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            stop_loss_threshold=stop_loss,
            take_profit_threshold=take_profit,
            expected_return=performance['expected_return'],
            sharpe_ratio=performance['sharpe_ratio'],
            max_drawdown=performance['max_drawdown'],
            win_rate=performance['win_rate'],
            avg_trades_per_period=performance['avg_trades'],
            avg_holding_period=performance.get('avg_holding', 1),
            precision=performance['precision'],
            recall=performance['recall'],
            objective_value=performance['objective_value'],
            convergence_achieved=True,
            iterations=performance.get('iterations', 100),
            threshold_performance=threshold_curve,
            sensitivity_analysis=sensitivity
        )
    
    def _optimize_entry_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Optimize entry threshold"""
        
        def objective(threshold):
            y_pred = (y_prob >= threshold).astype(int)
            
            # Check constraints
            if sum(y_pred) == 0:
                return 1e10  # No trades = bad
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            # Constraint violations
            if precision < self.constraints.min_precision:
                return 1e10
            if recall < self.constraints.min_recall:
                return 1e10
            
            # Calculate objective based on primary metric
            if self.objective.primary_metric == "profit":
                # Simple profit calculation
                profit = self._calculate_simple_profit(y_true, y_pred)
                return -profit  # Minimize negative profit
            elif self.objective.primary_metric == "f1":
                f1 = f1_score(y_true, y_pred, zero_division=0)
                return -f1
            elif self.objective.primary_metric == "precision":
                return -precision
            elif self.objective.primary_metric == "sharpe":
                sharpe = self._calculate_simple_sharpe(y_true, y_pred)
                return -sharpe
            else:
                # Composite score
                metrics = {
                    'profit': self._calculate_simple_profit(y_true, y_pred),
                    'accuracy': accuracy_score(y_true, y_pred),
                    'max_drawdown': -0.1,  # Placeholder
                    'stability': 0.5  # Placeholder
                }
                return -self.objective.calculate_composite_score(metrics)
        
        # Optimize using bounded search
        result = minimize_scalar(
            objective,
            bounds=(self.constraints.min_confidence, 0.9),
            method='bounded'
        )
        
        optimal_threshold = result.x
        logger.info(f"Optimal entry threshold: {optimal_threshold:.3f}")
        
        return optimal_threshold
    
    def _optimize_exit_threshold(self, y_true: np.ndarray, y_prob: np.ndarray,
                                prices: pd.Series, entry_threshold: float) -> float:
        """Optimize exit threshold given entry threshold"""
        
        def objective(exit_threshold):
            if exit_threshold >= entry_threshold:
                return 1e10  # Exit should be lower than entry
            
            # Simulate trading with entry and exit thresholds
            returns = self._simulate_threshold_trading(
                y_prob, prices, entry_threshold, exit_threshold
            )
            
            if len(returns) == 0:
                return 1e10
            
            # Calculate Sharpe ratio
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
            return -sharpe
        
        # Optimize
        result = minimize_scalar(
            objective,
            bounds=(0.3, entry_threshold - 0.05),
            method='bounded'
        )
        
        return result.x
    
    def _optimize_risk_thresholds(self, y_true: np.ndarray, y_prob: np.ndarray,
                                 prices: pd.Series, entry_threshold: float) -> Tuple[float, float]:
        """Optimize stop loss and take profit thresholds"""
        
        def objective(params):
            stop_loss, take_profit = params
            
            # Constraints
            if stop_loss > -0.005:  # Minimum 0.5% stop loss
                return 1e10
            if take_profit < 0.005:  # Minimum 0.5% take profit
                return 1e10
            if take_profit / abs(stop_loss) < 1.5:  # Minimum 1.5:1 R:R ratio
                return 1e10
            
            # Simulate trading with risk management
            returns = self._simulate_risk_managed_trading(
                y_prob, prices, entry_threshold, stop_loss, take_profit
            )
            
            if len(returns) == 0:
                return 1e10
            
            # Optimize for risk-adjusted return
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
            max_dd = self._calculate_max_drawdown(returns)
            
            # Composite objective
            score = sharpe - self.objective.risk_aversion * abs(max_dd)
            return -score
        
        # Optimize using differential evolution for global optimization
        bounds = [
            (-self.constraints.max_loss_per_trade, -0.005),  # Stop loss
            (0.005, self.constraints.max_loss_per_trade * 4)  # Take profit
        ]
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=50,
            popsize=15,
            seed=42
        )
        
        stop_loss, take_profit = result.x
        logger.info(f"Optimal stop loss: {stop_loss:.3%}, take profit: {take_profit:.3%}")
        
        return stop_loss, take_profit
    
    def _calculate_performance(self, y_true: np.ndarray, y_prob: np.ndarray,
                              threshold: float, prices: Optional[pd.Series]) -> Dict[str, float]:
        """Calculate performance metrics at given threshold"""
        y_pred = (y_prob >= threshold).astype(int)
        
        # Basic metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Trading metrics
        if prices is not None:
            returns = self._simulate_threshold_trading(
                y_prob, prices, threshold, threshold * 0.9
            )
            
            if len(returns) > 0:
                expected_return = np.mean(returns) * 252  # Annualized
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
                max_drawdown = self._calculate_max_drawdown(returns)
                win_rate = np.mean(np.array(returns) > 0)
            else:
                expected_return = 0
                sharpe_ratio = 0
                max_drawdown = -1
                win_rate = 0
        else:
            # Estimate from classification metrics
            expected_return = (precision - 0.5) * 0.2  # Rough estimate
            sharpe_ratio = (precision - 0.5) * 2  # Rough estimate
            max_drawdown = -0.2
            win_rate = precision
        
        # Calculate objective value
        metrics = {
            'profit': expected_return,
            'accuracy': accuracy,
            'max_drawdown': max_drawdown,
            'stability': 0.5  # Placeholder
        }
        objective_value = self.objective.calculate_composite_score(metrics)
        
        return {
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'expected_return': expected_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_trades': np.sum(y_pred),
            'objective_value': objective_value
        }
    
    def _simulate_threshold_trading(self, y_prob: np.ndarray, prices: pd.Series,
                                   entry_threshold: float, exit_threshold: float) -> List[float]:
        """Simulate trading with entry/exit thresholds"""
        returns = []
        in_position = False
        entry_price = 0
        
        for i in range(len(y_prob)):
            if not in_position and y_prob[i] >= entry_threshold:
                # Enter position
                in_position = True
                entry_price = prices.iloc[i] if isinstance(prices, pd.Series) else prices[i]
            elif in_position and y_prob[i] < exit_threshold:
                # Exit position
                exit_price = prices.iloc[i] if isinstance(prices, pd.Series) else prices[i]
                ret = (exit_price - entry_price) / entry_price - self.constraints.commission_rate * 2
                returns.append(ret)
                in_position = False
        
        # Close any open position
        if in_position and len(prices) > 0:
            exit_price = prices.iloc[-1] if isinstance(prices, pd.Series) else prices[-1]
            ret = (exit_price - entry_price) / entry_price - self.constraints.commission_rate * 2
            returns.append(ret)
        
        return returns
    
    def _simulate_risk_managed_trading(self, y_prob: np.ndarray, prices: pd.Series,
                                      entry_threshold: float, stop_loss: float,
                                      take_profit: float) -> List[float]:
        """Simulate trading with risk management"""
        returns = []
        in_position = False
        entry_price = 0
        
        for i in range(len(y_prob)):
            if not in_position and y_prob[i] >= entry_threshold:
                # Enter position
                in_position = True
                entry_price = prices.iloc[i] if isinstance(prices, pd.Series) else prices[i]
            elif in_position:
                current_price = prices.iloc[i] if isinstance(prices, pd.Series) else prices[i]
                current_return = (current_price - entry_price) / entry_price
                
                # Check stop loss
                if current_return <= stop_loss:
                    returns.append(stop_loss - self.constraints.commission_rate * 2)
                    in_position = False
                # Check take profit
                elif current_return >= take_profit:
                    returns.append(take_profit - self.constraints.commission_rate * 2)
                    in_position = False
        
        return returns
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        if len(returns) == 0:
            return 0
        
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_simple_profit(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate simple profit metric"""
        # True positives gain, false positives lose
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        # Simple profit calculation
        profit = tp * 0.02 - fp * 0.01  # 2% gain on correct, 1% loss on incorrect
        
        # Normalize by number of trades
        n_trades = np.sum(y_pred)
        if n_trades > 0:
            profit = profit / n_trades
        
        return profit
    
    def _calculate_simple_sharpe(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate simple Sharpe ratio approximation"""
        if np.sum(y_pred) == 0:
            return 0
        
        # Calculate returns for each prediction
        returns = np.where(
            y_pred == 1,
            np.where(y_true == 1, 0.02, -0.01),
            0
        )
        
        returns = returns[returns != 0]  # Only count trades
        
        if len(returns) == 0:
            return 0
        
        return np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
    
    def _sensitivity_analysis(self, y_true: np.ndarray, y_prob: np.ndarray,
                            optimal_threshold: float, prices: Optional[pd.Series]) -> Dict[str, pd.DataFrame]:
        """Perform sensitivity analysis around optimal threshold"""
        
        # Test range around optimal
        test_range = np.linspace(
            max(0.3, optimal_threshold - 0.1),
            min(0.9, optimal_threshold + 0.1),
            21
        )
        
        results = []
        for threshold in test_range:
            perf = self._calculate_performance(y_true, y_prob, threshold, prices)
            perf['threshold'] = threshold
            results.append(perf)
        
        sensitivity_df = pd.DataFrame(results)
        
        # Calculate gradients
        gradients = {}
        for col in ['precision', 'recall', 'sharpe_ratio', 'expected_return']:
            if col in sensitivity_df.columns:
                gradients[col] = np.gradient(sensitivity_df[col].values)
        
        return {
            'performance': sensitivity_df,
            'gradients': pd.DataFrame(gradients, index=sensitivity_df.index)
        }
    
    def _generate_threshold_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                                 prices: Optional[pd.Series]) -> pd.DataFrame:
        """Generate complete threshold performance curve"""
        thresholds = np.linspace(0.3, 0.9, 31)
        
        results = []
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            # Skip if no trades
            if np.sum(y_pred) == 0:
                continue
            
            perf = self._calculate_performance(y_true, y_prob, threshold, prices)
            perf['threshold'] = threshold
            perf['n_trades'] = np.sum(y_pred)
            
            results.append(perf)
        
        return pd.DataFrame(results)
    
    def plot_optimization_results(self, result: ThresholdOptimizationResult,
                                 y_true: np.ndarray, y_prob: np.ndarray):
        """Plot comprehensive optimization results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Threshold performance curve
        ax = axes[0, 0]
        df = result.threshold_performance
        ax.plot(df['threshold'], df['precision'], label='Precision', color='blue')
        ax.plot(df['threshold'], df['recall'], label='Recall', color='green')
        ax.plot(df['threshold'], df['sharpe_ratio'] / 3, label='Sharpe/3', color='red')
        ax.axvline(x=result.entry_threshold, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Threshold Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Precision-Recall curve
        ax = axes[0, 1]
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        ax.plot(recall, precision, color='blue')
        
        # Mark optimal point
        y_pred_opt = (y_prob >= result.entry_threshold).astype(int)
        prec_opt = precision_score(y_true, y_pred_opt, zero_division=0)
        rec_opt = recall_score(y_true, y_pred_opt, zero_division=0)
        ax.plot(rec_opt, prec_opt, 'ro', markersize=10, label=f'Optimal ({result.entry_threshold:.2f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Expected return vs risk
        ax = axes[0, 2]
        if 'expected_return' in df.columns and 'max_drawdown' in df.columns:
            ax.scatter(df['max_drawdown'], df['expected_return'], c=df['threshold'], cmap='viridis')
            ax.scatter(result.max_drawdown, result.expected_return, 
                      color='red', s=100, marker='*', label='Optimal')
            ax.set_xlabel('Max Drawdown')
            ax.set_ylabel('Expected Return')
            ax.set_title('Risk-Return Trade-off')
            ax.legend()
            plt.colorbar(ax.collections[0], ax=ax, label='Threshold')
        
        # 4. Sensitivity analysis
        ax = axes[1, 0]
        if 'performance' in result.sensitivity_analysis:
            sens_df = result.sensitivity_analysis['performance']
            ax.plot(sens_df['threshold'], sens_df['sharpe_ratio'], label='Sharpe', color='blue')
            ax.fill_between(sens_df['threshold'],
                           sens_df['sharpe_ratio'] - sens_df['sharpe_ratio'].std(),
                           sens_df['sharpe_ratio'] + sens_df['sharpe_ratio'].std(),
                           alpha=0.3, color='blue')
            ax.axvline(x=result.entry_threshold, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Sharpe Ratio')
            ax.set_title('Sensitivity Analysis')
            ax.grid(True, alpha=0.3)
        
        # 5. Confusion matrix at optimal threshold
        ax = axes[1, 1]
        y_pred_opt = (y_prob >= result.entry_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_opt)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix (t={result.entry_threshold:.2f})')
        
        # 6. Summary statistics
        ax = axes[1, 2]
        ax.axis('off')
        summary_text = f"""
Optimal Thresholds:
  Entry: {result.entry_threshold:.3f}
  Exit: {result.exit_threshold:.3f}
  Stop Loss: {result.stop_loss_threshold:.3%}
  Take Profit: {result.take_profit_threshold:.3%}

Performance:
  Expected Return: {result.expected_return:.1%}
  Sharpe Ratio: {result.sharpe_ratio:.2f}
  Max Drawdown: {result.max_drawdown:.1%}
  Win Rate: {result.win_rate:.1%}

Trade Statistics:
  Precision: {result.precision:.3f}
  Recall: {result.recall:.3f}
  Avg Trades: {result.avg_trades_per_period:.1f}
"""
        ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
        ax.set_title('Optimization Summary')
        
        plt.suptitle('Threshold Optimization Results', fontsize=14)
        plt.tight_layout()
        
        return fig


def create_threshold_optimizer(constraints: Optional[TradingConstraints] = None,
                              objective: Optional[OptimizationObjective] = None) -> ThresholdOptimizer:
    """Create threshold optimizer instance"""
    return ThresholdOptimizer(constraints, objective)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    y_true = np.random.randint(0, 2, n_samples)
    
    # Generate probabilities with some signal
    y_prob = np.where(y_true == 1,
                     np.random.beta(7, 3, n_samples),  # Higher prob for positive
                     np.random.beta(3, 7, n_samples))  # Lower prob for negative
    
    # Generate fake price data
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(n_samples) * 0.01)))
    
    # Set up optimization
    constraints = TradingConstraints(
        min_precision=0.6,
        min_recall=0.3,
        commission_rate=0.001
    )
    
    objective = OptimizationObjective(
        primary_metric="sharpe",
        risk_aversion=2.0
    )
    
    # Create optimizer
    optimizer = create_threshold_optimizer(constraints, objective)
    
    # Optimize thresholds
    result = optimizer.optimize(y_true, y_prob, prices)
    
    # Display results
    print("\nOptimization Results:")
    print(f"  Entry Threshold: {result.entry_threshold:.3f}")
    print(f"  Exit Threshold: {result.exit_threshold:.3f}")
    print(f"  Stop Loss: {result.stop_loss_threshold:.3%}")
    print(f"  Take Profit: {result.take_profit_threshold:.3%}")
    print(f"\nExpected Performance:")
    print(f"  Annual Return: {result.expected_return:.1%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown:.1%}")
    print(f"  Win Rate: {result.win_rate:.1%}")
    print(f"  Precision: {result.precision:.3f}")
    print(f"  Recall: {result.recall:.3f}")
    
    # Plot results
    fig = optimizer.plot_optimization_results(result, y_true, y_prob)
    plt.show()