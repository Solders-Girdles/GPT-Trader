"""
Confidence Threshold Optimization Tool
Analyzes historical performance to find optimal confidence thresholds
for minimizing overtrading while maximizing returns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")


@dataclass
class ThresholdAnalysisResult:
    """Results from threshold optimization analysis"""
    
    optimal_threshold: float
    expected_trades_per_year: float
    expected_win_rate: float
    expected_sharpe: float
    expected_return: float
    
    # Trade-off analysis
    threshold_curve: pd.DataFrame
    pareto_efficient_points: List[Tuple[float, float, float]]  # (threshold, trades/year, sharpe)
    
    # Sensitivity analysis
    sensitivity_metrics: Dict[str, float]
    robustness_score: float


class ConfidenceThresholdOptimizer:
    """
    Optimize confidence thresholds for trading frequency and performance
    """
    
    def __init__(self, transaction_cost: float = 0.002, target_trades_per_year: int = 40):
        """
        Initialize optimizer
        
        Args:
            transaction_cost: Transaction cost per trade
            target_trades_per_year: Target number of trades per year
        """
        self.transaction_cost = transaction_cost
        self.target_trades_per_year = target_trades_per_year
        
        # Analysis parameters
        self.min_threshold = 0.3
        self.max_threshold = 0.95
        self.threshold_steps = 66  # Fine-grained analysis
        
    def analyze_threshold_impact(
        self,
        confidence_scores: np.ndarray,
        signals: np.ndarray,
        returns: np.ndarray,
        market_data: Optional[pd.DataFrame] = None
    ) -> ThresholdAnalysisResult:
        """
        Comprehensive analysis of threshold impact on trading performance
        
        Args:
            confidence_scores: Historical confidence scores [0, 1]
            signals: Trading signals (-1, 0, 1)
            returns: Actual returns when trades executed
            market_data: Market data for regime analysis
            
        Returns:
            Comprehensive threshold analysis results
        """
        print("🔍 Starting comprehensive threshold analysis...")
        
        # Generate threshold candidates
        thresholds = np.linspace(self.min_threshold, self.max_threshold, self.threshold_steps)
        
        # Analyze each threshold
        results = []
        for threshold in thresholds:
            result = self._analyze_single_threshold(
                threshold, confidence_scores, signals, returns
            )
            results.append(result)
            
        results_df = pd.DataFrame(results)
        
        # Find optimal threshold using multiple criteria
        optimal_threshold = self._find_optimal_threshold(results_df)
        
        # Pareto efficiency analysis
        pareto_points = self._find_pareto_efficient_points(results_df)
        
        # Sensitivity analysis
        sensitivity = self._analyze_sensitivity(results_df, optimal_threshold)
        
        # Robustness testing
        robustness = self._test_robustness(
            optimal_threshold, confidence_scores, signals, returns
        )
        
        # Get optimal results
        optimal_row = results_df[results_df['threshold'] == optimal_threshold].iloc[0]
        
        result = ThresholdAnalysisResult(
            optimal_threshold=optimal_threshold,
            expected_trades_per_year=optimal_row['trades_per_year'],
            expected_win_rate=optimal_row['win_rate'],
            expected_sharpe=optimal_row['sharpe_ratio'],
            expected_return=optimal_row['annual_return'],
            threshold_curve=results_df,
            pareto_efficient_points=pareto_points,
            sensitivity_metrics=sensitivity,
            robustness_score=robustness
        )
        
        print(f"✅ Analysis complete. Optimal threshold: {optimal_threshold:.3f}")
        return result
    
    def _analyze_single_threshold(
        self,
        threshold: float,
        confidence_scores: np.ndarray,
        signals: np.ndarray,
        returns: np.ndarray
    ) -> Dict:
        """Analyze performance at a single threshold"""
        
        # Apply threshold filter
        trade_mask = confidence_scores >= threshold
        filtered_signals = signals.copy()
        filtered_signals[~trade_mask] = 0
        
        # Calculate trading metrics
        total_trades = np.sum(filtered_signals != 0)
        if total_trades == 0:
            return {
                'threshold': threshold,
                'trades_count': 0,
                'trades_per_year': 0,
                'win_rate': 0,
                'avg_return_per_trade': 0,
                'annual_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'total_fees': 0,
                'net_return': 0,
                'confidence_coverage': 0,
                'signal_retention': 0
            }
        
        # Get returns for executed trades
        trade_returns = returns[filtered_signals != 0]
        
        # Basic metrics
        win_rate = np.mean(trade_returns > 0)
        avg_return = np.mean(trade_returns)
        
        # Annualized metrics (assuming daily data)
        days_span = len(signals)
        trades_per_year = total_trades / days_span * 252
        annual_return = avg_return * trades_per_year
        
        # Risk metrics
        if len(trade_returns) > 1:
            sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252)
            
            # Calculate drawdown
            cumulative_returns = np.cumprod(1 + trade_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (running_max - cumulative_returns) / running_max
            max_drawdown = np.max(drawdowns)
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Profit factor
        winning_trades = trade_returns[trade_returns > 0]
        losing_trades = trade_returns[trade_returns < 0]
        
        if len(losing_trades) > 0:
            profit_factor = np.sum(winning_trades) / abs(np.sum(losing_trades))
        else:
            profit_factor = np.inf if len(winning_trades) > 0 else 0
        
        # Transaction costs
        total_fees = total_trades * self.transaction_cost
        net_annual_return = annual_return - (trades_per_year * self.transaction_cost)
        
        # Coverage metrics
        confidence_coverage = np.mean(confidence_scores >= threshold)
        original_signals = np.sum(signals != 0)
        signal_retention = total_trades / original_signals if original_signals > 0 else 0
        
        return {
            'threshold': threshold,
            'trades_count': total_trades,
            'trades_per_year': trades_per_year,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'total_fees': total_fees,
            'net_return': net_annual_return,
            'confidence_coverage': confidence_coverage,
            'signal_retention': signal_retention
        }
    
    def _find_optimal_threshold(self, results_df: pd.DataFrame) -> float:
        """Find optimal threshold using multi-criteria optimization"""
        
        # Filter out thresholds with too few or too many trades
        min_trades = max(10, self.target_trades_per_year * 0.5)
        max_trades = self.target_trades_per_year * 2.0
        
        valid_results = results_df[
            (results_df['trades_per_year'] >= min_trades) &
            (results_df['trades_per_year'] <= max_trades)
        ].copy()
        
        if len(valid_results) == 0:
            # Fallback to best Sharpe ratio
            return results_df.loc[results_df['sharpe_ratio'].idxmax(), 'threshold']
        
        # Multi-criteria scoring
        # Normalize metrics to [0, 1] for scoring
        valid_results['sharpe_score'] = self._normalize_metric(valid_results['sharpe_ratio'])
        valid_results['return_score'] = self._normalize_metric(valid_results['net_return'])
        valid_results['winrate_score'] = valid_results['win_rate']  # Already [0, 1]
        
        # Trade frequency score (closer to target = higher score)
        freq_diff = abs(valid_results['trades_per_year'] - self.target_trades_per_year)
        valid_results['freq_score'] = 1 / (1 + freq_diff / self.target_trades_per_year)
        
        # Drawdown score (lower drawdown = higher score)
        max_dd = valid_results['max_drawdown'].max()
        if max_dd > 0:
            valid_results['dd_score'] = 1 - (valid_results['max_drawdown'] / max_dd)
        else:
            valid_results['dd_score'] = 1.0
        
        # Combined score with weights
        weights = {
            'sharpe': 0.3,
            'return': 0.25,
            'winrate': 0.2,
            'freq': 0.15,
            'dd': 0.1
        }
        
        valid_results['combined_score'] = (
            weights['sharpe'] * valid_results['sharpe_score'] +
            weights['return'] * valid_results['return_score'] +
            weights['winrate'] * valid_results['winrate_score'] +
            weights['freq'] * valid_results['freq_score'] +
            weights['dd'] * valid_results['dd_score']
        )
        
        # Return threshold with highest combined score
        optimal_idx = valid_results['combined_score'].idxmax()
        return valid_results.loc[optimal_idx, 'threshold']
    
    def _normalize_metric(self, values: pd.Series) -> pd.Series:
        """Normalize metric to [0, 1] range"""
        min_val = values.min()
        max_val = values.max()
        if max_val == min_val:
            return pd.Series(0.5, index=values.index)
        return (values - min_val) / (max_val - min_val)
    
    def _find_pareto_efficient_points(self, results_df: pd.DataFrame) -> List[Tuple[float, float, float]]:
        """Find Pareto-efficient points in trade frequency vs performance space"""
        
        points = []
        for _, row in results_df.iterrows():
            points.append((
                row['threshold'],
                row['trades_per_year'],
                row['sharpe_ratio']
            ))
        
        # Find Pareto frontier
        pareto_points = []
        for i, (thresh_i, freq_i, sharpe_i) in enumerate(points):
            is_dominated = False
            
            for j, (thresh_j, freq_j, sharpe_j) in enumerate(points):
                if i != j:
                    # Point j dominates point i if it has better or equal frequency
                    # and better performance
                    if (freq_j <= freq_i and sharpe_j >= sharpe_i and 
                        (freq_j < freq_i or sharpe_j > sharpe_i)):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_points.append((thresh_i, freq_i, sharpe_i))
        
        # Sort by trade frequency
        pareto_points.sort(key=lambda x: x[1])
        return pareto_points
    
    def _analyze_sensitivity(self, results_df: pd.DataFrame, optimal_threshold: float) -> Dict[str, float]:
        """Analyze sensitivity of optimal threshold"""
        
        # Find nearby thresholds
        threshold_range = 0.1  # ±10% around optimal
        lower_bound = max(optimal_threshold - threshold_range, self.min_threshold)
        upper_bound = min(optimal_threshold + threshold_range, self.max_threshold)
        
        nearby_results = results_df[
            (results_df['threshold'] >= lower_bound) &
            (results_df['threshold'] <= upper_bound)
        ]
        
        if len(nearby_results) <= 1:
            return {'sensitivity_low': 0.0, 'stability_high': 1.0}
        
        # Calculate coefficient of variation for key metrics
        metrics = ['sharpe_ratio', 'trades_per_year', 'win_rate', 'net_return']
        sensitivities = {}
        
        for metric in metrics:
            values = nearby_results[metric]
            if values.mean() != 0:
                cv = values.std() / abs(values.mean())
                sensitivities[f'{metric}_sensitivity'] = cv
            else:
                sensitivities[f'{metric}_sensitivity'] = 0.0
        
        # Overall sensitivity score (lower is better)
        avg_sensitivity = np.mean(list(sensitivities.values()))
        
        return {
            **sensitivities,
            'overall_sensitivity': avg_sensitivity,
            'stability_score': 1 / (1 + avg_sensitivity)
        }
    
    def _test_robustness(
        self,
        threshold: float,
        confidence_scores: np.ndarray,
        signals: np.ndarray,
        returns: np.ndarray,
        n_bootstrap: int = 100
    ) -> float:
        """Test robustness using bootstrap sampling"""
        
        if len(returns) < 50:
            return 0.5  # Insufficient data
        
        n_samples = len(returns)
        bootstrap_metrics = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_confidence = confidence_scores[indices]
            boot_signals = signals[indices]
            boot_returns = returns[indices]
            
            # Analyze threshold on bootstrap sample
            result = self._analyze_single_threshold(
                threshold, boot_confidence, boot_signals, boot_returns
            )
            
            # Store key metrics
            if result['trades_count'] > 0:
                bootstrap_metrics.append({
                    'sharpe': result['sharpe_ratio'],
                    'trades_per_year': result['trades_per_year'],
                    'win_rate': result['win_rate']
                })
        
        if len(bootstrap_metrics) == 0:
            return 0.0
        
        # Calculate robustness as inverse of average coefficient of variation
        bootstrap_df = pd.DataFrame(bootstrap_metrics)
        avg_cv = 0
        n_metrics = 0
        
        for col in bootstrap_df.columns:
            values = bootstrap_df[col]
            if values.mean() != 0:
                cv = values.std() / abs(values.mean())
                avg_cv += cv
                n_metrics += 1
        
        if n_metrics > 0:
            avg_cv /= n_metrics
            robustness = 1 / (1 + avg_cv)
        else:
            robustness = 0.5
        
        return np.clip(robustness, 0, 1)
    
    def plot_threshold_analysis(self, analysis_result: ThresholdAnalysisResult, save_path: Optional[str] = None):
        """Create comprehensive visualization of threshold analysis"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Confidence Threshold Analysis', fontsize=16, fontweight='bold')
        
        df = analysis_result.threshold_curve
        optimal_thresh = analysis_result.optimal_threshold
        
        # 1. Trade Frequency vs Threshold
        ax = axes[0, 0]
        ax.plot(df['threshold'], df['trades_per_year'], 'b-', linewidth=2)
        ax.axhline(y=self.target_trades_per_year, color='r', linestyle='--', alpha=0.7, label='Target')
        ax.axvline(x=optimal_thresh, color='g', linestyle='--', alpha=0.7, label='Optimal')
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('Trades per Year')
        ax.set_title('Trade Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Sharpe Ratio vs Threshold
        ax = axes[0, 1]
        ax.plot(df['threshold'], df['sharpe_ratio'], 'purple', linewidth=2)
        ax.axvline(x=optimal_thresh, color='g', linestyle='--', alpha=0.7, label='Optimal')
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Risk-Adjusted Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Win Rate vs Threshold
        ax = axes[0, 2]
        ax.plot(df['threshold'], df['win_rate'] * 100, 'orange', linewidth=2)
        ax.axvline(x=optimal_thresh, color='g', linestyle='--', alpha=0.7, label='Optimal')
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Prediction Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Net Return vs Threshold
        ax = axes[1, 0]
        ax.plot(df['threshold'], df['net_return'] * 100, 'darkgreen', linewidth=2)
        ax.axvline(x=optimal_thresh, color='g', linestyle='--', alpha=0.7, label='Optimal')
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('Net Annual Return (%)')
        ax.set_title('Net Returns (After Costs)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Pareto Frontier
        ax = axes[1, 1]
        ax.scatter(df['trades_per_year'], df['sharpe_ratio'], alpha=0.6, c='lightblue', s=30)
        
        # Highlight Pareto efficient points
        pareto_trades = [p[1] for p in analysis_result.pareto_efficient_points]
        pareto_sharpe = [p[2] for p in analysis_result.pareto_efficient_points]
        ax.plot(pareto_trades, pareto_sharpe, 'r-o', linewidth=2, markersize=6, label='Pareto Frontier')
        
        # Highlight optimal point
        optimal_row = df[df['threshold'] == optimal_thresh].iloc[0]
        ax.plot(optimal_row['trades_per_year'], optimal_row['sharpe_ratio'], 
                'g*', markersize=15, label='Optimal')
        
        ax.set_xlabel('Trades per Year')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Efficiency Frontier')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Summary Statistics
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = f"""
OPTIMAL THRESHOLD ANALYSIS

Optimal Threshold: {optimal_thresh:.3f}

Expected Performance:
  Trades/Year: {analysis_result.expected_trades_per_year:.1f}
  Win Rate: {analysis_result.expected_win_rate*100:.1f}%
  Sharpe Ratio: {analysis_result.expected_sharpe:.2f}
  Annual Return: {analysis_result.expected_return*100:.1f}%

Robustness Metrics:
  Stability Score: {analysis_result.robustness_score:.3f}
  Sensitivity: {analysis_result.sensitivity_metrics.get('overall_sensitivity', 0):.3f}

Trade-off Analysis:
  Pareto Points: {len(analysis_result.pareto_efficient_points)}
  Frequency Target: {self.target_trades_per_year} trades/year
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Analysis plot saved to {save_path}")
        
        return fig
    
    def generate_optimization_report(self, analysis_result: ThresholdAnalysisResult) -> str:
        """Generate detailed optimization report"""
        
        report = []
        report.append("=" * 80)
        report.append("CONFIDENCE THRESHOLD OPTIMIZATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Optimal Confidence Threshold: {analysis_result.optimal_threshold:.3f}")
        report.append(f"Expected Trades per Year: {analysis_result.expected_trades_per_year:.1f}")
        report.append(f"Target was: {self.target_trades_per_year}")
        report.append(f"Expected Win Rate: {analysis_result.expected_win_rate*100:.1f}%")
        report.append(f"Expected Sharpe Ratio: {analysis_result.expected_sharpe:.2f}")
        report.append(f"Expected Annual Return: {analysis_result.expected_return*100:.1f}%")
        report.append("")
        
        # Performance Improvements
        df = analysis_result.threshold_curve
        baseline_row = df[df['threshold'] == 0.5].iloc[0] if len(df[df['threshold'] == 0.5]) > 0 else df.iloc[0]
        optimal_row = df[df['threshold'] == analysis_result.optimal_threshold].iloc[0]
        
        report.append("EXPECTED IMPROVEMENTS vs BASELINE (50% threshold)")
        report.append("-" * 40)
        
        trade_reduction = (baseline_row['trades_per_year'] - optimal_row['trades_per_year']) / baseline_row['trades_per_year']
        sharpe_improvement = (optimal_row['sharpe_ratio'] - baseline_row['sharpe_ratio']) / abs(baseline_row['sharpe_ratio']) if baseline_row['sharpe_ratio'] != 0 else 0
        winrate_improvement = (optimal_row['win_rate'] - baseline_row['win_rate']) / baseline_row['win_rate'] if baseline_row['win_rate'] != 0 else 0
        
        report.append(f"Trade Frequency Reduction: {trade_reduction*100:.1f}%")
        report.append(f"Sharpe Ratio Improvement: {sharpe_improvement*100:.1f}%")
        report.append(f"Win Rate Improvement: {winrate_improvement*100:.1f}%")
        report.append("")
        
        # Robustness Analysis
        report.append("ROBUSTNESS & SENSITIVITY ANALYSIS")
        report.append("-" * 40)
        report.append(f"Robustness Score: {analysis_result.robustness_score:.3f} (0=fragile, 1=robust)")
        report.append(f"Overall Sensitivity: {analysis_result.sensitivity_metrics.get('overall_sensitivity', 0):.3f}")
        report.append(f"Stability Score: {analysis_result.sensitivity_metrics.get('stability_score', 0):.3f}")
        report.append("")
        
        # Pareto Analysis
        report.append("PARETO EFFICIENCY ANALYSIS")
        report.append("-" * 40)
        report.append(f"Number of Pareto-efficient points: {len(analysis_result.pareto_efficient_points)}")
        report.append("Top 5 Pareto-efficient thresholds:")
        
        pareto_sorted = sorted(analysis_result.pareto_efficient_points, key=lambda x: x[2], reverse=True)[:5]
        for i, (thresh, trades, sharpe) in enumerate(pareto_sorted, 1):
            report.append(f"  {i}. Threshold={thresh:.3f}, Trades/Year={trades:.1f}, Sharpe={sharpe:.2f}")
        report.append("")
        
        # Risk Analysis
        report.append("RISK ANALYSIS")
        report.append("-" * 40)
        max_dd = optimal_row['max_drawdown']
        profit_factor = optimal_row['profit_factor']
        
        report.append(f"Maximum Drawdown: {max_dd*100:.2f}%")
        report.append(f"Profit Factor: {profit_factor:.2f}")
        
        if max_dd > 0.2:
            report.append("⚠️  WARNING: High drawdown risk")
        else:
            report.append("✅ Drawdown within acceptable limits")
            
        if profit_factor < 1.5:
            report.append("⚠️  WARNING: Low profit factor")
        else:
            report.append("✅ Healthy profit factor")
        report.append("")
        
        # Implementation Recommendations
        report.append("IMPLEMENTATION RECOMMENDATIONS")
        report.append("-" * 40)
        
        if analysis_result.robustness_score > 0.7:
            report.append("✅ High robustness - implement with confidence")
        elif analysis_result.robustness_score > 0.5:
            report.append("⚠️  Moderate robustness - monitor closely")
        else:
            report.append("❌ Low robustness - consider alternative approaches")
            
        if optimal_row['trades_per_year'] > self.target_trades_per_year * 1.5:
            report.append("⚠️  Trade frequency higher than target - consider increasing threshold")
        elif optimal_row['trades_per_year'] < self.target_trades_per_year * 0.5:
            report.append("⚠️  Trade frequency lower than target - consider decreasing threshold")
        else:
            report.append("✅ Trade frequency within target range")
            
        # Threshold ranges for different scenarios
        report.append("")
        report.append("THRESHOLD RECOMMENDATIONS BY SCENARIO")
        report.append("-" * 40)
        
        conservative_thresh = min(analysis_result.optimal_threshold + 0.1, 0.9)
        aggressive_thresh = max(analysis_result.optimal_threshold - 0.05, 0.3)
        
        report.append(f"Conservative trading: {conservative_thresh:.3f} (fewer, higher confidence trades)")
        report.append(f"Balanced trading: {analysis_result.optimal_threshold:.3f} (optimal risk-return)")
        report.append(f"Aggressive trading: {aggressive_thresh:.3f} (more trades, accept lower confidence)")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def demonstrate_threshold_optimization():
    """Demonstrate threshold optimization with synthetic data"""
    
    print("🚀 Confidence Threshold Optimization Demo")
    print("=" * 50)
    
    # Generate synthetic trading data
    np.random.seed(42)
    n_days = 500
    
    # Synthetic confidence scores (beta distribution for realism)
    confidence_scores = np.random.beta(2, 3, n_days)
    
    # Synthetic signals with higher probability when confidence is high
    signal_prob = confidence_scores * 0.8 + 0.1
    signals = np.random.choice([-1, 0, 1], n_days, p=[0.3, 0.4, 0.3])
    
    # Synthetic returns correlated with confidence
    base_returns = np.random.normal(0.001, 0.02, n_days)
    confidence_boost = (confidence_scores - 0.5) * 0.01  # Higher confidence = better expected return
    returns = base_returns + confidence_boost
    
    # Add some market regime effects
    regime_changes = np.random.poisson(0.01, n_days).astype(bool)
    regime = np.zeros(n_days)
    current_regime = 0
    for i in range(n_days):
        if regime_changes[i]:
            current_regime = 1 - current_regime
        regime[i] = current_regime
    
    # Adjust returns based on regime
    returns[regime == 0] *= 1.2  # Bull market
    returns[regime == 1] *= 0.8  # Bear market
    
    print(f"📊 Generated {n_days} days of synthetic trading data")
    print(f"   Confidence scores: {confidence_scores.min():.3f} - {confidence_scores.max():.3f}")
    print(f"   Signals: {np.sum(signals != 0)} non-zero")
    print(f"   Average return: {np.mean(returns)*100:.3f}%")
    
    # Initialize optimizer
    optimizer = ConfidenceThresholdOptimizer(
        transaction_cost=0.002,
        target_trades_per_year=40
    )
    
    # Run optimization analysis
    print("\n🔍 Running threshold optimization analysis...")
    analysis_result = optimizer.analyze_threshold_impact(
        confidence_scores=confidence_scores,
        signals=signals,
        returns=returns
    )
    
    # Generate report
    print("\n📋 Generating optimization report...")
    report = optimizer.generate_optimization_report(analysis_result)
    print(report)
    
    # Create visualization
    print("\n📊 Creating analysis plots...")
    try:
        fig = optimizer.plot_threshold_analysis(analysis_result)
        plt.show()
    except Exception as e:
        print(f"   Plot generation failed: {e}")
    
    # Summary of key findings
    print("\n" + "=" * 50)
    print("KEY FINDINGS SUMMARY")
    print("=" * 50)
    
    df = analysis_result.threshold_curve
    baseline_trades = df[df['threshold'] == 0.5]['trades_per_year'].iloc[0] if len(df[df['threshold'] == 0.5]) > 0 else 100
    optimal_trades = analysis_result.expected_trades_per_year
    
    reduction = (baseline_trades - optimal_trades) / baseline_trades * 100
    
    print(f"🎯 Optimal threshold: {analysis_result.optimal_threshold:.3f}")
    print(f"📉 Trade reduction: {reduction:.1f}%")
    print(f"📈 Expected Sharpe: {analysis_result.expected_sharpe:.2f}")
    print(f"🏆 Win rate: {analysis_result.expected_win_rate*100:.1f}%")
    print(f"🛡️  Robustness: {analysis_result.robustness_score:.3f}")
    
    # Success assessment
    success_count = 0
    if reduction >= 30:
        print("✅ Significant trade reduction achieved")
        success_count += 1
    if analysis_result.expected_sharpe > 0.5:
        print("✅ Good risk-adjusted returns")
        success_count += 1
    if analysis_result.expected_win_rate > 0.55:
        print("✅ High win rate achieved")
        success_count += 1
    if analysis_result.robustness_score > 0.6:
        print("✅ Robust threshold found")
        success_count += 1
    
    print(f"\n🎉 Success score: {success_count}/4 criteria met")
    
    if success_count >= 3:
        print("🏆 EXCELLENT: Confidence filtering highly effective!")
    elif success_count >= 2:
        print("👍 GOOD: Confidence filtering shows promise")
    else:
        print("⚠️  NEEDS IMPROVEMENT: Consider alternative approaches")


if __name__ == "__main__":
    demonstrate_threshold_optimization()