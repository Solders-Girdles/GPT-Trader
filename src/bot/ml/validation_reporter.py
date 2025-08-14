"""
Validation Report Generator
Phase 2.5 - Day 7

Generates comprehensive validation reports from walk-forward validation results.
"""

import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Import our validation framework
from .walk_forward_validator import WalkForwardResults, FoldResult
from .model_degradation_monitor import DegradationMetrics, DegradationStatus

logger = logging.getLogger(__name__)


class ValidationReporter:
    """
    Generates comprehensive validation reports with visualizations.
    
    Features:
    - HTML report generation
    - Performance summary tables
    - Degradation analysis
    - Feature importance visualization
    - Backtesting results summary
    """
    
    def __init__(self, output_dir: str = "validation_reports"):
        """
        Initialize reporter.
        
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for visualizations
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        logger.info(f"ValidationReporter initialized with output directory: {self.output_dir}")
    
    def generate_report(self,
                       results: WalkForwardResults,
                       model_name: Optional[str] = None,
                       save_html: bool = True,
                       save_json: bool = True,
                       save_plots: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            results: Walk-forward validation results
            model_name: Optional model name override
            save_html: Save HTML report
            save_json: Save JSON data
            save_plots: Save plot images
            
        Returns:
            Report summary dictionary
        """
        model_name = model_name or results.model_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Generating validation report for {model_name}")
        
        # Create report structure
        report = {
            'metadata': self._generate_metadata(results, timestamp),
            'summary': self._generate_summary(results),
            'fold_analysis': self._analyze_folds(results),
            'degradation_analysis': self._analyze_degradation(results),
            'feature_analysis': self._analyze_features(results),
            'backtesting_summary': self._summarize_backtesting(results),
            'recommendations': self._generate_recommendations(results)
        }
        
        # Save outputs
        if save_json:
            json_path = self._save_json_report(report, model_name, timestamp)
            report['json_path'] = str(json_path)
        
        if save_plots:
            plot_paths = self._generate_and_save_plots(results, model_name, timestamp)
            report['plot_paths'] = plot_paths
        
        if save_html:
            html_path = self._generate_html_report(report, results, model_name, timestamp)
            report['html_path'] = str(html_path)
        
        logger.info(f"Report generation complete for {model_name}")
        
        return report
    
    def _generate_metadata(self, results: WalkForwardResults, timestamp: str) -> Dict:
        """Generate report metadata"""
        return {
            'model_name': results.model_name,
            'timestamp': timestamp,
            'n_folds': results.n_folds,
            'train_window': results.config.train_window,
            'test_window': results.config.test_window,
            'expanding_window': results.config.expanding_window,
            'backtest_enabled': results.config.backtest_each_fold
        }
    
    def _generate_summary(self, results: WalkForwardResults) -> Dict:
        """Generate performance summary"""
        return {
            'accuracy': {
                'mean': results.mean_test_accuracy,
                'std': results.std_test_accuracy,
                'min': min(f.test_accuracy for f in results.fold_results),
                'max': max(f.test_accuracy for f in results.fold_results)
            },
            'f1_score': {
                'mean': results.mean_test_f1,
                'std': results.std_test_f1,
                'min': min(f.test_f1 for f in results.fold_results),
                'max': max(f.test_f1 for f in results.fold_results)
            },
            'sharpe_ratio': {
                'mean': results.mean_sharpe,
                'min': min((f.sharpe_ratio for f in results.fold_results if f.sharpe_ratio), default=None),
                'max': max((f.sharpe_ratio for f in results.fold_results if f.sharpe_ratio), default=None)
            } if results.mean_sharpe else None,
            'returns': {
                'mean': results.mean_return,
                'total': sum(f.total_return for f in results.fold_results if f.total_return)
            } if results.mean_return else None,
            'stability_score': results.stability_score,
            'degradation_detected': results.degradation_detected
        }
    
    def _analyze_folds(self, results: WalkForwardResults) -> Dict:
        """Analyze individual fold performance"""
        fold_analysis = []
        
        for fold in results.fold_results:
            analysis = {
                'fold_number': fold.fold_number,
                'date_range': f"{fold.train_start.date()} to {fold.test_end.date()}",
                'train_samples': (fold.train_end - fold.train_start).days,
                'test_samples': (fold.test_end - fold.test_start).days,
                'accuracy': fold.test_accuracy,
                'f1_score': fold.test_f1,
                'precision': fold.precision,
                'recall': fold.recall,
                'roc_auc': fold.roc_auc,
                'is_degraded': fold.is_degraded,
                'performance_change': fold.performance_change
            }
            
            if fold.backtest_results:
                analysis['backtesting'] = {
                    'sharpe_ratio': fold.sharpe_ratio,
                    'max_drawdown': fold.max_drawdown,
                    'total_return': fold.total_return
                }
            
            fold_analysis.append(analysis)
        
        return {
            'folds': fold_analysis,
            'best_fold': max(fold_analysis, key=lambda x: x['accuracy'])['fold_number'],
            'worst_fold': min(fold_analysis, key=lambda x: x['accuracy'])['fold_number']
        }
    
    def _analyze_degradation(self, results: WalkForwardResults) -> Dict:
        """Analyze model degradation patterns"""
        degradation_analysis = {
            'degradation_detected': results.degradation_detected,
            'degraded_folds': results.degradation_folds,
            'degradation_rate': len(results.degradation_folds) / results.n_folds if results.n_folds > 0 else 0
        }
        
        if results.degradation_folds:
            # Analyze degradation patterns
            degraded_accuracies = [f.test_accuracy for f in results.fold_results if f.is_degraded]
            normal_accuracies = [f.test_accuracy for f in results.fold_results if not f.is_degraded]
            
            degradation_analysis['degraded_performance'] = {
                'mean_accuracy': np.mean(degraded_accuracies) if degraded_accuracies else None,
                'accuracy_drop': np.mean(normal_accuracies) - np.mean(degraded_accuracies) if normal_accuracies and degraded_accuracies else None
            }
            
            # Identify degradation triggers
            triggers = []
            for fold in results.fold_results:
                if fold.is_degraded and fold.performance_change:
                    if fold.performance_change < -0.1:
                        triggers.append(f"Fold {fold.fold_number}: Sudden drop ({fold.performance_change:.3f})")
                    elif fold.performance_change < -0.05:
                        triggers.append(f"Fold {fold.fold_number}: Gradual decline ({fold.performance_change:.3f})")
            
            degradation_analysis['triggers'] = triggers
        
        return degradation_analysis
    
    def _analyze_features(self, results: WalkForwardResults) -> Dict:
        """Analyze feature importance and stability"""
        if not results.feature_importance_stability:
            return {'stability_analysis': 'Not available'}
        
        # Sort features by stability
        sorted_features = sorted(
            results.feature_importance_stability.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'n_features': len(sorted_features),
            'most_stable': sorted_features[:10],
            'least_stable': sorted_features[-10:],
            'mean_stability': np.mean(list(results.feature_importance_stability.values())),
            'stability_distribution': {
                'high': sum(1 for _, v in sorted_features if v > 0.8),
                'medium': sum(1 for _, v in sorted_features if 0.5 <= v <= 0.8),
                'low': sum(1 for _, v in sorted_features if v < 0.5)
            }
        }
    
    def _summarize_backtesting(self, results: WalkForwardResults) -> Optional[Dict]:
        """Summarize backtesting results"""
        if not results.mean_sharpe:
            return None
        
        # Collect all backtesting metrics
        sharpe_ratios = [f.sharpe_ratio for f in results.fold_results if f.sharpe_ratio is not None]
        returns = [f.total_return for f in results.fold_results if f.total_return is not None]
        drawdowns = [f.max_drawdown for f in results.fold_results if f.max_drawdown is not None]
        
        return {
            'sharpe_ratio': {
                'mean': np.mean(sharpe_ratios) if sharpe_ratios else None,
                'std': np.std(sharpe_ratios) if sharpe_ratios else None,
                'positive_rate': sum(1 for s in sharpe_ratios if s > 0) / len(sharpe_ratios) if sharpe_ratios else 0
            },
            'returns': {
                'mean': np.mean(returns) if returns else None,
                'total': sum(returns) if returns else None,
                'positive_rate': sum(1 for r in returns if r > 0) / len(returns) if returns else 0
            },
            'max_drawdown': {
                'mean': np.mean(drawdowns) if drawdowns else None,
                'worst': min(drawdowns) if drawdowns else None
            },
            'risk_adjusted_return': results.mean_sharpe * np.sqrt(252) if results.mean_sharpe else None  # Annualized
        }
    
    def _generate_recommendations(self, results: WalkForwardResults) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Performance recommendations
        if results.mean_test_accuracy < 0.55:
            recommendations.append("Model accuracy below acceptable threshold. Consider feature engineering improvements.")
        
        if results.std_test_accuracy > 0.1:
            recommendations.append("High variance in accuracy across folds. Model may be unstable.")
        
        # Degradation recommendations
        if results.degradation_detected:
            degradation_rate = len(results.degradation_folds) / results.n_folds
            if degradation_rate > 0.3:
                recommendations.append(f"High degradation rate ({degradation_rate:.1%}). Implement regular retraining.")
            else:
                recommendations.append(f"Degradation detected in {len(results.degradation_folds)} folds. Monitor closely.")
        
        # Stability recommendations
        if results.stability_score < 0.7:
            recommendations.append("Low stability score. Consider ensemble methods or regularization.")
        
        # Backtesting recommendations
        if results.mean_sharpe and results.mean_sharpe < 0.5:
            recommendations.append("Low Sharpe ratio. Review risk management and signal generation.")
        
        if results.mean_drawdown and results.mean_drawdown < -0.2:
            recommendations.append("High average drawdown. Implement stricter position sizing.")
        
        # Feature recommendations
        if results.feature_importance_stability:
            low_stability = sum(1 for v in results.feature_importance_stability.values() if v < 0.5)
            if low_stability > len(results.feature_importance_stability) * 0.3:
                recommendations.append("Many unstable features. Consider feature selection refinement.")
        
        if not recommendations:
            recommendations.append("Model performance is acceptable. Continue monitoring.")
        
        return recommendations
    
    def _generate_and_save_plots(self, results: WalkForwardResults, model_name: str, timestamp: str) -> Dict[str, str]:
        """Generate and save visualization plots"""
        plot_paths = {}
        
        # 1. Performance over folds
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        df = results.to_dataframe()
        
        # Accuracy plot
        ax = axes[0, 0]
        ax.plot(df['fold'], df['train_accuracy'], 'b-', label='Train', marker='o')
        ax.plot(df['fold'], df['test_accuracy'], 'r-', label='Test', marker='s')
        ax.axhline(y=results.mean_test_accuracy, color='r', linestyle='--', alpha=0.5)
        ax.fill_between(df['fold'],
                        results.mean_test_accuracy - results.std_test_accuracy,
                        results.mean_test_accuracy + results.std_test_accuracy,
                        alpha=0.2, color='r')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Across Folds')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # F1 Score plot
        ax = axes[0, 1]
        ax.plot(df['fold'], df['train_f1'], 'b-', label='Train', marker='o')
        ax.plot(df['fold'], df['test_f1'], 'r-', label='Test', marker='s')
        ax.set_xlabel('Fold')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score Across Folds')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Sharpe Ratio plot
        if 'sharpe_ratio' in df.columns and df['sharpe_ratio'].notna().any():
            ax = axes[1, 0]
            ax.bar(df['fold'], df['sharpe_ratio'].fillna(0))
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            if results.mean_sharpe:
                ax.axhline(y=results.mean_sharpe, color='g', linestyle='--', alpha=0.5, label=f'Mean: {results.mean_sharpe:.2f}')
            ax.set_xlabel('Fold')
            ax.set_ylabel('Sharpe Ratio')
            ax.set_title('Sharpe Ratio by Fold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Returns distribution
        if 'total_return' in df.columns and df['total_return'].notna().any():
            ax = axes[1, 1]
            returns = df['total_return'].dropna() * 100
            ax.hist(returns, bins=min(len(returns), 20), edgecolor='black', alpha=0.7)
            ax.axvline(x=returns.mean(), color='r', linestyle='--', label=f'Mean: {returns.mean():.1f}%')
            ax.set_xlabel('Return (%)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Returns')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Mark degraded folds
        if results.degradation_folds:
            for fold in results.degradation_folds:
                for ax in axes.flat:
                    ax.axvspan(fold - 0.5, fold + 0.5, alpha=0.2, color='red')
        
        plt.suptitle(f'Walk-Forward Validation: {model_name}', fontsize=14)
        plt.tight_layout()
        
        plot_path = self.output_dir / f"{model_name}_{timestamp}_performance.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        plot_paths['performance'] = str(plot_path)
        
        # 2. Feature stability plot
        if results.feature_importance_stability:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Get top 20 features by stability
            sorted_features = sorted(
                results.feature_importance_stability.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
            
            features, stabilities = zip(*sorted_features)
            colors = ['green' if s > 0.8 else 'orange' if s > 0.5 else 'red' for s in stabilities]
            
            ax.barh(range(len(features)), stabilities, color=colors)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Stability Score')
            ax.set_title(f'Top 20 Features by Stability - {model_name}')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_path = self.output_dir / f"{model_name}_{timestamp}_feature_stability.png"
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            plot_paths['feature_stability'] = str(plot_path)
        
        return plot_paths
    
    def _save_json_report(self, report: Dict, model_name: str, timestamp: str) -> Path:
        """Save report as JSON"""
        json_path = self.output_dir / f"{model_name}_{timestamp}_report.json"
        
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"JSON report saved to {json_path}")
        return json_path
    
    def _generate_html_report(self, report: Dict, results: WalkForwardResults, 
                             model_name: str, timestamp: str) -> Path:
        """Generate HTML report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Validation Report - {model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metric {{ background-color: #e8f5e9; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .warning {{ background-color: #fff3e0; }}
        .error {{ background-color: #ffebee; }}
        .recommendation {{ background-color: #e3f2fd; padding: 10px; margin: 5px 0; border-left: 4px solid #2196F3; }}
        .plot {{ margin: 20px 0; text-align: center; }}
        .plot img {{ max-width: 100%; border: 1px solid #ddd; }}
    </style>
</head>
<body>
    <h1>Walk-Forward Validation Report: {model_name}</h1>
    <p><strong>Generated:</strong> {timestamp}</p>
    
    <h2>Summary</h2>
    <div class="metric">
        <p><strong>Accuracy:</strong> {report['summary']['accuracy']['mean']:.3f} ± {report['summary']['accuracy']['std']:.3f}</p>
        <p><strong>F1 Score:</strong> {report['summary']['f1_score']['mean']:.3f} ± {report['summary']['f1_score']['std']:.3f}</p>
        <p><strong>Stability Score:</strong> {report['summary']['stability_score']:.3f}</p>
        <p><strong>Degradation Detected:</strong> {'Yes' if report['summary']['degradation_detected'] else 'No'}</p>
    </div>
    
    <h2>Configuration</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Number of Folds</td><td>{report['metadata']['n_folds']}</td></tr>
        <tr><td>Train Window</td><td>{report['metadata']['train_window']} days</td></tr>
        <tr><td>Test Window</td><td>{report['metadata']['test_window']} days</td></tr>
        <tr><td>Window Type</td><td>{'Expanding' if report['metadata']['expanding_window'] else 'Rolling'}</td></tr>
        <tr><td>Backtesting</td><td>{'Enabled' if report['metadata']['backtest_enabled'] else 'Disabled'}</td></tr>
    </table>
    
    <h2>Fold Performance</h2>
    <table>
        <tr>
            <th>Fold</th>
            <th>Date Range</th>
            <th>Accuracy</th>
            <th>F1 Score</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>Status</th>
        </tr>
"""
        
        for fold in report['fold_analysis']['folds']:
            status_class = 'error' if fold['is_degraded'] else ''
            status_text = 'Degraded' if fold['is_degraded'] else 'Normal'
            html_content += f"""
        <tr class="{status_class}">
            <td>{fold['fold_number']}</td>
            <td>{fold['date_range']}</td>
            <td>{fold['accuracy']:.3f}</td>
            <td>{fold['f1_score']:.3f}</td>
            <td>{fold['precision']:.3f}</td>
            <td>{fold['recall']:.3f}</td>
            <td>{status_text}</td>
        </tr>
"""
        
        html_content += """
    </table>
"""
        
        # Backtesting results if available
        if report['backtesting_summary']:
            html_content += f"""
    <h2>Backtesting Results</h2>
    <div class="metric">
        <p><strong>Mean Sharpe Ratio:</strong> {report['backtesting_summary']['sharpe_ratio']['mean']:.2f}</p>
        <p><strong>Mean Return:</strong> {report['backtesting_summary']['returns']['mean']*100:.1f}%</p>
        <p><strong>Mean Max Drawdown:</strong> {report['backtesting_summary']['max_drawdown']['mean']*100:.1f}%</p>
        <p><strong>Risk-Adjusted Return:</strong> {report['backtesting_summary']['risk_adjusted_return']:.2f}</p>
    </div>
"""
        
        # Degradation analysis
        if report['degradation_analysis']['degradation_detected']:
            html_content += f"""
    <h2>Degradation Analysis</h2>
    <div class="metric warning">
        <p><strong>Degraded Folds:</strong> {report['degradation_analysis']['degraded_folds']}</p>
        <p><strong>Degradation Rate:</strong> {report['degradation_analysis']['degradation_rate']*100:.1f}%</p>
"""
            if report['degradation_analysis'].get('triggers'):
                html_content += "<p><strong>Triggers:</strong></p><ul>"
                for trigger in report['degradation_analysis']['triggers']:
                    html_content += f"<li>{trigger}</li>"
                html_content += "</ul>"
            html_content += "</div>"
        
        # Recommendations
        html_content += """
    <h2>Recommendations</h2>
"""
        for rec in report['recommendations']:
            html_content += f'    <div class="recommendation">{rec}</div>\n'
        
        # Plots
        if 'plot_paths' in report:
            html_content += """
    <h2>Visualizations</h2>
"""
            for plot_name, plot_path in report['plot_paths'].items():
                if Path(plot_path).exists():
                    html_content += f"""
    <div class="plot">
        <h3>{plot_name.replace('_', ' ').title()}</h3>
        <img src="{Path(plot_path).name}" alt="{plot_name}">
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        html_path = self.output_dir / f"{model_name}_{timestamp}_report.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {html_path}")
        return html_path


def create_validation_reporter(output_dir: str = "validation_reports") -> ValidationReporter:
    """Create validation reporter instance"""
    return ValidationReporter(output_dir)


if __name__ == "__main__":
    # Example usage
    from .walk_forward_validator import create_walk_forward_validator, WalkForwardConfig
    import yfinance as yf
    from sklearn.ensemble import RandomForestClassifier
    
    # Get sample data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="3y")
    
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
    
    # Get prices
    prices = data['Close'].loc[X.index]
    
    # Run walk-forward validation
    config = WalkForwardConfig(
        train_window=252,
        test_window=63,
        step_size=21,
        backtest_each_fold=True
    )
    
    validator = create_walk_forward_validator(config)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print("Running walk-forward validation...")
    results = validator.validate(model, X, y, prices, "RandomForest")
    
    # Generate report
    reporter = create_validation_reporter()
    print("Generating validation report...")
    report = reporter.generate_report(results)
    
    print(f"\nReport saved to: {report.get('html_path', 'N/A')}")
    print("\nSummary:")
    print(f"  Accuracy: {results.mean_test_accuracy:.3f} ± {results.std_test_accuracy:.3f}")
    print(f"  Stability Score: {results.stability_score:.3f}")
    print(f"  Degradation Detected: {results.degradation_detected}")