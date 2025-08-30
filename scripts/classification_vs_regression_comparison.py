#!/usr/bin/env python3
"""
Classification vs Regression Trading Comparison
==============================================

Comprehensive comparison between binary classification and regression
approaches for ML-based trading strategies.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.dataflow.sources.yfinance_source import YFinanceSource
from bot.ml.regression_position_sizer import (
    RegressionPositionSizer, 
    PositionSizingConfig
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create features for both classification and regression models."""
    features = pd.DataFrame(index=data.index)
    
    # Price momentum features
    for period in [1, 3, 5, 10, 20]:
        features[f'return_{period}d'] = data['close'].pct_change(period)
        features[f'momentum_{period}d'] = data['close'] / data['close'].shift(period) - 1
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        ma = data['close'].rolling(period).mean()
        features[f'sma_{period}'] = ma
        features[f'price_to_sma_{period}'] = data['close'] / ma - 1
    
    # Cross-MA features
    features['sma5_vs_sma20'] = features['sma_5'] / features['sma_20'] - 1
    features['sma10_vs_sma50'] = features['sma_10'] / features['sma_50'] - 1
    
    # Volatility
    for period in [5, 10, 20]:
        vol = data['close'].rolling(period).std()
        features[f'volatility_{period}d'] = vol
        features[f'vol_ratio_{period}d'] = vol / vol.rolling(50).mean()
    
    # Volume features
    features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
    features['volume_trend'] = (data['volume'].rolling(5).mean() / 
                               data['volume'].rolling(20).mean())
    
    # Technical indicators
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    sma20 = data['close'].rolling(20).mean()
    std20 = data['close'].rolling(20).std()
    features['bb_position'] = (data['close'] - (sma20 - 2*std20)) / (4 * std20)
    
    return features


class ClassificationStrategy:
    """Traditional binary classification trading strategy."""
    
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.model = None
        
    def train(self, X: pd.DataFrame, prices: pd.DataFrame, horizon: int = 5):
        """Train classification model."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create binary labels: 1 if future return > threshold, 0 otherwise
        future_returns = prices['close'].pct_change(horizon).shift(-horizon)
        y = (future_returns > self.threshold).astype(int)
        
        # Remove NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=50,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_clean, y_clean)
        
        return self.model
    
    def generate_positions(self, X: pd.DataFrame, base_position: float = 0.1):
        """Generate positions based on classification predictions."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Get predictions
        probabilities = self.model.predict_proba(X.fillna(0))[:, 1]
        
        # Generate fixed-size positions
        positions = np.zeros_like(probabilities)
        positions[probabilities > 0.6] = base_position   # Buy signal
        positions[probabilities < 0.4] = -base_position  # Sell signal
        
        return positions


class RegressionStrategy:
    """Regression-based trading strategy with dynamic position sizing."""
    
    def __init__(self):
        self.models = {}
        self.position_sizer = RegressionPositionSizer(
            PositionSizingConfig(
                base_risk_budget=0.02,
                max_position_size=0.2,
                transaction_cost=0.001
            )
        )
        
    def train(self, X: pd.DataFrame, prices: pd.DataFrame, horizon: int = 5):
        """Train regression models."""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge
        
        # Create continuous target: actual future returns
        y = prices['close'].pct_change(horizon).shift(-horizon)
        
        # Remove NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        # Train ensemble of regressors
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=50,
                random_state=42,
                n_jobs=-1
            ),
            'gbr': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0, random_state=42)
        }
        
        # Train all models
        for name, model in self.models.items():
            model.fit(X_clean, y_clean)
        
        return self.models
    
    def predict_returns(self, X: pd.DataFrame):
        """Predict returns using ensemble."""
        if not self.models:
            raise ValueError("Models not trained")
        
        predictions = []
        weights = {'rf': 0.4, 'gbr': 0.4, 'ridge': 0.2}
        
        for name, model in self.models.items():
            pred = model.predict(X.fillna(0))
            weight = weights[name]
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)
    
    def generate_positions(self, X: pd.DataFrame, symbol: str = 'SYMBOL', price: float = 100.0):
        """Generate positions based on regression predictions."""
        predicted_returns = self.predict_returns(X)
        
        # Calculate confidence (simplified)
        feature_completeness = (1 - X.isna().sum(axis=1) / len(X.columns))
        confidences = np.clip(feature_completeness * 0.8 + 0.2, 0.3, 1.0)
        
        # Calculate individual position sizes
        positions = []
        for pred_return, confidence in zip(predicted_returns, confidences):
            pos_size = self.position_sizer.calculate_position_size(
                predicted_return=pred_return,
                current_price=price,
                symbol=symbol,
                confidence=confidence,
                volatility=0.25  # Assume 25% volatility
            )
            positions.append(pos_size)
        
        return np.array(positions)


def backtest_strategy(strategy, features: pd.DataFrame, prices: pd.DataFrame, 
                     symbol: str = 'SYMBOL', horizon: int = 5):
    """Backtest a strategy."""
    
    # Use first 80% for training
    split_idx = int(len(features) * 0.8)
    
    X_train = features.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    prices_train = prices.iloc[:split_idx]
    prices_test = prices.iloc[split_idx:]
    
    # Train strategy
    strategy.train(X_train, prices_train, horizon)
    
    # Generate test positions
    if isinstance(strategy, ClassificationStrategy):
        positions = strategy.generate_positions(X_test)
    else:
        positions = strategy.generate_positions(X_test, symbol, prices_test['close'].iloc[0])
    
    # Calculate actual returns
    actual_returns = prices_test['close'].pct_change(horizon).shift(-horizon).iloc[:-horizon]
    positions = positions[:len(actual_returns)]
    
    # Calculate strategy returns
    strategy_returns = positions * actual_returns
    
    # Performance metrics
    total_return = strategy_returns.sum()
    volatility = strategy_returns.std() * np.sqrt(252/horizon)
    sharpe_ratio = (strategy_returns.mean() / strategy_returns.std() * 
                   np.sqrt(252/horizon)) if strategy_returns.std() > 0 else 0
    
    max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min()
    
    return {
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'n_trades': len(strategy_returns),
        'strategy_returns': strategy_returns,
        'positions': positions,
        'actual_returns': actual_returns
    }


def plot_comparison(classification_results: dict, regression_results: dict, symbol: str):
    """Plot comparison between strategies."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Cumulative returns
    clf_cumret = classification_results['strategy_returns'].cumsum()
    reg_cumret = regression_results['strategy_returns'].cumsum()
    
    axes[0, 0].plot(clf_cumret.index, clf_cumret.values, label='Classification', alpha=0.7)
    axes[0, 0].plot(reg_cumret.index, reg_cumret.values, label='Regression', alpha=0.7)
    axes[0, 0].set_title(f'Cumulative Returns - {symbol}')
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Position sizes over time
    axes[0, 1].plot(classification_results['positions'], label='Classification', alpha=0.7)
    axes[0, 1].plot(regression_results['positions'], label='Regression', alpha=0.7)
    axes[0, 1].set_title('Position Sizes Over Time')
    axes[0, 1].set_ylabel('Position Size')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Return distributions
    axes[1, 0].hist(classification_results['strategy_returns'], bins=30, alpha=0.7, 
                   label='Classification', density=True)
    axes[1, 0].hist(regression_results['strategy_returns'], bins=30, alpha=0.7, 
                   label='Regression', density=True)
    axes[1, 0].set_title('Return Distributions')
    axes[1, 0].set_xlabel('Daily Return')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Performance metrics comparison
    metrics = ['total_return', 'sharpe_ratio', 'volatility']
    clf_values = [classification_results[m] for m in metrics]
    reg_values = [regression_results[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, clf_values, width, label='Classification', alpha=0.7)
    axes[1, 1].bar(x + width/2, reg_values, width, label='Regression', alpha=0.7)
    axes[1, 1].set_title('Performance Metrics Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'classification_vs_regression_{symbol.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main comparison execution."""
    print("🤖 CLASSIFICATION vs REGRESSION COMPARISON")
    print("=" * 60)
    
    # Configuration
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    horizon = 5
    
    source = YFinanceSource()
    all_results = {}
    
    for symbol in symbols:
        print(f"\n📊 Analyzing {symbol}...")
        
        try:
            # Load data
            data = source.get_daily_bars(symbol, start_date, end_date)
            if data.empty:
                print(f"❌ No data for {symbol}")
                continue
            
            data.columns = data.columns.str.lower()
            
            # Create features
            features = create_features(data)
            
            # Initialize strategies
            classification_strategy = ClassificationStrategy(threshold=0.01)
            regression_strategy = RegressionStrategy()
            
            # Backtest both strategies
            print(f"  🔄 Testing classification strategy...")
            clf_results = backtest_strategy(
                classification_strategy, features, data, symbol, horizon
            )
            
            print(f"  🔄 Testing regression strategy...")
            reg_results = backtest_strategy(
                regression_strategy, features, data, symbol, horizon
            )
            
            # Store results
            all_results[symbol] = {
                'classification': clf_results,
                'regression': reg_results
            }
            
            # Print individual results
            print(f"\n  📈 {symbol} Results:")
            print(f"    Classification - Return: {clf_results['total_return']:8.2%}, "
                  f"Sharpe: {clf_results['sharpe_ratio']:6.2f}, "
                  f"MaxDD: {clf_results['max_drawdown']:8.2%}")
            print(f"    Regression     - Return: {reg_results['total_return']:8.2%}, "
                  f"Sharpe: {reg_results['sharpe_ratio']:6.2f}, "
                  f"MaxDD: {reg_results['max_drawdown']:8.2%}")
            
            # Create comparison plot
            plot_comparison(clf_results, reg_results, symbol)
            print(f"  📊 Comparison plot saved: classification_vs_regression_{symbol.lower()}.png")
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
    
    # Overall summary
    if all_results:
        print(f"\n🏆 OVERALL COMPARISON SUMMARY:")
        print("=" * 60)
        print(f"{'Symbol':<8} {'Approach':<15} {'Return':<10} {'Sharpe':<8} {'MaxDD':<10} {'Trades':<8}")
        print("-" * 60)
        
        clf_total_return = 0
        reg_total_return = 0
        clf_sharpe_sum = 0
        reg_sharpe_sum = 0
        
        for symbol, results in all_results.items():
            clf = results['classification']
            reg = results['regression']
            
            print(f"{symbol:<8} {'Classification':<15} {clf['total_return']:8.2%} "
                  f"{clf['sharpe_ratio']:6.2f} {clf['max_drawdown']:8.2%} {clf['n_trades']:<8}")
            print(f"{symbol:<8} {'Regression':<15} {reg['total_return']:8.2%} "
                  f"{reg['sharpe_ratio']:6.2f} {reg['max_drawdown']:8.2%} {reg['n_trades']:<8}")
            print()
            
            clf_total_return += clf['total_return']
            reg_total_return += reg['total_return']
            clf_sharpe_sum += clf['sharpe_ratio']
            reg_sharpe_sum += reg['sharpe_ratio']
        
        # Average metrics
        n_symbols = len(all_results)
        print(f"📊 AVERAGE PERFORMANCE:")
        print(f"  Classification: Return={clf_total_return/n_symbols:8.2%}, "
              f"Avg Sharpe={clf_sharpe_sum/n_symbols:6.2f}")
        print(f"  Regression:     Return={reg_total_return/n_symbols:8.2%}, "
              f"Avg Sharpe={reg_sharpe_sum/n_symbols:6.2f}")
        
        # Winner summary
        improvements = {
            'return': (reg_total_return - clf_total_return) / abs(clf_total_return) * 100,
            'sharpe': (reg_sharpe_sum - clf_sharpe_sum) / n_symbols
        }
        
        print(f"\n🎯 KEY IMPROVEMENTS (Regression vs Classification):")
        print(f"  • Average Return Improvement: {improvements['return']:+.1f}%")
        print(f"  • Average Sharpe Improvement: {improvements['sharpe']:+.2f}")
        print(f"  • Better Position Sizing: Dynamic vs Fixed")
        print(f"  • Transaction Cost Awareness: Yes vs No")
        print(f"  • Risk Management: Adaptive vs Static")
    
    print(f"\n✅ Comparison analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())