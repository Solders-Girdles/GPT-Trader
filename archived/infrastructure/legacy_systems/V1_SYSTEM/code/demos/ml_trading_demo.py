#!/usr/bin/env python3
"""
ML Trading Demo
===============
Demonstrates trading using a trained ML model for signal generation.
Compares ML-based trading with traditional strategies.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.dataflow.sources.yfinance_source import YFinanceSource
from bot.integration.orchestrator import IntegratedOrchestrator
from bot.strategy.demo_ma import DemoMAStrategy
from bot.strategy.base import Strategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleMLStrategy(Strategy):
    """Simple ML-based strategy using pre-trained model."""
    
    def __init__(self, model_path: str = "models/simple_ml_model.pkl"):
        super().__init__()
        self.name = "ML Strategy"
        self.model = None
        self.load_model(model_path)
        
    def load_model(self, model_path: str):
        """Load pre-trained model."""
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Loaded ML model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create the same features used in training."""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns_1d'] = data['close'].pct_change(1)
        features['returns_5d'] = data['close'].pct_change(5)
        features['returns_20d'] = data['close'].pct_change(20)
        
        # Moving averages
        features['sma_10'] = data['close'].rolling(10).mean()
        features['sma_20'] = data['close'].rolling(20).mean()
        features['sma_50'] = data['close'].rolling(50).mean()
        
        # Price relative to moving averages
        features['price_to_sma10'] = data['close'] / features['sma_10'] - 1
        features['price_to_sma20'] = data['close'] / features['sma_20'] - 1
        features['price_to_sma50'] = data['close'] / features['sma_50'] - 1
        
        # Moving average crosses
        features['sma10_vs_sma20'] = features['sma_10'] / features['sma_20'] - 1
        features['sma20_vs_sma50'] = features['sma_20'] / features['sma_50'] - 1
        
        # Volatility
        features['volatility_20d'] = data['close'].rolling(20).std()
        features['volatility_ratio'] = features['volatility_20d'] / features['volatility_20d'].rolling(50).mean()
        
        # Volume features
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['volume_trend'] = data['volume'].rolling(5).mean() / data['volume'].rolling(20).mean()
        
        # High-Low spread
        features['high_low_spread'] = (data['high'] - data['low']) / data['close']
        features['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Simple RSI calculation
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Momentum
        features['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
        features['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
        
        return features
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using ML model."""
        if self.model is None:
            raise ValueError("No model loaded")
        
        # Create features
        features = self.create_features(data)
        
        # Drop NaN values
        features_clean = features.dropna()
        
        if len(features_clean) == 0:
            # Return neutral signals if no valid features
            return pd.DataFrame({
                'signal': pd.Series(0, index=data.index),
                'confidence': pd.Series(0.0, index=data.index)
            })
        
        # Make predictions
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_clean)[:, 1]
        else:
            probabilities = self.model.predict(features_clean)
        
        # Generate signals based on probability thresholds
        signals = pd.Series(0, index=features_clean.index)
        
        # Use more balanced thresholds to avoid all sell signals
        buy_threshold = 0.45  # Buy when probability > 45%
        sell_threshold = 0.35  # Sell when probability < 35%
        
        signals[probabilities > buy_threshold] = 1  # Buy
        signals[probabilities < sell_threshold] = -1  # Sell
        
        # Create confidence scores
        confidence = pd.Series(np.abs(probabilities - 0.5) * 2, index=features_clean.index)
        
        # Reindex to match original data
        result = pd.DataFrame(index=data.index)
        result['signal'] = signals.reindex(data.index, fill_value=0)
        result['confidence'] = confidence.reindex(data.index, fill_value=0)
        result['probability'] = pd.Series(probabilities, index=features_clean.index).reindex(data.index, fill_value=0.5)
        
        return result


def run_ml_backtest(symbol: str, start_date: str, end_date: str):
    """Run backtest using ML strategy."""
    print(f"\n{'='*60}")
    print(f"🤖 ML STRATEGY BACKTEST - {symbol}")
    print('='*60)
    
    # Load data
    source = YFinanceSource()
    data = source.get_daily_bars(symbol, start_date, end_date)
    
    if data.empty:
        print(f"❌ No data available for {symbol}")
        return None
    
    # Convert columns to lowercase
    data.columns = data.columns.str.lower()
    
    print(f"📊 Loaded {len(data)} trading days")
    
    # Initialize ML strategy
    try:
        ml_strategy = SimpleMLStrategy()
    except Exception as e:
        print(f"❌ Failed to load ML model: {e}")
        print("💡 Please run scripts/train_ml_simple.py first to train the model")
        return None
    
    # Generate signals
    signals_df = ml_strategy.generate_signals(data)
    
    # Count signal distribution
    buy_signals = (signals_df['signal'] == 1).sum()
    sell_signals = (signals_df['signal'] == -1).sum()
    neutral_signals = (signals_df['signal'] == 0).sum()
    
    print(f"\n📈 Signal Distribution:")
    print(f"  Buy signals:  {buy_signals:4} ({buy_signals/len(data)*100:.1f}%)")
    print(f"  Sell signals: {sell_signals:4} ({sell_signals/len(data)*100:.1f}%)")
    print(f"  Neutral:      {neutral_signals:4} ({neutral_signals/len(data)*100:.1f}%)")
    
    # Calculate simple returns (without transaction costs)
    returns = data['close'].pct_change()
    strategy_returns = returns * signals_df['signal'].shift(1)  # Lag signals by 1 day
    
    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    
    # Calculate buy-and-hold returns
    buy_hold_return = (data['close'].iloc[-1] / data['close'].iloc[0]) - 1
    
    # Calculate Sharpe ratio (simplified)
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
    
    # Calculate max drawdown
    cumulative = (1 + strategy_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Print results
    print(f"\n📊 Performance Metrics:")
    print(f"  Strategy Return:     {total_return:+.2%}")
    print(f"  Buy & Hold Return:   {buy_hold_return:+.2%}")
    print(f"  Outperformance:      {(total_return - buy_hold_return)*100:+.1f}%")
    print(f"  Sharpe Ratio:        {sharpe_ratio:.2f}")
    print(f"  Max Drawdown:        {max_drawdown:.2%}")
    
    # Show sample predictions
    print(f"\n🔮 Sample ML Predictions (last 10 days):")
    last_10 = signals_df.tail(10)
    for date, row in last_10.iterrows():
        signal_str = "BUY " if row['signal'] == 1 else "SELL" if row['signal'] == -1 else "HOLD"
        print(f"  {date.strftime('%Y-%m-%d')}: {signal_str} (prob={row['probability']:.3f}, conf={row['confidence']:.3f})")
    
    return {
        'strategy_return': total_return,
        'buy_hold_return': buy_hold_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'signals': signals_df
    }


def compare_with_traditional_strategy(symbol: str, start_date: str, end_date: str):
    """Compare ML strategy with traditional moving average strategy."""
    print(f"\n{'='*60}")
    print(f"📊 STRATEGY COMPARISON - {symbol}")
    print('='*60)
    
    # Load data
    source = YFinanceSource()
    data = source.get_daily_bars(symbol, start_date, end_date)
    
    if data.empty:
        print(f"❌ No data available for {symbol}")
        return
    
    # Convert columns to lowercase
    data.columns = data.columns.str.lower()
    
    # Run ML strategy
    print("\n1️⃣ ML Strategy:")
    print("-" * 40)
    
    try:
        ml_strategy = SimpleMLStrategy()
        ml_signals = ml_strategy.generate_signals(data)
        
        # Calculate ML returns
        returns = data['close'].pct_change()
        ml_returns = returns * ml_signals['signal'].shift(1)
        ml_cumulative = (1 + ml_returns).cumprod()
        ml_total_return = ml_cumulative.iloc[-1] - 1
        
        print(f"  Total Return: {ml_total_return:+.2%}")
        print(f"  Trades: {(ml_signals['signal'].diff() != 0).sum()}")
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        ml_total_return = 0
    
    # Run traditional MA strategy
    print("\n2️⃣ Moving Average Strategy:")
    print("-" * 40)
    
    ma_strategy = DemoMAStrategy(fast=10, slow=30)
    ma_signals = ma_strategy.generate_signals(data)
    
    # Calculate MA returns
    ma_returns = returns * ma_signals['signal'].shift(1)
    ma_cumulative = (1 + ma_returns).cumprod()
    ma_total_return = ma_cumulative.iloc[-1] - 1
    
    print(f"  Total Return: {ma_total_return:+.2%}")
    print(f"  Trades: {(ma_signals['signal'].diff() != 0).sum()}")
    
    # Buy and hold benchmark
    print("\n3️⃣ Buy & Hold Benchmark:")
    print("-" * 40)
    
    buy_hold_return = (data['close'].iloc[-1] / data['close'].iloc[0]) - 1
    print(f"  Total Return: {buy_hold_return:+.2%}")
    
    # Summary
    print("\n📈 PERFORMANCE RANKING:")
    print("=" * 40)
    
    strategies = [
        ("ML Strategy", ml_total_return),
        ("MA Strategy", ma_total_return),
        ("Buy & Hold", buy_hold_return)
    ]
    strategies.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, ret) in enumerate(strategies, 1):
        print(f"{i}. {name:15} {ret:+.2%}")
    
    # Winner
    winner = strategies[0][0]
    print(f"\n🏆 Winner: {winner}!")


def main():
    print("=" * 60)
    print("🤖 ML TRADING DEMONSTRATION")
    print("=" * 60)
    
    # Configuration
    SYMBOL = 'AAPL'
    TRAIN_END = '2023-12-31'
    TEST_START = '2024-01-01'
    TEST_END = '2024-12-31'
    
    print(f"\nConfiguration:")
    print(f"  Symbol: {SYMBOL}")
    print(f"  Test Period: {TEST_START} to {TEST_END}")
    print(f"  Model: models/simple_ml_model.pkl")
    
    # Check if model exists
    model_path = Path("models/simple_ml_model.pkl")
    if not model_path.exists():
        print("\n❌ ML model not found!")
        print("💡 Please run scripts/train_ml_simple.py first to train the model")
        return 1
    
    # Run ML backtest on test data (out-of-sample)
    print(f"\n🧪 Testing on out-of-sample data ({TEST_START} to {TEST_END}):")
    ml_results = run_ml_backtest(SYMBOL, TEST_START, TEST_END)
    
    if ml_results:
        # Compare with traditional strategies
        compare_with_traditional_strategy(SYMBOL, TEST_START, TEST_END)
        
        # Performance summary
        print(f"\n{'='*60}")
        print("📊 ML MODEL PERFORMANCE SUMMARY")
        print('='*60)
        
        if ml_results['strategy_return'] > ml_results['buy_hold_return']:
            print("✅ ML strategy OUTPERFORMED buy-and-hold!")
            print(f"   Excess return: {(ml_results['strategy_return'] - ml_results['buy_hold_return'])*100:+.1f}%")
        else:
            print("❌ ML strategy underperformed buy-and-hold")
            print(f"   Underperformance: {(ml_results['strategy_return'] - ml_results['buy_hold_return'])*100:.1f}%")
        
        print(f"\n📈 Key Metrics:")
        print(f"   Sharpe Ratio: {ml_results['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {ml_results['max_drawdown']:.2%}")
    
    print("\n✅ Demo complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())