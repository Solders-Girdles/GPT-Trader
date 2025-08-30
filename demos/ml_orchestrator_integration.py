#!/usr/bin/env python3
"""
ML Orchestrator Integration Demo
=================================
Demonstrates how to integrate ML strategies with the orchestrator
for a complete backtesting pipeline.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.integration.orchestrator import IntegratedOrchestrator
from bot.strategy.base import Strategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLStrategyForOrchestrator(Strategy):
    """ML strategy adapted for orchestrator integration."""
    
    def __init__(self, model_path: str = "models/simple_ml_model.pkl", **kwargs):
        super().__init__()
        self.name = "ML_Integrated"
        self.model = None
        
        # Try to load model
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Loaded ML model from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}")
            logger.info("Strategy will return neutral signals")
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create simple features from OHLCV data."""
        features = pd.DataFrame(index=data.index)
        
        # Ensure we have the required columns
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            logger.warning("Missing required OHLCV columns")
            return features
        
        # Price features
        features['returns_1d'] = data['close'].pct_change(1)
        features['returns_5d'] = data['close'].pct_change(5)
        features['returns_20d'] = data['close'].pct_change(20)
        
        # Moving averages
        features['sma_10'] = data['close'].rolling(10).mean()
        features['sma_20'] = data['close'].rolling(20).mean()
        features['sma_50'] = data['close'].rolling(50).mean()
        
        # Price relative to MA
        features['price_to_sma10'] = data['close'] / features['sma_10'] - 1
        features['price_to_sma20'] = data['close'] / features['sma_20'] - 1
        features['price_to_sma50'] = data['close'] / features['sma_50'] - 1
        
        # MA crosses
        features['sma10_vs_sma20'] = features['sma_10'] / features['sma_20'] - 1
        features['sma20_vs_sma50'] = features['sma_20'] / features['sma_50'] - 1
        
        # Volatility
        features['volatility_20d'] = data['close'].rolling(20).std()
        features['volatility_ratio'] = features['volatility_20d'] / features['volatility_20d'].rolling(50).mean()
        
        # Volume
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['volume_trend'] = data['volume'].rolling(5).mean() / data['volume'].rolling(20).mean()
        
        # Price range
        features['high_low_spread'] = (data['high'] - data['low']) / data['close']
        features['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # RSI
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
        """Generate trading signals compatible with orchestrator."""
        
        # Default neutral signals
        result = pd.DataFrame(index=data.index)
        result['signal'] = 0
        result['confidence'] = 0.0
        result['strategy'] = self.name
        
        if self.model is None:
            logger.warning("No ML model loaded, returning neutral signals")
            return result
        
        # Create features
        features = self.create_features(data)
        
        # Drop NaN values
        features_clean = features.dropna()
        
        if len(features_clean) == 0:
            logger.warning("No valid features after cleaning")
            return result
        
        try:
            # Make predictions
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_clean)[:, 1]
            else:
                probabilities = self.model.predict(features_clean)
            
            # Generate signals with improved thresholds
            # Use adaptive thresholds based on probability distribution
            prob_mean = probabilities.mean()
            prob_std = probabilities.std()
            
            # Dynamic thresholds
            buy_threshold = min(0.5, prob_mean + 0.5 * prob_std)
            sell_threshold = max(0.3, prob_mean - 0.5 * prob_std)
            
            signals = pd.Series(0, index=features_clean.index)
            signals[probabilities > buy_threshold] = 1
            signals[probabilities < sell_threshold] = -1
            
            # Calculate confidence
            confidence = pd.Series(np.abs(probabilities - prob_mean) / prob_std, index=features_clean.index)
            confidence = confidence.clip(0, 1)  # Normalize to [0, 1]
            
            # Update result
            result.loc[features_clean.index, 'signal'] = signals
            result.loc[features_clean.index, 'confidence'] = confidence
            
            # Log signal distribution
            buy_count = (signals == 1).sum()
            sell_count = (signals == -1).sum()
            logger.info(f"ML signals: {buy_count} buys, {sell_count} sells, {len(signals) - buy_count - sell_count} neutral")
            
        except Exception as e:
            logger.error(f"Error generating ML signals: {e}")
        
        return result


def run_integrated_ml_backtest():
    """Run backtest using ML strategy with orchestrator."""
    
    print("=" * 60)
    print("🤖 ML ORCHESTRATOR INTEGRATION")
    print("=" * 60)
    
    # Configuration
    symbols = ['AAPL', 'MSFT']
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)
    initial_capital = 100000
    
    print(f"\n📊 Configuration:")
    print(f"  Symbols: {symbols}")
    print(f"  Period: {start_date.date()} to {end_date.date()}")
    print(f"  Initial Capital: ${initial_capital:,.0f}")
    
    # Check if ML model exists
    model_path = Path("models/simple_ml_model.pkl")
    if not model_path.exists():
        print("\n❌ ML model not found!")
        print("💡 Running without ML model (neutral signals)")
    else:
        print(f"✅ ML model found: {model_path}")
    
    # Initialize orchestrator
    print("\n🔧 Initializing orchestrator...")
    orchestrator = IntegratedOrchestrator()
    
    # Create ML strategy
    ml_strategy = MLStrategyForOrchestrator()
    
    # Run backtest
    print("\n🚀 Running integrated backtest with ML strategy...")
    
    try:
        results = orchestrator.run_backtest(
            strategy=ml_strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        
        if results:
            print("\n✅ Backtest completed successfully!")
            
            # Display results
            print("\n📊 BACKTEST RESULTS:")
            print("=" * 60)
            
            # Performance metrics
            if hasattr(results, 'total_return'):
                print(f"  Total Return: {results.total_return:.2%}")
            if hasattr(results, 'sharpe_ratio'):
                print(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
            if hasattr(results, 'max_drawdown'):
                print(f"  Max Drawdown: {results.max_drawdown:.2%}")
            if hasattr(results, 'win_rate'):
                print(f"  Win Rate: {results.win_rate:.2%}")
            if hasattr(results, 'total_trades'):
                print(f"  Total Trades: {results.total_trades}")
            
            # Show sample trades if available
            if hasattr(results, 'trades') and len(results.trades) > 0:
                print(f"\n📈 Sample Trades (first 5):")
                for i, trade in enumerate(results.trades[:5], 1):
                    print(f"  {i}. {trade}")
        else:
            print("\n⚠️ No results returned from backtest")
            
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Compare with traditional strategy
    print("\n" + "=" * 60)
    print("📊 COMPARISON WITH TRADITIONAL STRATEGY")
    print("=" * 60)
    
    from bot.strategy.demo_ma import DemoMAStrategy
    ma_strategy = DemoMAStrategy(fast=10, slow=30)
    
    print("\n🚀 Running backtest with MA strategy...")
    
    try:
        ma_results = orchestrator.run_backtest(
            strategy=ma_strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        
        if ma_results:
            print("\n📊 MA Strategy Results:")
            if hasattr(ma_results, 'total_return'):
                print(f"  Total Return: {ma_results.total_return:.2%}")
            if hasattr(ma_results, 'sharpe_ratio'):
                print(f"  Sharpe Ratio: {ma_results.sharpe_ratio:.2f}")
            if hasattr(ma_results, 'max_drawdown'):
                print(f"  Max Drawdown: {ma_results.max_drawdown:.2%}")
                
            # Compare returns
            if results and hasattr(results, 'total_return') and hasattr(ma_results, 'total_return'):
                print("\n🏆 WINNER:")
                if results.total_return > ma_results.total_return:
                    print(f"  ML Strategy wins! ({results.total_return:.2%} vs {ma_results.total_return:.2%})")
                else:
                    print(f"  MA Strategy wins! ({ma_results.total_return:.2%} vs {results.total_return:.2%})")
                    
    except Exception as e:
        print(f"\n❌ MA backtest failed: {e}")
    
    print("\n✅ Integration demo complete!")


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("ML STRATEGY ORCHESTRATOR INTEGRATION")
    print("=" * 60)
    print("\nThis demo shows how ML strategies integrate with")
    print("the existing orchestrator infrastructure.")
    
    run_integrated_ml_backtest()
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("=" * 60)
    print("1. ML models can be seamlessly integrated as strategies")
    print("2. The orchestrator handles data, allocation, and risk")
    print("3. ML strategies can be compared with traditional ones")
    print("4. The same infrastructure supports both approaches")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())