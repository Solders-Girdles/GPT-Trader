"""
Confidence Filter Integration Demo
Shows how to integrate the ML confidence filter with existing trading strategies
and demonstrate the expected improvements in performance.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# Add source to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import our components
from scripts.ml_confidence_filter import MLConfidenceFilter, ConfidenceConfig
from bot.strategy.demo_ma import MovingAverageStrategy
from bot.strategy.trend_breakout import TrendBreakoutStrategy

try:
    from bot.ml.integrated_pipeline import IntegratedMLPipeline
    from bot.ml.ensemble_manager import EnsembleManager
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML components not available, using simplified demo")


class ConfidenceEnhancedStrategy:
    """
    Enhanced trading strategy with ML confidence filtering
    """
    
    def __init__(self, base_strategy, confidence_filter, models=None):
        """
        Initialize confidence-enhanced strategy
        
        Args:
            base_strategy: Base trading strategy
            confidence_filter: ML confidence filter
            models: Dictionary of ML models (optional)
        """
        self.base_strategy = base_strategy
        self.confidence_filter = confidence_filter
        self.models = models or {}
        
        # Performance tracking
        self.original_signals = []
        self.filtered_signals = []
        self.confidence_scores = []
        self.trade_returns = []
        
        self.original_performance = {
            'trades': 0, 'wins': 0, 'total_return': 0.0, 'fees': 0.0
        }
        self.filtered_performance = {
            'trades': 0, 'wins': 0, 'total_return': 0.0, 'fees': 0.0
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals with confidence filtering
        
        Args:
            data: Market data
            
        Returns:
            DataFrame with original and filtered signals
        """
        # Get base strategy signals
        try:
            original_signals = self.base_strategy.generate_signals(data)
        except:
            # Fallback for strategies without this method
            original_signals = self._generate_ma_signals(data)
        
        results = pd.DataFrame(index=data.index)
        results['original_signal'] = original_signals
        results['confidence'] = 0.5  # Default confidence
        results['filtered_signal'] = 0
        results['should_trade'] = False
        
        # Apply confidence filtering if models available
        if self.models and len(self.models) > 0:
            features = self._create_features(data)
            
            for i in range(len(data)):
                if i < 50:  # Need sufficient history
                    continue
                    
                # Get recent market data for regime analysis
                recent_data = data.iloc[max(0, i-50):i+1]
                
                # Calculate confidence
                try:
                    confidence_metrics = self.confidence_filter.calculate_prediction_confidence(
                        self.models,
                        features.iloc[i:i+1].values,
                        recent_data,
                        model_type="trend_following"
                    )
                    
                    results.iloc[i, results.columns.get_loc('confidence')] = confidence_metrics.overall_confidence
                    results.iloc[i, results.columns.get_loc('should_trade')] = confidence_metrics.should_trade
                    
                    if confidence_metrics.should_trade:
                        # Apply position sizing
                        signal_strength = confidence_metrics.position_size_multiplier
                        results.iloc[i, results.columns.get_loc('filtered_signal')] = (
                            original_signals.iloc[i] * signal_strength
                        )
                    
                except Exception as e:
                    # Fallback to original signal with reduced strength
                    results.iloc[i, results.columns.get_loc('confidence')] = 0.3
                    results.iloc[i, results.columns.get_loc('filtered_signal')] = original_signals.iloc[i] * 0.5
        else:
            # Simple confidence-based filtering without ML models
            # Use technical indicators as proxy for confidence
            confidence_proxy = self._calculate_technical_confidence(data)
            results['confidence'] = confidence_proxy
            
            # Apply threshold
            high_conf_mask = confidence_proxy >= self.confidence_filter.config.base_confidence_threshold
            results['should_trade'] = high_conf_mask
            results.loc[high_conf_mask, 'filtered_signal'] = original_signals[high_conf_mask]
        
        return results
    
    def _generate_ma_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate simple moving average signals as fallback"""
        close = data['close'] if 'close' in data.columns else data['Close']
        
        # Simple moving average crossover
        sma_fast = close.rolling(20).mean()
        sma_slow = close.rolling(50).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[sma_fast > sma_slow] = 1
        signals[sma_fast < sma_slow] = -1
        
        # Only signal on crossovers
        signal_changes = signals.diff()
        signals[signal_changes == 0] = 0
        
        return signals
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for ML models"""
        close = data['close'] if 'close' in data.columns else data['Close']
        volume = data['volume'] if 'volume' in data.columns else data.get('Volume', pd.Series(1, index=data.index))
        
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = close.pct_change()
        features['sma_20'] = close.rolling(20).mean()
        features['sma_50'] = close.rolling(50).mean()
        features['price_to_sma20'] = close / features['sma_20']
        features['price_to_sma50'] = close / features['sma_50']
        
        # Volatility features
        features['volatility'] = features['returns'].rolling(20).std()
        features['atr'] = self._calculate_atr(data)
        
        # Volume features
        features['volume_ma'] = volume.rolling(20).mean()
        features['volume_ratio'] = volume / features['volume_ma']
        
        # Momentum features
        features['rsi'] = self._calculate_rsi(close)
        features['momentum_5'] = close / close.shift(5) - 1
        features['momentum_20'] = close / close.shift(20) - 1
        
        return features.fillna(0)
    
    def _calculate_technical_confidence(self, data: pd.DataFrame) -> pd.Series:
        """Calculate confidence based on technical indicators"""
        close = data['close'] if 'close' in data.columns else data['Close']
        
        # RSI confidence (higher at extremes)
        rsi = self._calculate_rsi(close)
        rsi_conf = np.where(rsi < 30, (30 - rsi) / 30, 
                           np.where(rsi > 70, (rsi - 70) / 30, 0))
        
        # Volatility confidence (higher in low vol)
        returns = close.pct_change()
        vol = returns.rolling(20).std()
        vol_conf = 1 / (1 + vol * 100)  # Normalize
        
        # Trend strength confidence
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        trend_strength = abs((close - sma_50) / sma_50)
        trend_conf = np.clip(trend_strength * 10, 0, 1)
        
        # Combined confidence
        confidence = (rsi_conf * 0.3 + vol_conf * 0.4 + trend_conf * 0.3)
        
        return pd.Series(confidence, index=data.index).fillna(0.5)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high'] if 'high' in data.columns else data.get('High', data['close'])
        low = data['low'] if 'low' in data.columns else data.get('Low', data['close'])
        close = data['close'] if 'close' in data.columns else data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def backtest_comparison(self, data: pd.DataFrame, initial_capital: float = 100000) -> dict:
        """
        Compare performance with and without confidence filtering
        
        Args:
            data: Historical market data
            initial_capital: Starting capital
            
        Returns:
            Dictionary with performance comparison
        """
        # Generate signals
        signals_df = self.generate_signals(data)
        
        # Simulate trading
        original_results = self._simulate_trading(
            data, signals_df['original_signal'], initial_capital
        )
        
        filtered_results = self._simulate_trading(
            data, signals_df['filtered_signal'], initial_capital
        )
        
        # Calculate metrics
        comparison = {
            'original': self._calculate_metrics(original_results, data.index),
            'filtered': self._calculate_metrics(filtered_results, data.index),
            'improvement': {}
        }
        
        # Calculate improvements
        for key in ['total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown']:
            if key in comparison['original'] and key in comparison['filtered']:
                if key == 'max_drawdown':
                    # Lower drawdown is better
                    improvement = (comparison['original'][key] - comparison['filtered'][key]) / comparison['original'][key]
                else:
                    # Higher is better
                    improvement = (comparison['filtered'][key] - comparison['original'][key]) / comparison['original'][key]
                comparison['improvement'][key] = improvement
        
        # Add confidence statistics
        comparison['confidence_stats'] = {
            'mean_confidence': signals_df['confidence'].mean(),
            'high_confidence_trades': (signals_df['confidence'] > 0.7).sum(),
            'total_original_signals': (signals_df['original_signal'] != 0).sum(),
            'total_filtered_signals': (signals_df['filtered_signal'] != 0).sum(),
            'signal_reduction': 1 - (signals_df['filtered_signal'] != 0).sum() / (signals_df['original_signal'] != 0).sum()
        }
        
        return comparison
    
    def _simulate_trading(
        self, 
        data: pd.DataFrame, 
        signals: pd.Series, 
        initial_capital: float,
        transaction_cost: float = 0.002
    ) -> pd.DataFrame:
        """Simulate trading with given signals"""
        close = data['close'] if 'close' in data.columns else data['Close']
        
        results = pd.DataFrame(index=data.index)
        results['signal'] = signals
        results['price'] = close
        results['position'] = 0
        results['returns'] = 0
        results['equity'] = initial_capital
        results['drawdown'] = 0
        
        current_position = 0
        cash = initial_capital
        shares = 0
        peak_equity = initial_capital
        
        for i in range(1, len(data)):
            signal = signals.iloc[i]
            price = close.iloc[i]
            prev_price = close.iloc[i-1]
            
            # Handle position changes
            if signal != 0 and signal != current_position:
                # Close existing position
                if current_position != 0:
                    cash = shares * price * (1 - transaction_cost)
                    shares = 0
                    
                # Open new position
                if signal != 0:
                    position_value = cash * 0.95  # Use 95% of cash
                    shares = position_value / price
                    cash = 0
                    
                current_position = signal
            
            # Calculate returns
            if shares > 0:
                equity = shares * price + cash
            else:
                equity = cash
                
            daily_return = (equity - results['equity'].iloc[i-1]) / results['equity'].iloc[i-1]
            
            results['position'].iloc[i] = current_position
            results['returns'].iloc[i] = daily_return
            results['equity'].iloc[i] = equity
            
            # Track drawdown
            if equity > peak_equity:
                peak_equity = equity
            drawdown = (peak_equity - equity) / peak_equity
            results['drawdown'].iloc[i] = drawdown
        
        return results
    
    def _calculate_metrics(self, results: pd.DataFrame, index: pd.Index) -> dict:
        """Calculate performance metrics"""
        equity = results['equity']
        returns = results['returns']
        positions = results['position']
        
        # Basic metrics
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        
        # Sharpe ratio
        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
            
        # Win rate
        trade_returns = []
        current_trade_return = 0
        for i in range(1, len(positions)):
            if positions.iloc[i] != 0:
                current_trade_return += returns.iloc[i]
            elif positions.iloc[i-1] != 0:  # Position closed
                trade_returns.append(current_trade_return)
                current_trade_return = 0
                
        win_rate = np.mean([r > 0 for r in trade_returns]) if trade_returns else 0
        
        # Max drawdown
        max_drawdown = results['drawdown'].max()
        
        # Trade frequency
        trades = (positions.diff() != 0).sum()
        days = len(index)
        trades_per_year = trades / days * 252
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'trades_per_year': trades_per_year,
            'total_trades': trades,
            'avg_trade_return': np.mean(trade_returns) if trade_returns else 0
        }


def create_demo_models():
    """Create simple demo models for testing"""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    
    models = {
        'random_forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'gradient_boost': GradientBoostingClassifier(n_estimators=50, random_state=42)
    }
    
    return models


def demonstrate_confidence_filtering():
    """Run demonstration of confidence filtering improvements"""
    print("=" * 60)
    print("ML CONFIDENCE FILTERING DEMONSTRATION")
    print("=" * 60)
    
    # Get sample data
    try:
        import yfinance as yf
        print("\n1. Loading market data...")
        ticker = yf.Ticker("SPY")
        data = ticker.history(period="2y")
        data.columns = [c.lower() for c in data.columns]
        print(f"   Loaded {len(data)} days of SPY data")
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create synthetic data for demo
        print("   Creating synthetic data for demo...")
        dates = pd.date_range('2022-01-01', '2024-01-01', freq='D')
        np.random.seed(42)
        price = 100
        prices = []
        for _ in range(len(dates)):
            price *= (1 + np.random.normal(0.0005, 0.02))
            prices.append(price)
        
        data = pd.DataFrame({
            'close': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    
    # Create base strategy
    print("\n2. Initializing trading strategy...")
    base_strategy = MovingAverageStrategy() if 'MovingAverageStrategy' in globals() else None
    
    # Create confidence filter
    print("   Setting up confidence filter...")
    config = ConfidenceConfig(
        base_confidence_threshold=0.65,
        target_trades_per_year=40,
        enable_regime_confidence=True,
        enable_performance_confidence=True,
        transaction_cost=0.002
    )
    confidence_filter = MLConfidenceFilter(config)
    
    # Create and train simple models
    print("   Training ML models...")
    models = create_demo_models()
    
    # Create features for training
    enhanced_strategy = ConfidenceEnhancedStrategy(base_strategy, confidence_filter, models)
    features = enhanced_strategy._create_features(data)
    
    # Create target (next day return direction)
    target = (data['close'].shift(-1) > data['close']).astype(int)
    
    # Train models on first 80% of data
    split_idx = int(len(data) * 0.8)
    X_train = features.iloc[:split_idx].fillna(0)
    y_train = target.iloc[:split_idx].fillna(0)
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            score = model.score(X_train, y_train)
            print(f"   {name} training accuracy: {score:.3f}")
        except Exception as e:
            print(f"   Warning: {name} training failed: {e}")
    
    # Run backtest comparison
    print("\n3. Running backtest comparison...")
    test_data = data.iloc[split_idx:]
    
    if len(test_data) > 50:
        comparison = enhanced_strategy.backtest_comparison(test_data)
        
        print("\n" + "=" * 60)
        print("PERFORMANCE COMPARISON RESULTS")
        print("=" * 60)
        
        # Original strategy performance
        orig = comparison['original']
        print(f"\nORIGINAL STRATEGY:")
        print(f"  Total Return:     {orig['total_return']*100:8.2f}%")
        print(f"  Sharpe Ratio:     {orig['sharpe_ratio']:8.2f}")
        print(f"  Win Rate:         {orig['win_rate']*100:8.1f}%")
        print(f"  Max Drawdown:     {orig['max_drawdown']*100:8.2f}%")
        print(f"  Trades/Year:      {orig['trades_per_year']:8.1f}")
        print(f"  Total Trades:     {orig['total_trades']:8.0f}")
        
        # Filtered strategy performance
        filt = comparison['filtered']
        print(f"\nCONFIDENCE FILTERED STRATEGY:")
        print(f"  Total Return:     {filt['total_return']*100:8.2f}%")
        print(f"  Sharpe Ratio:     {filt['sharpe_ratio']:8.2f}")
        print(f"  Win Rate:         {filt['win_rate']*100:8.1f}%")
        print(f"  Max Drawdown:     {filt['max_drawdown']*100:8.2f}%")
        print(f"  Trades/Year:      {filt['trades_per_year']:8.1f}")
        print(f"  Total Trades:     {filt['total_trades']:8.0f}")
        
        # Improvements
        imp = comparison['improvement']
        print(f"\nIMPROVEMENTS:")
        print(f"  Return Improvement:     {imp.get('total_return', 0)*100:8.2f}%")
        print(f"  Sharpe Improvement:     {imp.get('sharpe_ratio', 0)*100:8.2f}%")
        print(f"  Win Rate Improvement:   {imp.get('win_rate', 0)*100:8.2f}%")
        print(f"  Drawdown Reduction:     {imp.get('max_drawdown', 0)*100:8.2f}%")
        
        # Confidence statistics
        conf = comparison['confidence_stats']
        print(f"\nCONFIDENCE STATISTICS:")
        print(f"  Mean Confidence:        {conf['mean_confidence']:8.3f}")
        print(f"  High Confidence Trades: {conf['high_confidence_trades']:8.0f}")
        print(f"  Signal Reduction:       {conf['signal_reduction']*100:8.1f}%")
        
        # Expected vs Actual improvements
        print(f"\n" + "=" * 60)
        print("TARGET vs ACTUAL IMPROVEMENTS")
        print("=" * 60)
        
        targets = {
            'trade_reduction': 70,  # Target 70% reduction
            'win_rate_improvement': 96,  # Target: 28% → 55% = 96% improvement
            'sharpe_improvement': 50,  # Target: +0.5 Sharpe
        }
        
        actual_trade_reduction = conf['signal_reduction'] * 100
        actual_win_rate_improvement = imp.get('win_rate', 0) * 100
        actual_sharpe_improvement = imp.get('sharpe_ratio', 0) * 100
        
        print(f"Trade Frequency Reduction:")
        print(f"  Target: {targets['trade_reduction']:>6.1f}%   Actual: {actual_trade_reduction:>6.1f}%")
        
        print(f"Win Rate Improvement:")
        print(f"  Target: {targets['win_rate_improvement']:>6.1f}%   Actual: {actual_win_rate_improvement:>6.1f}%")
        
        print(f"Sharpe Ratio Improvement:")
        print(f"  Target: {targets['sharpe_improvement']:>6.1f}%   Actual: {actual_sharpe_improvement:>6.1f}%")
        
        # Success assessment
        print(f"\n" + "=" * 60)
        print("SUCCESS ASSESSMENT")
        print("=" * 60)
        
        success_criteria = 0
        if actual_trade_reduction >= 50:  # At least 50% reduction
            print("✓ Trade frequency reduction: SUCCESS")
            success_criteria += 1
        else:
            print("✗ Trade frequency reduction: NEEDS IMPROVEMENT")
            
        if actual_win_rate_improvement >= 20:  # At least 20% improvement
            print("✓ Win rate improvement: SUCCESS")
            success_criteria += 1
        else:
            print("✗ Win rate improvement: NEEDS IMPROVEMENT")
            
        if filt['sharpe_ratio'] > orig['sharpe_ratio']:
            print("✓ Sharpe ratio improvement: SUCCESS")
            success_criteria += 1
        else:
            print("✗ Sharpe ratio improvement: NEEDS IMPROVEMENT")
            
        if filt['max_drawdown'] <= orig['max_drawdown']:
            print("✓ Drawdown control: SUCCESS")
            success_criteria += 1
        else:
            print("✗ Drawdown control: NEEDS IMPROVEMENT")
            
        print(f"\nOverall Success Rate: {success_criteria}/4 criteria met")
        
        if success_criteria >= 3:
            print("🎯 CONFIDENCE FILTERING: SUCCESSFUL IMPLEMENTATION")
        elif success_criteria >= 2:
            print("⚠️  CONFIDENCE FILTERING: PARTIAL SUCCESS - NEEDS TUNING")
        else:
            print("❌ CONFIDENCE FILTERING: REQUIRES SIGNIFICANT IMPROVEMENT")
    
    else:
        print("   Warning: Insufficient test data for comprehensive backtest")
    
    print(f"\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_confidence_filtering()