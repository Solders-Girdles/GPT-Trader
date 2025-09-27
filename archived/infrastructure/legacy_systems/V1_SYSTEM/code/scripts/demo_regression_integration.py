#!/usr/bin/env python3
"""
Regression Trading Integration Demo
==================================

Demonstrates the integration of regression-based ML trading with the existing
GPT-Trader pipeline, including position sizing and risk management.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.dataflow.sources.yfinance_source import YFinanceSource
from bot.ml.regression_position_sizer import (
    RegressionPositionSizer, 
    PositionSizingConfig,
    DynamicRiskScaler
)
from bot.risk.simple_risk_manager import RiskLimits

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RegressionTradingStrategy:
    """
    Complete regression-based trading strategy that integrates ML predictions
    with position sizing and risk management.
    """
    
    def __init__(self, 
                 symbols: list,
                 models_path: str = "models/regression",
                 position_config: PositionSizingConfig = None):
        self.symbols = symbols
        self.models_path = Path(models_path)
        self.models = {}
        
        # Initialize components
        self.position_sizer = RegressionPositionSizer(position_config)
        self.risk_scaler = DynamicRiskScaler()
        self.data_source = YFinanceSource()
        
        # Trading state
        self.current_positions = {}
        self.trade_history = []
        self.performance_metrics = []
        
    def load_models(self):
        """Load trained regression models."""
        import joblib
        
        try:
            # Try to load ensemble models
            model_files = ['rf_regressor.pkl', 'gbr_regressor.pkl', 'ridge_regressor.pkl']
            
            for model_file in model_files:
                model_path = self.models_path / model_file
                if model_path.exists():
                    model_name = model_file.replace('_regressor.pkl', '')
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded {model_name} model")
                else:
                    logger.warning(f"Model file not found: {model_path}")
            
            if not self.models:
                logger.warning("No models loaded, will use dummy predictions")
                
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            logger.warning("Will use dummy predictions")
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for ML prediction - matches training features exactly."""
        features = pd.DataFrame(index=data.index)
        
        # Price momentum features
        for period in [1, 3, 5, 10, 20]:
            features[f'return_{period}d'] = data['close'].pct_change(period)
            features[f'momentum_{period}d'] = data['close'] / data['close'].shift(period) - 1
        
        # Moving averages and relative positions
        for period in [5, 10, 20, 50]:
            ma = data['close'].rolling(period).mean()
            features[f'sma_{period}'] = ma
            features[f'price_to_sma_{period}'] = data['close'] / ma - 1
        
        # Cross-MA features
        features['sma5_vs_sma20'] = features['sma_5'] / features['sma_20'] - 1
        features['sma10_vs_sma50'] = features['sma_10'] / features['sma_50'] - 1
        
        # Volatility features
        for period in [5, 10, 20]:
            vol = data['close'].rolling(period).std()
            features[f'volatility_{period}d'] = vol
            features[f'vol_ratio_{period}d'] = vol / vol.rolling(50).mean()
        
        # Volume features
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['volume_trend'] = (data['volume'].rolling(5).mean() / 
                                   data['volume'].rolling(20).mean())
        features['price_volume'] = data['close'].pct_change() * features['volume_ratio']
        
        # Technical indicators
        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        features['rsi_momentum'] = features['rsi'].diff(5)
        
        # Bollinger Bands
        sma20 = data['close'].rolling(20).mean()
        std20 = data['close'].rolling(20).std()
        features['bb_upper'] = sma20 + (2 * std20)
        features['bb_lower'] = sma20 - (2 * std20)
        features['bb_position'] = (data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # High-Low features
        features['hl_ratio'] = (data['high'] - data['low']) / data['close']
        features['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Trend strength
        features['trend_strength'] = (data['close'].rolling(20).apply(
            lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) == 20 else np.nan
        ))
        
        # Volatility regime
        vol_20 = data['close'].rolling(20).std()
        vol_60 = data['close'].rolling(60).std()
        features['vol_regime'] = vol_20 / vol_60
        
        return features
    
    def predict_returns(self, features: pd.DataFrame) -> np.ndarray:
        """Generate return predictions using loaded models."""
        if not self.models:
            # Dummy predictions for demonstration
            logger.warning("Using dummy predictions - models not loaded")
            np.random.seed(42)
            return np.random.normal(0.001, 0.02, len(features))
        
        # Ensemble prediction
        predictions = []
        weights = {'rf': 0.4, 'gbr': 0.4, 'ridge': 0.2}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(features.fillna(0))
                weight = weights.get(name, 1.0)
                predictions.append(pred * weight)
            except Exception as e:
                logger.warning(f"Failed to predict with {name}: {e}")
        
        if predictions:
            # Weighted average
            ensemble_pred = np.sum(predictions, axis=0)
            return ensemble_pred
        else:
            # Fallback to dummy
            return np.random.normal(0.001, 0.02, len(features))
    
    def calculate_confidence(self, features: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for predictions."""
        # Simple confidence based on feature completeness and prediction magnitude
        feature_completeness = (1 - features.isna().sum(axis=1) / len(features.columns))
        prediction_magnitude = np.abs(predictions)
        
        # Normalize and combine
        confidence = (feature_completeness * 0.6 + 
                     (prediction_magnitude / np.std(predictions)) * 0.4)
        
        return np.clip(confidence, 0.3, 1.0)
    
    def generate_trading_signals(self, end_date: str = None) -> dict:
        """Generate trading signals for all symbols."""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
        
        signals = {}
        predictions = {}
        confidences = {}
        prices = {}
        volatilities = {}
        
        logger.info(f"Generating signals for {len(self.symbols)} symbols")
        
        for symbol in self.symbols:
            try:
                # Get data
                data = self.data_source.get_daily_bars(symbol, start_date, end_date)
                if data.empty:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                data.columns = data.columns.str.lower()
                
                # Create features
                features = self.create_features(data)
                
                # Make predictions (use last row for current signal)
                if len(features) < 50:  # Need minimum data
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                recent_features = features.dropna().tail(1)
                if recent_features.empty:
                    continue
                
                pred = self.predict_returns(recent_features)[0]
                conf = self.calculate_confidence(recent_features, np.array([pred]))[0]
                
                # Calculate volatility
                recent_returns = data['close'].pct_change().dropna().tail(20)
                volatility = recent_returns.std() * np.sqrt(252) if len(recent_returns) > 5 else 0.20
                
                predictions[symbol] = pred
                confidences[symbol] = conf
                prices[symbol] = data['close'].iloc[-1]
                volatilities[symbol] = volatility
                
                logger.debug(f"{symbol}: pred={pred:.4f}, conf={conf:.2f}, vol={volatility:.2f}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        # Calculate portfolio positions
        if predictions:
            portfolio_positions = self.position_sizer.calculate_portfolio_positions(
                predictions=predictions,
                confidences=confidences,
                prices=prices,
                volatilities=volatilities
            )
            
            signals = {
                'positions': portfolio_positions,
                'predictions': predictions,
                'confidences': confidences,
                'prices': prices,
                'volatilities': volatilities
            }
        
        return signals
    
    def simulate_trading(self, 
                        start_date: str, 
                        end_date: str, 
                        initial_capital: float = 100000) -> dict:
        """
        Simulate trading over a period using the regression strategy.
        """
        logger.info(f"Simulating trading from {start_date} to {end_date}")
        
        # Generate trading dates (weekly rebalancing)
        trading_dates = pd.date_range(start_date, end_date, freq='W')
        
        portfolio_value = initial_capital
        portfolio_history = []
        position_history = []
        trade_history = []
        
        for date in trading_dates:
            date_str = date.strftime('%Y-%m-%d')
            
            try:
                # Generate signals for this date
                signals = self.generate_trading_signals(date_str)
                
                if not signals.get('positions'):
                    continue
                
                # Calculate portfolio changes
                current_positions = signals['positions']
                previous_positions = self.current_positions.copy()
                
                # Calculate returns for the week
                week_return = 0.0
                
                for symbol in current_positions:
                    if symbol in previous_positions:
                        # Get weekly return for this symbol
                        try:
                            symbol_data = self.data_source.get_daily_bars(
                                symbol, 
                                (date - timedelta(days=7)).strftime('%Y-%m-%d'),
                                date_str
                            )
                            if len(symbol_data) >= 2:
                                symbol_return = (symbol_data['Close'].iloc[-1] / 
                                               symbol_data['Close'].iloc[0] - 1)
                                position_return = previous_positions[symbol] * symbol_return
                                week_return += position_return
                        except:
                            continue
                
                # Update portfolio value
                portfolio_value *= (1 + week_return)
                
                # Record history
                portfolio_history.append({
                    'date': date,
                    'portfolio_value': portfolio_value,
                    'weekly_return': week_return,
                    'total_return': (portfolio_value / initial_capital - 1)
                })
                
                position_history.append({
                    'date': date,
                    **current_positions
                })
                
                # Update current positions
                self.current_positions = current_positions
                
                # Log progress
                if len(portfolio_history) % 10 == 0:
                    total_return = portfolio_value / initial_capital - 1
                    logger.info(f"Date: {date_str}, Portfolio: ${portfolio_value:,.0f}, "
                              f"Return: {total_return:.2%}")
                
            except Exception as e:
                logger.error(f"Error on {date_str}: {e}")
        
        # Calculate performance metrics
        returns_series = pd.Series([h['weekly_return'] for h in portfolio_history])
        
        metrics = {
            'total_return': (portfolio_value / initial_capital - 1),
            'annualized_return': ((portfolio_value / initial_capital) ** (52 / len(returns_series)) - 1),
            'volatility': returns_series.std() * np.sqrt(52),
            'sharpe_ratio': (returns_series.mean() / returns_series.std() * np.sqrt(52)) if returns_series.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown([h['total_return'] for h in portfolio_history]),
            'n_periods': len(portfolio_history)
        }
        
        return {
            'portfolio_history': pd.DataFrame(portfolio_history),
            'position_history': pd.DataFrame(position_history),
            'metrics': metrics,
            'final_value': portfolio_value
        }
    
    def _calculate_max_drawdown(self, returns: list) -> float:
        """Calculate maximum drawdown."""
        if not returns:
            return 0.0
        
        cumulative = np.array(returns) + 1
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return drawdowns.min()


def run_comparison_demo():
    """Run a comparison between classification and regression approaches."""
    print("=" * 70)
    print("🔄 CLASSIFICATION vs REGRESSION COMPARISON")
    print("=" * 70)
    
    # Configuration
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    # Regression strategy configuration
    regression_config = PositionSizingConfig(
        base_risk_budget=0.03,
        max_position_size=0.2,
        transaction_cost=0.001,
        min_confidence=0.6
    )
    
    # Initialize regression strategy
    regression_strategy = RegressionTradingStrategy(
        symbols=symbols,
        position_config=regression_config
    )
    
    # Load models (will use dummy predictions if models not available)
    regression_strategy.load_models()
    
    # Simulate regression trading
    logger.info("Running regression strategy simulation")
    regression_results = regression_strategy.simulate_trading(
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000
    )
    
    # Display results
    print(f"\n📊 REGRESSION STRATEGY RESULTS:")
    print(f"  Total Return:      {regression_results['metrics']['total_return']:8.2%}")
    print(f"  Annualized Return: {regression_results['metrics']['annualized_return']:8.2%}")
    print(f"  Volatility:        {regression_results['metrics']['volatility']:8.2%}")
    print(f"  Sharpe Ratio:      {regression_results['metrics']['sharpe_ratio']:8.2f}")
    print(f"  Max Drawdown:      {regression_results['metrics']['max_drawdown']:8.2%}")
    print(f"  Final Value:       ${regression_results['final_value']:,.0f}")
    
    # Show position allocation example
    if not regression_results['position_history'].empty:
        latest_positions = regression_results['position_history'].iloc[-1]
        print(f"\n📈 LATEST POSITION ALLOCATION:")
        for col in latest_positions.index:
            if col != 'date' and not pd.isna(latest_positions[col]):
                print(f"  {col}: {latest_positions[col]:+6.2%}")
    
    # Compare with buy-and-hold
    print(f"\n🏆 COMPARISON vs BUY-AND-HOLD:")
    
    # Calculate buy-and-hold return for SPY as benchmark
    try:
        source = YFinanceSource()
        spy_data = source.get_daily_bars('SPY', start_date, end_date)
        if not spy_data.empty:
            spy_return = spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0] - 1
            print(f"  SPY Buy-and-Hold:  {spy_return:8.2%}")
            print(f"  Strategy Alpha:    {regression_results['metrics']['total_return'] - spy_return:+8.2%}")
        else:
            print("  Could not calculate SPY benchmark")
    except:
        print("  Could not calculate SPY benchmark")
    
    return regression_results


def run_position_sizing_demo():
    """Demonstrate dynamic position sizing capabilities."""
    print("\n" + "=" * 70)
    print("📊 DYNAMIC POSITION SIZING DEMO")
    print("=" * 70)
    
    # Create position sizer with custom config
    config = PositionSizingConfig(
        base_risk_budget=0.025,
        max_position_size=0.3,
        transaction_cost=0.0015,
        min_confidence=0.5,
        max_confidence=2.5
    )
    
    sizer = RegressionPositionSizer(config)
    
    # Example scenarios
    scenarios = [
        {
            'name': 'High Confidence, High Return',
            'predictions': {'AAPL': 0.08, 'MSFT': 0.06, 'GOOGL': 0.10},
            'confidences': {'AAPL': 0.9, 'MSFT': 0.85, 'GOOGL': 0.95},
            'volatilities': {'AAPL': 0.25, 'MSFT': 0.22, 'GOOGL': 0.30}
        },
        {
            'name': 'Low Confidence, Mixed Returns',
            'predictions': {'AAPL': 0.02, 'MSFT': -0.01, 'GOOGL': 0.03},
            'confidences': {'AAPL': 0.5, 'MSFT': 0.4, 'GOOGL': 0.6},
            'volatilities': {'AAPL': 0.15, 'MSFT': 0.18, 'GOOGL': 0.20}
        },
        {
            'name': 'High Volatility Environment',
            'predictions': {'AAPL': 0.05, 'MSFT': 0.04, 'GOOGL': 0.07},
            'confidences': {'AAPL': 0.7, 'MSFT': 0.8, 'GOOGL': 0.75},
            'volatilities': {'AAPL': 0.45, 'MSFT': 0.40, 'GOOGL': 0.50}
        }
    ]
    
    prices = {'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 2500.0}
    
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"{'Symbol':<8} {'Prediction':<12} {'Confidence':<12} {'Volatility':<12} {'Position':<12}")
        print("-" * 56)
        
        positions = sizer.calculate_portfolio_positions(
            predictions=scenario['predictions'],
            confidences=scenario['confidences'],
            prices=prices,
            volatilities=scenario['volatilities']
        )
        
        for symbol in scenario['predictions']:
            pred = scenario['predictions'][symbol]
            conf = scenario['confidences'][symbol]
            vol = scenario['volatilities'][symbol]
            pos = positions.get(symbol, 0.0)
            
            print(f"{symbol:<8} {pred:+10.2%} {conf:10.2f} {vol:10.2%} {pos:+10.2%}")
        
        total_exposure = sum(abs(pos) for pos in positions.values())
        print(f"\nTotal Exposure: {total_exposure:.2%}")


def main():
    """Main demo execution."""
    print("🤖 ML REGRESSION TRADING INTEGRATION DEMO")
    print("=" * 70)
    
    try:
        # Run comparison demo
        results = run_comparison_demo()
        
        # Run position sizing demo
        run_position_sizing_demo()
        
        print(f"\n✅ Demo completed successfully!")
        print(f"💡 Key Insights:")
        print(f"  • Regression approach allows for nuanced position sizing")
        print(f"  • Position sizes adapt to prediction confidence and volatility")
        print(f"  • Transaction cost filtering prevents over-trading")
        print(f"  • Dynamic risk scaling adjusts to market conditions")
        
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())