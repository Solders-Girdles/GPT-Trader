#!/usr/bin/env python3
"""
ML Regression Trading Implementation
===================================
Switch from classification to regression for better position sizing.

Predicts actual returns and sizes positions accordingly rather than
just binary buy/sell decisions.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import joblib
import matplotlib.pyplot as plt
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.dataflow.sources.yfinance_source import YFinanceSource

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_return_predictor(features: pd.DataFrame, prices: pd.DataFrame, horizon: int = 5):
    """
    Train model to predict return MAGNITUDE, not just direction
    
    Args:
        features: Feature matrix
        prices: Price data with 'close' column
        horizon: Days ahead to predict
        
    Returns:
        Dict of trained models
    """
    logger.info(f"Training return predictor with {horizon}-day horizon")
    
    # Calculate actual future returns (what we want to predict)
    future_returns = prices['close'].pct_change(horizon).shift(-horizon)
    
    # Remove NaN values
    valid_idx = ~(features.isna().any(axis=1) | future_returns.isna())
    X = features[valid_idx].copy()
    y = future_returns[valid_idx].copy()
    
    logger.info(f"Valid samples: {len(X)} (removed {(~valid_idx).sum()} NaN samples)")
    logger.info(f"Target return stats: mean={y.mean():.4f}, std={y.std():.4f}")
    
    # Train ensemble of regressors
    models = {
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
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Train all models
    model_scores = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Validate
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        
        model_scores[name] = {'r2': r2, 'mse': mse, 'mae': mae}
        logger.info(f"{name} - R²: {r2:.4f}, MSE: {mse:.6f}, MAE: {mae:.4f}")
    
    return models, model_scores, X_train, X_val, y_train, y_val


def predict_returns_ensemble(models: dict, features: pd.DataFrame, weights: dict = None):
    """
    Ensemble prediction for robustness
    
    Args:
        models: Dictionary of trained models
        features: Feature matrix
        weights: Model weights (optional)
        
    Returns:
        Array of predicted returns
    """
    if weights is None:
        weights = {name: 1.0 for name in models.keys()}
    
    predictions = []
    total_weight = 0
    
    for name, model in models.items():
        weight = weights.get(name, 1.0)
        pred = model.predict(features) * weight
        predictions.append(pred)
        total_weight += weight
    
    # Weighted average predictions
    ensemble_pred = np.sum(predictions, axis=0) / total_weight
    return ensemble_pred


def calculate_position_sizes(predicted_returns: np.ndarray, 
                           risk_budget: float = 0.02, 
                           max_position: float = 0.25):
    """
    Convert return predictions to position sizes
    
    Key insight: Size position proportional to expected return
    
    Args:
        predicted_returns: Array of predicted returns
        risk_budget: Maximum risk per trade (default 2%)
        max_position: Maximum position size (default 25%)
        
    Returns:
        Array of position sizes
    """
    # Kelly Criterion simplified: position_size = expected_return / variance
    # For simplicity, we'll use a fixed risk estimate
    
    # Calculate dynamic position sizes
    positions = predicted_returns / risk_budget
    
    # Apply hard limits
    positions = np.clip(positions, -max_position, max_position)
    
    # Apply confidence scaling (reduce size for uncertain predictions)
    abs_returns = np.abs(predicted_returns)
    mean_abs_return = np.mean(abs_returns)
    confidence = np.clip(abs_returns / (mean_abs_return + 1e-6), 0.5, 2.0)
    
    positions = positions * confidence
    
    return positions


def apply_transaction_cost_filter(positions: np.ndarray, 
                                current_positions: np.ndarray,
                                predicted_returns: np.ndarray, 
                                cost: float = 0.001):
    """
    Only trade if expected profit > transaction cost
    
    Args:
        positions: Target position sizes
        current_positions: Current position sizes
        predicted_returns: Predicted returns
        cost: Transaction cost (default 0.1%)
        
    Returns:
        Filtered position sizes
    """
    position_changes = np.abs(positions - current_positions)
    expected_profit = predicted_returns * np.abs(positions)
    transaction_cost = position_changes * cost
    
    # Don't trade if cost > profit (with 2x safety margin)
    profitable_trades = expected_profit > transaction_cost * 2
    
    # Keep current position if not profitable to change
    final_positions = np.where(profitable_trades, positions, current_positions)
    
    return final_positions


def create_regression_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create enhanced features for regression model."""
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


def backtest_regression_strategy(data: pd.DataFrame, 
                                models: dict, 
                                features: pd.DataFrame,
                                horizon: int = 5,
                                transaction_cost: float = 0.001):
    """
    Backtest the regression-based strategy
    
    Args:
        data: Price data
        models: Trained models
        features: Feature matrix
        horizon: Prediction horizon
        transaction_cost: Transaction cost
        
    Returns:
        Backtest results
    """
    logger.info("Starting regression strategy backtest")
    
    # Align data
    common_idx = data.index.intersection(features.index)
    data = data.loc[common_idx]
    features = features.loc[common_idx]
    
    # Initialize tracking
    positions = []
    returns = []
    current_position = 0.0
    
    # Walk forward simulation
    min_samples = 252  # 1 year of data needed
    
    for i in range(min_samples, len(data) - horizon):
        # Get historical data up to current point
        hist_features = features.iloc[:i]
        hist_data = data.iloc[:i]
        
        # Make prediction for current period
        current_features = features.iloc[i:i+1]
        predicted_return = predict_returns_ensemble(models, current_features)[0]
        
        # Calculate position size
        position = calculate_position_sizes(
            np.array([predicted_return]), 
            risk_budget=0.02, 
            max_position=0.25
        )[0]
        
        # Apply transaction cost filter
        position = apply_transaction_cost_filter(
            np.array([position]),
            np.array([current_position]),
            np.array([predicted_return]),
            cost=transaction_cost
        )[0]
        
        # Calculate actual return
        actual_return = data['close'].iloc[i + horizon] / data['close'].iloc[i] - 1
        
        # Strategy return = position * actual_return - transaction_costs
        position_change = abs(position - current_position)
        trade_cost = position_change * transaction_cost
        strategy_return = position * actual_return - trade_cost
        
        positions.append(position)
        returns.append(strategy_return)
        current_position = position
    
    # Convert to series
    backtest_idx = data.index[min_samples:-horizon]
    positions_series = pd.Series(positions, index=backtest_idx)
    returns_series = pd.Series(returns, index=backtest_idx)
    
    # Calculate performance metrics
    total_return = (1 + returns_series).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns_series)) - 1
    volatility = returns_series.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    max_drawdown = (returns_series.cumsum() - returns_series.cumsum().cummax()).min()
    
    results = {
        'positions': positions_series,
        'returns': returns_series,
        'cumulative_returns': returns_series.cumsum(),
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'n_trades': len(returns_series)
    }
    
    return results


def compare_classification_vs_regression(data: pd.DataFrame, features: pd.DataFrame):
    """
    Compare classification vs regression approaches
    
    Args:
        data: Price data
        features: Feature matrix
        
    Returns:
        Comparison results
    """
    logger.info("Comparing classification vs regression approaches")
    
    # Prepare data
    horizon = 5
    returns_5d = data['close'].pct_change(horizon).shift(-horizon)
    
    # Remove NaN
    valid_idx = ~(features.isna().any(axis=1) | returns_5d.isna())
    X = features[valid_idx]
    y_regression = returns_5d[valid_idx]
    y_classification = (y_regression > 0.01).astype(int)  # 1% threshold
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_reg_train, y_reg_test = y_regression.iloc[:split_idx], y_regression.iloc[split_idx:]
    y_clf_train, y_clf_test = y_classification.iloc[:split_idx], y_classification.iloc[split_idx:]
    
    # Train classification model
    from sklearn.ensemble import RandomForestClassifier
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train, y_clf_train)
    
    # Train regression models
    reg_models, _, _, _, _, _ = train_return_predictor(X_train, pd.DataFrame({'close': data['close'].iloc[:split_idx]}), horizon)
    
    # Generate predictions
    clf_probs = clf_model.predict_proba(X_test)[:, 1]
    reg_preds = predict_returns_ensemble(reg_models, X_test)
    
    # Convert to position sizes
    # Classification: threshold-based positions
    clf_positions = np.zeros_like(clf_probs)
    clf_positions[clf_probs > 0.6] = 0.2  # 20% position for high confidence
    clf_positions[clf_probs < 0.4] = -0.2  # -20% position for low confidence
    
    # Regression: predicted return-based positions
    reg_positions = calculate_position_sizes(reg_preds, risk_budget=0.02, max_position=0.25)
    
    # Calculate returns for both approaches
    actual_returns = y_reg_test.values
    
    clf_strategy_returns = clf_positions * actual_returns
    reg_strategy_returns = reg_positions * actual_returns
    
    # Performance metrics
    clf_sharpe = np.mean(clf_strategy_returns) / np.std(clf_strategy_returns) * np.sqrt(252) if np.std(clf_strategy_returns) > 0 else 0
    reg_sharpe = np.mean(reg_strategy_returns) / np.std(reg_strategy_returns) * np.sqrt(252) if np.std(reg_strategy_returns) > 0 else 0
    
    comparison = {
        'classification': {
            'total_return': np.sum(clf_strategy_returns),
            'sharpe_ratio': clf_sharpe,
            'volatility': np.std(clf_strategy_returns) * np.sqrt(252),
            'max_position': np.max(np.abs(clf_positions)),
            'avg_position': np.mean(np.abs(clf_positions))
        },
        'regression': {
            'total_return': np.sum(reg_strategy_returns),
            'sharpe_ratio': reg_sharpe,
            'volatility': np.std(reg_strategy_returns) * np.sqrt(252),
            'max_position': np.max(np.abs(reg_positions)),
            'avg_position': np.mean(np.abs(reg_positions))
        }
    }
    
    return comparison, clf_strategy_returns, reg_strategy_returns


def plot_results(backtest_results: dict, comparison_results: dict = None):
    """Plot backtest and comparison results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Cumulative returns
    axes[0, 0].plot(backtest_results['cumulative_returns'])
    axes[0, 0].set_title('Cumulative Returns - Regression Strategy')
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].grid(True)
    
    # Position sizes over time
    axes[0, 1].plot(backtest_results['positions'])
    axes[0, 1].set_title('Position Sizes Over Time')
    axes[0, 1].set_ylabel('Position Size')
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].grid(True)
    
    # Return distribution
    axes[1, 0].hist(backtest_results['returns'], bins=50, alpha=0.7)
    axes[1, 0].set_title('Return Distribution')
    axes[1, 0].set_xlabel('Daily Return')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].grid(True)
    
    # Performance comparison (if available)
    if comparison_results:
        metrics = ['total_return', 'sharpe_ratio', 'volatility']
        clf_values = [comparison_results['classification'][m] for m in metrics]
        reg_values = [comparison_results['regression'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, clf_values, width, label='Classification', alpha=0.7)
        axes[1, 1].bar(x + width/2, reg_values, width, label='Regression', alpha=0.7)
        axes[1, 1].set_title('Classification vs Regression')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'No comparison data', ha='center', va='center')
        axes[1, 1].set_title('Comparison (Not Available)')
    
    plt.tight_layout()
    plt.savefig('ml_regression_results.png', dpi=300, bbox_inches='tight')
    logger.info("Results plot saved as ml_regression_results.png")


def main():
    """Main execution function."""
    print("=" * 70)
    print("🤖 ML REGRESSION TRADING IMPLEMENTATION")
    print("=" * 70)
    
    # Configuration
    SYMBOL = 'AAPL'
    START_DATE = '2020-01-01'
    END_DATE = '2023-12-31'
    HORIZON = 5  # Predict 5-day returns
    
    try:
        # Load data
        logger.info(f"Loading data for {SYMBOL}")
        source = YFinanceSource()
        data = source.get_daily_bars(SYMBOL, START_DATE, END_DATE)
        
        if data.empty:
            raise ValueError(f"No data available for {SYMBOL}")
        
        # Convert columns to lowercase
        data.columns = data.columns.str.lower()
        logger.info(f"Loaded {len(data)} bars")
        
        # Create features
        logger.info("Creating regression features")
        features = create_regression_features(data)
        logger.info(f"Created {features.shape[1]} features")
        
        # Train regression models
        models, scores, X_train, X_val, y_train, y_val = train_return_predictor(
            features, data, horizon=HORIZON
        )
        
        print(f"\n📊 Model Performance Summary:")
        for name, score in scores.items():
            print(f"  {name:12}: R² = {score['r2']:6.4f}, MSE = {score['mse']:8.6f}")
        
        # Backtest the strategy
        logger.info("Running backtest")
        backtest_results = backtest_regression_strategy(
            data, models, features, horizon=HORIZON
        )
        
        print(f"\n📈 Backtest Results:")
        print(f"  Total Return:      {backtest_results['total_return']:8.2%}")
        print(f"  Annualized Return: {backtest_results['annualized_return']:8.2%}")
        print(f"  Volatility:        {backtest_results['volatility']:8.2%}")
        print(f"  Sharpe Ratio:      {backtest_results['sharpe_ratio']:8.2f}")
        print(f"  Max Drawdown:      {backtest_results['max_drawdown']:8.2%}")
        print(f"  Number of Trades:  {backtest_results['n_trades']:8.0f}")
        
        # Compare with classification approach
        logger.info("Comparing with classification approach")
        comparison, clf_returns, reg_returns = compare_classification_vs_regression(data, features)
        
        print(f"\n🏆 Classification vs Regression Comparison:")
        print(f"{'Metric':<20} {'Classification':<15} {'Regression':<15} {'Improvement':<15}")
        print("-" * 65)
        
        for metric in ['total_return', 'sharpe_ratio', 'volatility']:
            clf_val = comparison['classification'][metric]
            reg_val = comparison['regression'][metric]
            
            if metric == 'volatility':
                improvement = (clf_val - reg_val) / clf_val if clf_val != 0 else 0
                improvement_str = f"{improvement:+.1%}"
            else:
                improvement = (reg_val - clf_val) / abs(clf_val) if clf_val != 0 else 0
                improvement_str = f"{improvement:+.1%}"
            
            print(f"{metric:<20} {clf_val:<15.4f} {reg_val:<15.4f} {improvement_str:<15}")
        
        # Save models
        models_dir = Path("models/regression")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in models.items():
            model_path = models_dir / f"{name}_regressor.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} model to {model_path}")
        
        # Save feature names
        feature_path = models_dir / "regression_features.txt"
        with open(feature_path, 'w') as f:
            for feature in features.columns:
                f.write(f"{feature}\n")
        logger.info(f"Saved feature names to {feature_path}")
        
        # Plot results
        plot_results(backtest_results, comparison)
        
        # Feature importance analysis
        if 'rf' in models:
            rf_model = models['rf']
            if hasattr(rf_model, 'feature_importances_'):
                importance = pd.Series(
                    rf_model.feature_importances_,
                    index=features.columns
                ).sort_values(ascending=False)
                
                print(f"\n📊 Top 15 Important Features (Random Forest):")
                for feat, imp in importance.head(15).items():
                    print(f"  {feat:<25}: {imp:.4f}")
        
        print(f"\n✅ Regression trading implementation complete!")
        print(f"💾 Models saved to: {models_dir}")
        print(f"📊 Results plot: ml_regression_results.png")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())