#!/usr/bin/env python3
"""
Simple ML Strategy Training
===========================
A simplified version that uses basic features to train an ML model for trading.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.dataflow.sources.yfinance_source import YFinanceSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create simple technical features without TA-Lib."""
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


def prepare_training_data(symbol: str, start_date: str, end_date: str, label_threshold: float = 0.01):
    """Prepare training data with simple features and labels."""
    logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
    
    # Load data
    source = YFinanceSource()
    data = source.get_daily_bars(symbol, start_date, end_date)
    
    if data.empty:
        raise ValueError(f"No data available for {symbol}")
    
    # Convert columns to lowercase
    data.columns = data.columns.str.lower()
    
    # Ensure numeric columns are float64
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in data.columns:
            data[col] = data[col].astype(np.float64)
    
    logger.info(f"Loaded {len(data)} bars")
    
    # Create features
    features = create_simple_features(data)
    
    # Create labels: 1 if next day return > threshold, 0 otherwise
    next_day_returns = data['close'].shift(-1) / data['close'] - 1
    labels = (next_day_returns > label_threshold).astype(int)
    
    # Remove last row (no label) and first rows with NaN features
    features = features.iloc[:-1]  # Remove last row (no future return)
    labels = labels.iloc[:-1]
    
    # Drop NaN values
    valid_idx = features.dropna().index
    X = features.loc[valid_idx]
    y = labels.loc[valid_idx]
    
    logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
    logger.info(f"Label distribution: {y.value_counts().to_dict()}")
    
    return X, y, data


def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and compare performance."""
    models = {}
    results = {}
    
    # 1. Random Forest
    logger.info("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['random_forest'] = rf_model
    
    # Evaluate
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results['random_forest'] = accuracy
    logger.info(f"Random Forest Accuracy: {accuracy:.4f}")
    
    # 2. XGBoost (if available)
    try:
        from xgboost import XGBClassifier
        
        logger.info("Training XGBoost...")
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='binary:logistic',
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train, verbose=False)
        models['xgboost'] = xgb_model
        
        # Evaluate
        y_pred = xgb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results['xgboost'] = accuracy
        logger.info(f"XGBoost Accuracy: {accuracy:.4f}")
        
    except ImportError:
        logger.warning("XGBoost not available")
    
    # 3. LightGBM (if available)
    try:
        import lightgbm as lgb
        
        logger.info("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        models['lightgbm'] = lgb_model
        
        # Evaluate
        y_pred = lgb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results['lightgbm'] = accuracy
        logger.info(f"LightGBM Accuracy: {accuracy:.4f}")
        
    except ImportError:
        logger.warning("LightGBM not available")
    
    return models, results


def generate_trading_signals(model, X_test, threshold=0.5):
    """Generate trading signals from model predictions."""
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = model.predict(X_test)
    
    # Generate signals
    signals = pd.Series(0, index=X_test.index)
    signals[probabilities > threshold + 0.1] = 1  # Buy
    signals[probabilities < threshold - 0.1] = -1  # Sell
    
    # Count signals
    buy_signals = (signals == 1).sum()
    sell_signals = (signals == -1).sum()
    neutral = (signals == 0).sum()
    
    print(f"\n📈 Trading Signals Generated:")
    print(f"  Buy signals:  {buy_signals:5} ({buy_signals/len(signals)*100:.1f}%)")
    print(f"  Sell signals: {sell_signals:5} ({sell_signals/len(signals)*100:.1f}%)")
    print(f"  Neutral:      {neutral:5} ({neutral/len(signals)*100:.1f}%)")
    
    return signals


def main():
    print("=" * 60)
    print("🤖 SIMPLE ML STRATEGY TRAINING")
    print("=" * 60)
    
    # Configuration
    SYMBOLS = ['AAPL', 'MSFT', 'GOOGL']
    START_DATE = '2020-01-01'
    END_DATE = '2023-12-31'
    TEST_SIZE = 0.2
    
    # Collect data from multiple symbols
    all_X = []
    all_y = []
    
    for symbol in SYMBOLS:
        try:
            X, y, data = prepare_training_data(symbol, START_DATE, END_DATE)
            all_X.append(X)
            all_y.append(y)
            print(f"✅ Loaded {symbol}: {len(X)} samples")
        except Exception as e:
            print(f"❌ Failed to load {symbol}: {e}")
    
    if not all_X:
        print("❌ No data loaded!")
        return 1
    
    # Combine all data
    X_combined = pd.concat(all_X, axis=0)
    y_combined = pd.concat(all_y, axis=0)
    
    print(f"\n📊 Combined dataset: {len(X_combined)} samples")
    print(f"   Features: {X_combined.shape[1]}")
    print(f"   Positive labels: {y_combined.sum()} ({y_combined.mean():.1%})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=y_combined
    )
    
    print(f"\n📚 Training set: {len(X_train)} samples")
    print(f"📝 Test set: {len(X_test)} samples")
    
    # Train models
    models, results = train_models(X_train, y_train, X_test, y_test)
    
    # Select best model
    if models:
        best_model_name = max(results, key=results.get)
        best_model = models[best_model_name]
        best_accuracy = results[best_model_name]
        
        print(f"\n🏆 Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        # Generate trading signals
        signals = generate_trading_signals(best_model, X_test)
        
        # Save the best model
        model_path = Path("models/simple_ml_model.pkl")
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(best_model, model_path)
        print(f"\n💾 Model saved to: {model_path}")
        
        # Save feature names for reference
        feature_path = Path("models/simple_ml_features.txt")
        with open(feature_path, 'w') as f:
            for feature in X_train.columns:
                f.write(f"{feature}\n")
        print(f"📝 Features saved to: {feature_path}")
        
        # Print feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            importance = pd.Series(
                best_model.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=False)
            
            print(f"\n📊 Top 10 Important Features:")
            for feat, imp in importance.head(10).items():
                print(f"  {feat:25}: {imp:.4f}")
    
    print("\n✅ Training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())