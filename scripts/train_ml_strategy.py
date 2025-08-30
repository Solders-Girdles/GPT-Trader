#!/usr/bin/env python3
"""
Train ML Strategy Models
========================
Script to train machine learning models for generating trading signals.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.strategy.ml_signal_strategy import MLSignalStrategy
from bot.dataflow.sources.yfinance_source import YFinanceSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_training_data(symbol: str, start_date: str, end_date: str, label_threshold: float = 0.01):
    """
    Prepare training data with features and labels.
    
    Args:
        symbol: Stock symbol
        start_date: Start date for data
        end_date: End date for data
        label_threshold: Return threshold for positive label (default 1%)
    
    Returns:
        X: Feature DataFrame
        y: Labels (1 for profitable, 0 for not)
        data: Original OHLCV data
    """
    logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
    
    # Load data
    source = YFinanceSource()
    data = source.get_daily_bars(symbol, start_date, end_date)
    
    if data.empty:
        raise ValueError(f"No data available for {symbol}")
    
    # Convert columns to lowercase (YFinanceSource returns Title Case)
    data.columns = data.columns.str.lower()
    
    # Ensure numeric columns are float64 for TA-Lib compatibility
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in data.columns:
            data[col] = data[col].astype(np.float64)
    
    logger.info(f"Loaded {len(data)} bars")
    
    # Create ML strategy to generate features
    strategy = MLSignalStrategy()
    features = strategy.generate_features(data)
    
    # Create labels: 1 if next day return > threshold, 0 otherwise
    next_day_returns = data['close'].shift(-1) / data['close'] - 1
    labels = (next_day_returns > label_threshold).astype(int)
    
    # Alternative labeling: 3-day forward return
    # forward_returns = data['close'].shift(-3) / data['close'] - 1
    # labels = (forward_returns > label_threshold * 3).astype(int)
    
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


def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train an XGBoost model."""
    from xgboost import XGBClassifier
    
    # Model parameters optimized for trading
    model_params = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'use_label_encoder': False
    }
    
    logger.info("Training XGBoost model...")
    model = XGBClassifier(**model_params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    return model


def train_lightgbm_model(X_train, y_train, X_test, y_test):
    """Train a LightGBM model."""
    import lightgbm as lgb
    
    # Model parameters
    model_params = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'random_state': 42,
        'verbose': -1
    }
    
    logger.info("Training LightGBM model...")
    model = lgb.LGBMClassifier(**model_params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='logloss',
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
    )
    
    return model


def train_random_forest_model(X_train, y_train):
    """Train a Random Forest model."""
    from sklearn.ensemble import RandomForestClassifier
    
    model_params = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }
    
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance."""
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_prob) if hasattr(model, 'predict_proba') else 0
    }
    
    print(f"\n{'='*60}")
    print(f"📊 {model_name} Performance")
    print('='*60)
    for metric, value in metrics.items():
        print(f"{metric.capitalize():12}: {value:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"TN: {cm[0,0]:5}  FP: {cm[0,1]:5}")
    print(f"FN: {cm[1,0]:5}  TP: {cm[1,1]:5}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = pd.Series(
            model.feature_importances_,
            index=X_test.columns
        ).sort_values(ascending=False)
        
        print(f"\nTop 10 Features:")
        for feat, imp in importance.head(10).items():
            print(f"  {feat:20}: {imp:.4f}")
    
    return metrics


def create_trading_signals_demo(model, X_test, y_test, threshold=0.5):
    """Demonstrate trading signal generation."""
    # Get predictions
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = model.predict(X_test)
    
    # Generate signals
    signals = pd.Series(0, index=X_test.index)
    signals[probabilities > threshold + 0.1] = 1  # Buy
    signals[probabilities < threshold - 0.1] = -1  # Sell
    
    # Calculate hypothetical returns
    # Note: This is simplified - real trading has costs, slippage, etc.
    signal_df = pd.DataFrame({
        'signal': signals,
        'actual': y_test,
        'probability': probabilities
    })
    
    # Count signals
    buy_signals = (signals == 1).sum()
    sell_signals = (signals == -1).sum()
    neutral = (signals == 0).sum()
    
    print(f"\n📈 Trading Signals Generated:")
    print(f"  Buy signals:  {buy_signals:5} ({buy_signals/len(signals)*100:.1f}%)")
    print(f"  Sell signals: {sell_signals:5} ({sell_signals/len(signals)*100:.1f}%)")
    print(f"  Neutral:      {neutral:5} ({neutral/len(signals)*100:.1f}%)")
    
    # Accuracy of buy signals
    buy_accuracy = signal_df[signal_df['signal'] == 1]['actual'].mean()
    if buy_signals > 0:
        print(f"\n  Buy signal accuracy: {buy_accuracy:.2%}")
    
    return signal_df


def main():
    print("="*60)
    print("🤖 ML STRATEGY TRAINING")
    print("="*60)
    
    # Configuration
    SYMBOLS = ['AAPL', 'MSFT', 'GOOGL']  # Train on multiple symbols
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
    models = {}
    
    # 1. XGBoost
    try:
        xgb_model = train_xgboost_model(X_train, y_train, X_test, y_test)
        models['xgboost'] = xgb_model
        evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    except ImportError:
        print("⚠️ XGBoost not available")
    
    # 2. LightGBM
    try:
        lgb_model = train_lightgbm_model(X_train, y_train, X_test, y_test)
        models['lightgbm'] = lgb_model
        evaluate_model(lgb_model, X_test, y_test, "LightGBM")
    except ImportError:
        print("⚠️ LightGBM not available")
    
    # 3. Random Forest
    rf_model = train_random_forest_model(X_train, y_train)
    models['random_forest'] = rf_model
    evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Select best model
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            from sklearn.metrics import roc_auc_score
            score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
    
    if best_model:
        print(f"\n🏆 Best model: {best_name} (AUC: {best_score:.4f})")
        
        # Generate trading signals demo
        signals = create_trading_signals_demo(best_model, X_test, y_test)
        
        # Save the best model
        model_path = Path("models/ml_strategy_model.pkl")
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(best_model, model_path)
        print(f"\n💾 Model saved to: {model_path}")
        
        # Save feature names for reference
        feature_path = Path("models/ml_strategy_features.txt")
        with open(feature_path, 'w') as f:
            for feature in X_train.columns:
                f.write(f"{feature}\n")
        print(f"📝 Features saved to: {feature_path}")
    
    print("\n✅ Training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())