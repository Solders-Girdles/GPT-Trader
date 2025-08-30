"""
Data collection and preparation for ML strategy selection - LOCAL to this slice.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple
from .types import StrategyPerformanceRecord, StrategyName, MarketConditions
from .features import extract_market_features


def collect_training_data(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime
) -> List[StrategyPerformanceRecord]:
    """
    Collect historical performance data for training.
    
    Runs backtests for each strategy on each symbol to build training set.
    """
    records = []
    
    for symbol in symbols:
        print(f"  Collecting data for {symbol}...")
        
        # Generate synthetic training data (in production, use real backtests)
        for i in range(50):  # 50 samples per symbol
            # Random date in range
            days_diff = (end_date - start_date).days
            random_days = np.random.randint(0, max(1, days_diff))
            sample_date = start_date + timedelta(days=random_days)
            
            # Random market conditions
            conditions = MarketConditions(
                volatility=np.random.uniform(5, 50),
                trend_strength=np.random.uniform(-100, 100),
                volume_ratio=np.random.uniform(0.5, 3),
                price_momentum=np.random.uniform(-20, 20),
                market_regime=np.random.choice(['bull', 'bear', 'sideways']),
                vix_level=np.random.uniform(10, 40),
                correlation_spy=np.random.uniform(-1, 1)
            )
            
            # Generate performance for each strategy
            for strategy in StrategyName:
                # Synthetic performance based on conditions
                performance = _generate_synthetic_performance(strategy, conditions)
                
                record = StrategyPerformanceRecord(
                    strategy=strategy,
                    symbol=symbol,
                    date=sample_date,
                    market_conditions=conditions,
                    actual_return=performance['return'],
                    actual_sharpe=performance['sharpe'],
                    actual_drawdown=performance['drawdown'],
                    trades_count=performance['trades'],
                    win_rate=performance['win_rate']
                )
                records.append(record)
    
    return records


def prepare_datasets(
    records: List[StrategyPerformanceRecord],
    validation_split: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training and validation datasets from records.
    
    Returns X_train, X_val, y_train, y_val.
    """
    # Extract features and labels
    X = []
    y = []
    
    for record in records:
        features = extract_market_features(record.market_conditions)
        X.append(features)
        
        # Label is combination of strategy and performance
        label = f"{record.strategy.value}_{record.actual_return:.2f}"
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split into train/validation
    split_idx = int(len(X) * (1 - validation_split))
    X_train = X[:split_idx]
    X_val = X[split_idx:]
    y_train = y[:split_idx]
    y_val = y[split_idx:]
    
    return X_train, X_val, y_train, y_val


def _generate_synthetic_performance(
    strategy: StrategyName,
    conditions: MarketConditions
) -> dict:
    """
    Generate synthetic performance data for a strategy.
    
    In production, this would come from actual backtests.
    """
    # Base performance
    base_return = np.random.normal(5, 10)
    base_sharpe = np.random.normal(0.5, 0.5)
    base_drawdown = -abs(np.random.normal(15, 10))
    base_trades = np.random.randint(5, 50)
    base_win_rate = np.random.uniform(0.3, 0.7)
    
    # Adjust based on strategy and conditions
    if strategy == StrategyName.MOMENTUM:
        if abs(conditions.trend_strength) > 50:
            base_return += 10
            base_sharpe += 0.5
            base_win_rate += 0.1
        else:
            base_return -= 5
            base_sharpe -= 0.2
            
    elif strategy == StrategyName.MEAN_REVERSION:
        if conditions.market_regime == 'sideways':
            base_return += 8
            base_sharpe += 0.3
            base_trades *= 2
        else:
            base_return -= 3
            
    elif strategy == StrategyName.VOLATILITY:
        if conditions.volatility > 30:
            base_return += 12
            base_sharpe += 0.4
            base_drawdown *= 1.5
        else:
            base_return -= 8
            base_sharpe -= 0.3
            
    elif strategy == StrategyName.BREAKOUT:
        if conditions.price_momentum > 10 and conditions.volume_ratio > 1.5:
            base_return += 15
            base_sharpe += 0.6
            base_win_rate += 0.15
        else:
            base_return -= 2
            
    elif strategy == StrategyName.SIMPLE_MA:
        # MA is steady but not spectacular
        base_return = np.clip(base_return, -5, 10)
        base_sharpe = np.clip(base_sharpe, 0, 1)
    
    return {
        'return': base_return,
        'sharpe': max(base_sharpe, -1),
        'drawdown': base_drawdown,
        'trades': base_trades,
        'win_rate': np.clip(base_win_rate, 0, 1)
    }