"""
Baseline Trading Models for Comparison
Phase 2.5 - Day 9

Collection of baseline models for benchmarking ML strategies.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
import talib

logger = logging.getLogger(__name__)


class BaselineModel(ABC):
    """Abstract base class for baseline models"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the model (if needed)"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions"""
        predictions = self.predict(X)
        # Convert to probability format
        proba = np.zeros((len(predictions), 2))
        proba[:, 0] = 1 - predictions
        proba[:, 1] = predictions
        return proba


class BuyAndHoldModel(BaselineModel):
    """Always buy strategy"""
    
    def __init__(self):
        super().__init__("BuyAndHold")
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """No fitting needed"""
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Always predict buy (1)"""
        return np.ones(len(X))


class RandomModel(BaselineModel):
    """Random predictions with configurable probability"""
    
    def __init__(self, buy_probability: float = 0.5, seed: int = 42):
        super().__init__("Random")
        self.buy_probability = buy_probability
        self.seed = seed
        np.random.seed(seed)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """No fitting needed"""
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Random predictions"""
        return (np.random.random(len(X)) < self.buy_probability).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Random probabilities"""
        proba = np.random.random((len(X), 2))
        proba[:, 0] = 1 - proba[:, 1]
        return proba


class SMAcrossoverModel(BaselineModel):
    """Simple Moving Average crossover strategy"""
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__("SMACrossover")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """No fitting needed"""
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate signals based on SMA crossover"""
        if 'close' not in X.columns and 'price' not in X.columns:
            # If no price data, return random
            return RandomModel().predict(X)
        
        price_col = 'close' if 'close' in X.columns else 'price'
        prices = X[price_col].values
        
        # Calculate SMAs
        sma_fast = talib.SMA(prices, timeperiod=self.fast_period)
        sma_slow = talib.SMA(prices, timeperiod=self.slow_period)
        
        # Generate signals (1 when fast > slow)
        signals = (sma_fast > sma_slow).astype(int)
        
        # Handle NaN values
        signals = np.nan_to_num(signals, nan=0)
        
        return signals


class MeanReversionModel(BaselineModel):
    """Mean reversion strategy"""
    
    def __init__(self, lookback: int = 20, z_threshold: float = 2.0):
        super().__init__("MeanReversion")
        self.lookback = lookback
        self.z_threshold = z_threshold
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """No fitting needed"""
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate signals based on mean reversion"""
        if 'returns' not in X.columns:
            # If no returns, calculate from price if available
            if 'close' in X.columns or 'price' in X.columns:
                price_col = 'close' if 'close' in X.columns else 'price'
                returns = X[price_col].pct_change().values
            else:
                return RandomModel().predict(X)
        else:
            returns = X['returns'].values
        
        # Calculate rolling statistics
        rolling_mean = pd.Series(returns).rolling(self.lookback).mean().values
        rolling_std = pd.Series(returns).rolling(self.lookback).std().values
        
        # Calculate z-score
        z_score = (returns - rolling_mean) / (rolling_std + 1e-10)
        
        # Generate signals (buy when oversold)
        signals = (z_score < -self.z_threshold).astype(int)
        
        # Handle NaN values
        signals = np.nan_to_num(signals, nan=0)
        
        return signals


class MomentumModel(BaselineModel):
    """Momentum-based strategy"""
    
    def __init__(self, lookback: int = 20):
        super().__init__("Momentum")
        self.lookback = lookback
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """No fitting needed"""
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate signals based on momentum"""
        if 'close' not in X.columns and 'price' not in X.columns:
            return RandomModel().predict(X)
        
        price_col = 'close' if 'close' in X.columns else 'price'
        prices = X[price_col].values
        
        # Calculate momentum (rate of change)
        momentum = talib.ROC(prices, timeperiod=self.lookback)
        
        # Generate signals (buy when momentum is positive)
        signals = (momentum > 0).astype(int)
        
        # Handle NaN values
        signals = np.nan_to_num(signals, nan=0)
        
        return signals


class RSIModel(BaselineModel):
    """RSI-based trading strategy"""
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__("RSI")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """No fitting needed"""
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate signals based on RSI"""
        if 'close' not in X.columns and 'price' not in X.columns:
            return RandomModel().predict(X)
        
        price_col = 'close' if 'close' in X.columns else 'price'
        prices = X[price_col].values
        
        # Calculate RSI
        rsi = talib.RSI(prices, timeperiod=self.period)
        
        # Generate signals (buy when oversold)
        signals = (rsi < self.oversold).astype(int)
        
        # Handle NaN values
        signals = np.nan_to_num(signals, nan=0)
        
        return signals


class BollingerBandsModel(BaselineModel):
    """Bollinger Bands trading strategy"""
    
    def __init__(self, period: int = 20, num_std: float = 2.0):
        super().__init__("BollingerBands")
        self.period = period
        self.num_std = num_std
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """No fitting needed"""
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate signals based on Bollinger Bands"""
        if 'close' not in X.columns and 'price' not in X.columns:
            return RandomModel().predict(X)
        
        price_col = 'close' if 'close' in X.columns else 'price'
        prices = X[price_col].values
        
        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            prices,
            timeperiod=self.period,
            nbdevup=self.num_std,
            nbdevdn=self.num_std,
            matype=0
        )
        
        # Generate signals (buy when price touches lower band)
        signals = (prices <= lower).astype(int)
        
        # Handle NaN values
        signals = np.nan_to_num(signals, nan=0)
        
        return signals


class MACDModel(BaselineModel):
    """MACD-based trading strategy"""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("MACD")
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """No fitting needed"""
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate signals based on MACD"""
        if 'close' not in X.columns and 'price' not in X.columns:
            return RandomModel().predict(X)
        
        price_col = 'close' if 'close' in X.columns else 'price'
        prices = X[price_col].values
        
        # Calculate MACD
        macd, macd_signal, macd_hist = talib.MACD(
            prices,
            fastperiod=self.fast,
            slowperiod=self.slow,
            signalperiod=self.signal
        )
        
        # Generate signals (buy when MACD crosses above signal)
        signals = (macd > macd_signal).astype(int)
        
        # Handle NaN values
        signals = np.nan_to_num(signals, nan=0)
        
        return signals


class VolumeWeightedModel(BaselineModel):
    """Volume-weighted trading strategy"""
    
    def __init__(self, lookback: int = 20):
        super().__init__("VolumeWeighted")
        self.lookback = lookback
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """No fitting needed"""
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate signals based on volume patterns"""
        if 'volume' not in X.columns:
            return RandomModel().predict(X)
        
        volumes = X['volume'].values
        
        # Calculate volume moving average
        volume_ma = pd.Series(volumes).rolling(self.lookback).mean().values
        
        # Generate signals (buy when volume is above average)
        signals = (volumes > volume_ma * 1.5).astype(int)
        
        # Handle NaN values
        signals = np.nan_to_num(signals, nan=0)
        
        return signals


class EnsembleBaselineModel(BaselineModel):
    """Ensemble of multiple baseline strategies"""
    
    def __init__(self, models: Optional[List[BaselineModel]] = None):
        super().__init__("EnsembleBaseline")
        if models is None:
            self.models = [
                SMAcrossoverModel(),
                RSIModel(),
                MomentumModel(),
                BollingerBandsModel()
            ]
        else:
            self.models = models
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit all models"""
        for model in self.models:
            model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Ensemble predictions (majority vote)"""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Majority vote
        ensemble_pred = np.mean(predictions, axis=0)
        return (ensemble_pred >= 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Ensemble probabilities (average)"""
        probas = []
        for model in self.models:
            proba = model.predict_proba(X)
            probas.append(proba)
        
        # Average probabilities
        return np.mean(probas, axis=0)


class AdaptiveBaselineModel(BaselineModel):
    """Adaptive strategy that switches between models based on market regime"""
    
    def __init__(self):
        super().__init__("AdaptiveBaseline")
        self.trend_model = MomentumModel()
        self.range_model = MeanReversionModel()
        self.volatility_threshold = 0.02
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit component models"""
        self.trend_model.fit(X, y)
        self.range_model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Adaptive predictions based on market regime"""
        if 'returns' not in X.columns:
            if 'close' in X.columns or 'price' in X.columns:
                price_col = 'close' if 'close' in X.columns else 'price'
                returns = X[price_col].pct_change()
            else:
                return RandomModel().predict(X)
        else:
            returns = X['returns']
        
        # Calculate rolling volatility
        volatility = returns.rolling(20).std()
        
        # Initialize predictions
        predictions = np.zeros(len(X))
        
        # Use different models based on volatility regime
        high_vol_mask = volatility > self.volatility_threshold
        low_vol_mask = ~high_vol_mask
        
        # High volatility: use mean reversion
        if high_vol_mask.any():
            predictions[high_vol_mask] = self.range_model.predict(
                X[high_vol_mask]
            )
        
        # Low volatility: use momentum
        if low_vol_mask.any():
            predictions[low_vol_mask] = self.trend_model.predict(
                X[low_vol_mask]
            )
        
        return predictions.astype(int)


def get_baseline_models() -> Dict[str, BaselineModel]:
    """Get dictionary of all baseline models"""
    return {
        'buy_hold': BuyAndHoldModel(),
        'random': RandomModel(),
        'sma_cross': SMAcrossoverModel(),
        'mean_reversion': MeanReversionModel(),
        'momentum': MomentumModel(),
        'rsi': RSIModel(),
        'bollinger': BollingerBandsModel(),
        'macd': MACDModel(),
        'volume': VolumeWeightedModel(),
        'ensemble': EnsembleBaselineModel(),
        'adaptive': AdaptiveBaselineModel()
    }


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Get sample data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="1y")
    
    # Prepare features
    X = pd.DataFrame(index=data.index)
    X['close'] = data['Close']
    X['volume'] = data['Volume']
    X['returns'] = data['Close'].pct_change()
    X = X.dropna()
    
    # Create target
    y = (data['Close'].shift(-1) > data['Close']).astype(int)
    y = y.loc[X.index]
    
    # Test all baseline models
    models = get_baseline_models()
    
    print("Testing Baseline Models:")
    print("-" * 50)
    
    for name, model in models.items():
        # Fit and predict
        model.fit(X, y)
        predictions = model.predict(X)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == y)
        
        # Calculate other metrics
        n_trades = np.sum(predictions)
        win_rate = np.mean(y[predictions == 1]) if n_trades > 0 else 0
        
        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Trades: {n_trades}")
        print(f"  Win Rate: {win_rate:.3f}")