"""
Technical indicator feature engineering for ML models
"""

import logging

import numpy as np
import pandas as pd
import talib

from ...core.base import ComponentConfig
from ..base import FeatureEngineer


class TechnicalFeatureEngineer(FeatureEngineer):
    """Generate technical indicator features for ML models"""

    def __init__(self, config: ComponentConfig | None = None):
        """Initialize technical feature engineer

        Args:
            config: Optional component configuration
        """
        if config is None:
            config = ComponentConfig(
                component_id="technical_features", component_type="feature_engineer"
            )
        super().__init__(config)

        self.logger = logging.getLogger(__name__)

        # Define feature groups
        self.feature_groups = {
            "momentum": ["rsi", "macd", "stoch", "williams_r", "roc"],
            "trend": ["sma", "ema", "adx", "cci", "aroon"],
            "volatility": ["bbands", "atr", "natr", "trange"],
            "volume": ["obv", "ad", "adosc", "mfi"],
            "pattern": ["cdl_patterns"],
        }

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate all technical indicator features

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with technical features
        """
        features = pd.DataFrame(index=data.index)

        # Validate input data
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            self.logger.warning(f"Missing columns for full feature generation: {missing_cols}")

        # Generate features by group
        if "close" in data.columns:
            features = pd.concat([features, self._generate_momentum_features(data)], axis=1)
            features = pd.concat([features, self._generate_trend_features(data)], axis=1)

        if all(col in data.columns for col in ["high", "low", "close"]):
            features = pd.concat([features, self._generate_volatility_features(data)], axis=1)

        if all(col in data.columns for col in ["close", "volume"]):
            features = pd.concat([features, self._generate_volume_features(data)], axis=1)

        if all(col in data.columns for col in ["open", "high", "low", "close"]):
            features = pd.concat([features, self._generate_pattern_features(data)], axis=1)

        # Add derived features
        features = pd.concat([features, self._generate_derived_features(features)], axis=1)

        # Store in cache
        self.feature_cache[data.index[-1]] = features

        return features

    def _generate_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum indicators

        Args:
            data: OHLCV data

        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=data.index)
        close = data["close"].values
        high = data.get("high", data["close"]).values
        low = data.get("low", data["close"]).values

        # RSI - Relative Strength Index
        for period in [14, 21, 28]:
            rsi = talib.RSI(close, timeperiod=period)
            features[f"rsi_{period}"] = rsi
            features[f"rsi_{period}_oversold"] = (rsi < 30).astype(int)
            features[f"rsi_{period}_overbought"] = (rsi > 70).astype(int)

        # MACD - Moving Average Convergence Divergence
        macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        features["macd"] = macd
        features["macd_signal"] = signal
        features["macd_histogram"] = hist
        features["macd_cross_up"] = (
            (macd > signal) & (pd.Series(macd).shift(1) <= pd.Series(signal).shift(1))
        ).astype(int)
        features["macd_cross_down"] = (
            (macd < signal) & (pd.Series(macd).shift(1) >= pd.Series(signal).shift(1))
        ).astype(int)

        # Stochastic Oscillator
        if "high" in data.columns and "low" in data.columns:
            slowk, slowd = talib.STOCH(
                high,
                low,
                close,
                fastk_period=14,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0,
            )
            features["stoch_k"] = slowk
            features["stoch_d"] = slowd
            features["stoch_oversold"] = (slowk < 20).astype(int)
            features["stoch_overbought"] = (slowk > 80).astype(int)

        # Williams %R
        if "high" in data.columns and "low" in data.columns:
            willr = talib.WILLR(high, low, close, timeperiod=14)
            features["williams_r"] = willr
            features["williams_oversold"] = (willr < -80).astype(int)
            features["williams_overbought"] = (willr > -20).astype(int)

        # Rate of Change
        for period in [10, 20]:
            features[f"roc_{period}"] = talib.ROC(close, timeperiod=period)

        return features

    def _generate_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trend indicators

        Args:
            data: OHLCV data

        Returns:
            DataFrame with trend features
        """
        features = pd.DataFrame(index=data.index)
        close = data["close"].values
        high = data.get("high", data["close"]).values
        low = data.get("low", data["close"]).values

        # Moving Averages
        for period in [10, 20, 50, 200]:
            sma = talib.SMA(close, timeperiod=period)
            ema = talib.EMA(close, timeperiod=period)

            features[f"sma_{period}"] = sma
            features[f"ema_{period}"] = ema
            features[f"price_to_sma_{period}"] = close / sma - 1
            features[f"price_to_ema_{period}"] = close / ema - 1

        # Moving Average Crossovers
        features["ma_cross_10_20"] = (features["sma_10"] > features["sma_20"]).astype(int)
        features["ma_cross_20_50"] = (features["sma_20"] > features["sma_50"]).astype(int)

        # ADX - Average Directional Index
        if "high" in data.columns and "low" in data.columns:
            adx = talib.ADX(high, low, close, timeperiod=14)
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)

            features["adx"] = adx
            features["plus_di"] = plus_di
            features["minus_di"] = minus_di
            features["di_diff"] = plus_di - minus_di
            features["trend_strength"] = (adx > 25).astype(int)

        # CCI - Commodity Channel Index
        if "high" in data.columns and "low" in data.columns:
            features["cci"] = talib.CCI(high, low, close, timeperiod=20)

        # Aroon Oscillator
        if "high" in data.columns and "low" in data.columns:
            aroon_up, aroon_down = talib.AROON(high, low, timeperiod=14)
            features["aroon_up"] = aroon_up
            features["aroon_down"] = aroon_down
            features["aroon_oscillator"] = aroon_up - aroon_down

        return features

    def _generate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility indicators

        Args:
            data: OHLCV data

        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=data.index)
        close = data["close"].values
        high = data["high"].values
        low = data["low"].values

        # Bollinger Bands
        for period in [20, 30]:
            upper, middle, lower = talib.BBANDS(
                close, timeperiod=period, nbdevup=2, nbdevdn=2, matype=0
            )

            features[f"bb_upper_{period}"] = upper
            features[f"bb_middle_{period}"] = middle
            features[f"bb_lower_{period}"] = lower
            features[f"bb_width_{period}"] = upper - lower
            features[f"bb_percent_{period}"] = (close - lower) / (upper - lower)
            features[f"bb_squeeze_{period}"] = features[f"bb_width_{period}"] / middle

        # ATR - Average True Range
        for period in [14, 20]:
            atr = talib.ATR(high, low, close, timeperiod=period)
            features[f"atr_{period}"] = atr
            features[f"atr_percent_{period}"] = atr / close

        # NATR - Normalized ATR
        features["natr"] = talib.NATR(high, low, close, timeperiod=14)

        # True Range
        features["trange"] = talib.TRANGE(high, low, close)

        # Historical Volatility (custom)
        returns = pd.Series(close).pct_change()
        for period in [20, 60]:
            features[f"volatility_{period}"] = returns.rolling(period).std() * np.sqrt(252)

        return features

    def _generate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume indicators

        Args:
            data: OHLCV data

        Returns:
            DataFrame with volume features
        """
        features = pd.DataFrame(index=data.index)
        close = data["close"].values
        volume = data["volume"].values
        high = data.get("high", data["close"]).values
        low = data.get("low", data["close"]).values

        # OBV - On Balance Volume
        obv = talib.OBV(close, volume)
        features["obv"] = obv
        features["obv_ma"] = talib.SMA(obv, timeperiod=20)
        features["obv_divergence"] = obv - features["obv_ma"]

        # AD - Accumulation/Distribution
        if "high" in data.columns and "low" in data.columns:
            ad = talib.AD(high, low, close, volume)
            features["ad"] = ad
            features["ad_ma"] = talib.SMA(ad, timeperiod=20)
            features["ad_divergence"] = ad - features["ad_ma"]

        # ADOSC - Chaikin A/D Oscillator
        if "high" in data.columns and "low" in data.columns:
            features["adosc"] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)

        # MFI - Money Flow Index
        if "high" in data.columns and "low" in data.columns:
            features["mfi"] = talib.MFI(high, low, close, volume, timeperiod=14)

        # Volume Rate of Change
        features["volume_roc"] = talib.ROC(volume, timeperiod=10)

        # Volume Moving Averages
        features["volume_sma_20"] = talib.SMA(volume, timeperiod=20)
        features["volume_ratio"] = volume / features["volume_sma_20"]

        # Price-Volume Trend
        features["pvt"] = (pd.Series(close).pct_change() * volume).cumsum()

        return features

    def _generate_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate candlestick pattern features

        Args:
            data: OHLCV data

        Returns:
            DataFrame with pattern features
        """
        features = pd.DataFrame(index=data.index)
        open_price = data["open"].values
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values

        # Candlestick patterns
        patterns = {
            "doji": talib.CDLDOJI,
            "hammer": talib.CDLHAMMER,
            "hanging_man": talib.CDLHANGINGMAN,
            "engulfing": talib.CDLENGULFING,
            "harami": talib.CDLHARAMI,
            "morning_star": talib.CDLMORNINGSTAR,
            "evening_star": talib.CDLEVENINGSTAR,
            "three_white_soldiers": talib.CDL3WHITESOLDIERS,
            "three_black_crows": talib.CDL3BLACKCROWS,
        }

        for name, pattern_func in patterns.items():
            try:
                features[f"cdl_{name}"] = pattern_func(open_price, high, low, close)
            except Exception as e:
                self.logger.debug(f"Could not calculate pattern {name}: {e}")

        # Aggregate pattern signals
        pattern_cols = [col for col in features.columns if col.startswith("cdl_")]
        if pattern_cols:
            features["bullish_patterns"] = features[pattern_cols].apply(
                lambda x: (x > 0).sum(), axis=1
            )
            features["bearish_patterns"] = features[pattern_cols].apply(
                lambda x: (x < 0).sum(), axis=1
            )
            features["pattern_strength"] = (
                features["bullish_patterns"] - features["bearish_patterns"]
            )

        return features

    def _generate_derived_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate derived features from existing features

        Args:
            features: DataFrame with existing features

        Returns:
            DataFrame with derived features
        """
        derived = pd.DataFrame(index=features.index)

        # Momentum composite
        momentum_cols = ["rsi_14", "macd_histogram", "stoch_k"]
        available_momentum = [col for col in momentum_cols if col in features.columns]
        if available_momentum:
            derived["momentum_composite"] = features[available_momentum].mean(axis=1)

        # Trend composite
        trend_cols = ["adx", "cci", "aroon_oscillator"]
        available_trend = [col for col in trend_cols if col in features.columns]
        if available_trend:
            derived["trend_composite"] = features[available_trend].mean(axis=1)

        # Volatility rank
        vol_cols = [col for col in features.columns if "volatility" in col or "atr" in col]
        if vol_cols:
            for col in vol_cols:
                if col in features.columns:
                    derived[f"{col}_rank"] = features[col].rank(pct=True)

        # Feature interactions
        if "rsi_14" in features.columns and "adx" in features.columns:
            derived["rsi_adx_interaction"] = features["rsi_14"] * features["adx"] / 100

        if "macd" in features.columns and "volume_ratio" in features.columns:
            derived["macd_volume_interaction"] = features["macd"] * features["volume_ratio"]

        return derived

    def get_feature_importance_hints(self) -> dict[str, float]:
        """Get hints about expected feature importance

        Returns:
            Dictionary of feature groups to expected importance
        """
        return {
            "momentum": 0.25,
            "trend": 0.25,
            "volatility": 0.20,
            "volume": 0.15,
            "pattern": 0.10,
            "derived": 0.05,
        }
