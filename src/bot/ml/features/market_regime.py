"""
Market regime feature engineering for ML models
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

from ...core.base import ComponentConfig
from ..base import FeatureEngineer


class MarketRegimeFeatures(FeatureEngineer):
    """Generate features for market regime detection"""

    def __init__(self, config: ComponentConfig | None = None):
        """Initialize market regime feature engineer

        Args:
            config: Optional component configuration
        """
        if config is None:
            config = ComponentConfig(
                component_id="regime_features", component_type="feature_engineer"
            )
        super().__init__(config)

        self.logger = logging.getLogger(__name__)

        # Define lookback periods for different timeframes
        self.lookback_periods = {"short": [1, 5, 10], "medium": [20, 30, 60], "long": [120, 250]}

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate market regime detection features

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with regime features
        """
        features = pd.DataFrame(index=data.index)

        # Price-based features
        features = pd.concat([features, self._generate_return_features(data)], axis=1)
        features = pd.concat([features, self._generate_volatility_features(data)], axis=1)
        features = pd.concat([features, self._generate_trend_features(data)], axis=1)

        # Volume-based features
        if "volume" in data.columns:
            features = pd.concat([features, self._generate_volume_features(data)], axis=1)

        # Market microstructure features
        features = pd.concat([features, self._generate_microstructure_features(data)], axis=1)

        # Statistical features
        features = pd.concat([features, self._generate_statistical_features(data)], axis=1)

        # Regime transition features
        features = pd.concat([features, self._generate_transition_features(features)], axis=1)

        # Store in cache
        self.feature_cache[data.index[-1]] = features

        return features

    def _generate_return_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate return-based features

        Args:
            data: OHLCV data

        Returns:
            DataFrame with return features
        """
        features = pd.DataFrame(index=data.index)
        close = data["close"]

        # Simple returns for different periods
        for period in [1, 5, 10, 20, 60, 120]:
            features[f"returns_{period}d"] = close.pct_change(period)

        # Log returns
        for period in [1, 5, 20]:
            features[f"log_returns_{period}d"] = np.log(close / close.shift(period))

        # Cumulative returns
        features["cum_returns_20d"] = (
            (1 + features["returns_1d"]).rolling(20).apply(lambda x: x.prod() - 1, raw=False)
        )
        features["cum_returns_60d"] = (
            (1 + features["returns_1d"]).rolling(60).apply(lambda x: x.prod() - 1, raw=False)
        )

        # Return momentum
        features["return_momentum"] = features["returns_5d"] - features["returns_20d"]

        # Return acceleration
        features["return_acceleration"] = features["returns_1d"] - features["returns_1d"].shift(1)

        # Relative returns (vs moving average)
        ma_20 = close.rolling(20).mean()
        ma_50 = close.rolling(50).mean()
        features["return_vs_ma20"] = (close - ma_20) / ma_20
        features["return_vs_ma50"] = (close - ma_50) / ma_50

        return features

    def _generate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility-based features

        Args:
            data: OHLCV data

        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=data.index)
        close = data["close"]
        returns = close.pct_change()

        # Rolling volatility (standard deviation)
        for period in [10, 20, 60]:
            features[f"volatility_{period}d"] = returns.rolling(period).std()
            # Annualized
            features[f"volatility_{period}d_ann"] = features[f"volatility_{period}d"] * np.sqrt(252)

        # Volatility ratios
        features["volatility_ratio_20_60"] = features["volatility_20d"] / features["volatility_60d"]
        features["volatility_ratio_10_20"] = features["volatility_10d"] / features["volatility_20d"]

        # EWMA volatility
        features["ewma_volatility"] = returns.ewm(span=20).std()

        # Volatility of volatility
        features["vol_of_vol"] = features["volatility_20d"].rolling(20).std()

        # Parkinson volatility (using high-low)
        if "high" in data.columns and "low" in data.columns:
            hl_ratio = np.log(data["high"] / data["low"])
            features["parkinson_vol"] = np.sqrt(1 / (4 * np.log(2))) * hl_ratio.rolling(20).std()

        # Garman-Klass volatility
        if all(col in data.columns for col in ["open", "high", "low"]):
            c = np.log(data["close"] / data["open"])
            hl = np.log(data["high"] / data["low"])
            features["garman_klass_vol"] = (
                np.sqrt(0.5 * hl**2 - (2 * np.log(2) - 1) * c**2).rolling(20).mean()
            )

        # Volatility regime
        vol_median = features["volatility_20d"].rolling(120).median()
        features["high_vol_regime"] = (features["volatility_20d"] > vol_median * 1.5).astype(int)
        features["low_vol_regime"] = (features["volatility_20d"] < vol_median * 0.7).astype(int)

        return features

    def _generate_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trend-based features

        Args:
            data: OHLCV data

        Returns:
            DataFrame with trend features
        """
        features = pd.DataFrame(index=data.index)
        close = data["close"]

        # Moving average trends
        for short, long in [(10, 20), (20, 50), (50, 200)]:
            ma_short = close.rolling(short).mean()
            ma_long = close.rolling(long).mean()

            features[f"ma_trend_{short}_{long}"] = (ma_short > ma_long).astype(int)
            features[f"ma_spread_{short}_{long}"] = (ma_short - ma_long) / ma_long

        # Linear regression trend
        for period in [20, 60]:

            def calc_trend(x):
                if len(x) < 2:
                    return np.nan
                y = np.arange(len(x))
                slope, _, r_value, _, _ = stats.linregress(y, x)
                return slope

            features[f"linear_trend_{period}d"] = close.rolling(period).apply(calc_trend, raw=False)
            features[f"trend_strength_{period}d"] = close.rolling(period).apply(
                lambda x: stats.linregress(np.arange(len(x)), x)[2] ** 2 if len(x) > 1 else np.nan,
                raw=False,
            )

        # Trend consistency
        features["trend_consistency"] = (
            features["ma_trend_10_20"]
            + features["ma_trend_20_50"]
            + features.get("ma_trend_50_200", 0)
        ) / 3

        # Higher highs and lower lows
        if "high" in data.columns and "low" in data.columns:
            high_20 = data["high"].rolling(20).max()
            low_20 = data["low"].rolling(20).min()

            features["higher_highs"] = (data["high"] > high_20.shift(1)).astype(int)
            features["lower_lows"] = (data["low"] < low_20.shift(1)).astype(int)
            features["trend_channel_position"] = (close - low_20) / (high_20 - low_20)

        return features

    def _generate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-based regime features

        Args:
            data: OHLCV data

        Returns:
            DataFrame with volume features
        """
        features = pd.DataFrame(index=data.index)
        volume = data["volume"]
        close = data["close"]

        # Volume trends
        for period in [10, 20, 60]:
            vol_ma = volume.rolling(period).mean()
            features[f"volume_ma_{period}d"] = vol_ma
            features[f"volume_ratio_{period}d"] = volume / vol_ma

        # Volume-price correlation
        for period in [20, 60]:
            features[f"volume_price_corr_{period}d"] = close.rolling(period).corr(volume)

        # Volume volatility
        features["volume_volatility"] = volume.pct_change().rolling(20).std()

        # Volume regime
        vol_median = volume.rolling(60).median()
        features["high_volume_regime"] = (volume > vol_median * 1.5).astype(int)
        features["low_volume_regime"] = (volume < vol_median * 0.7).astype(int)

        # Dollar volume
        dollar_volume = close * volume
        features["dollar_volume"] = dollar_volume
        features["dollar_volume_ma"] = dollar_volume.rolling(20).mean()

        # Volume-weighted average price (VWAP)
        features["vwap"] = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
        features["price_to_vwap"] = close / features["vwap"] - 1

        return features

    def _generate_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate market microstructure features

        Args:
            data: OHLCV data

        Returns:
            DataFrame with microstructure features
        """
        features = pd.DataFrame(index=data.index)

        if all(col in data.columns for col in ["open", "high", "low", "close"]):
            open_price = data["open"]
            high = data["high"]
            low = data["low"]
            close = data["close"]

            # Intraday range
            features["intraday_range"] = (high - low) / close
            features["intraday_range_ma"] = features["intraday_range"].rolling(20).mean()

            # Gap analysis
            features["gap"] = (open_price - close.shift(1)) / close.shift(1)
            features["gap_filled"] = ((features["gap"] > 0) & (low <= close.shift(1))) | (
                (features["gap"] < 0) & (high >= close.shift(1))
            )

            # Close position in daily range
            features["close_position"] = (close - low) / (high - low + 1e-10)

            # Body to range ratio (candlestick body)
            features["body_ratio"] = np.abs(close - open_price) / (high - low + 1e-10)

            # Upper and lower shadows
            features["upper_shadow"] = (high - np.maximum(open_price, close)) / (high - low + 1e-10)
            features["lower_shadow"] = (np.minimum(open_price, close) - low) / (high - low + 1e-10)

            # Efficiency ratio
            net_change = np.abs(close - close.shift(20))
            sum_changes = np.abs(close.diff()).rolling(20).sum()
            features["efficiency_ratio"] = net_change / (sum_changes + 1e-10)

        return features

    def _generate_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical features for regime detection

        Args:
            data: OHLCV data

        Returns:
            DataFrame with statistical features
        """
        features = pd.DataFrame(index=data.index)
        close = data["close"]
        returns = close.pct_change()

        # Distribution moments
        for period in [20, 60]:
            # Skewness
            features[f"skewness_{period}d"] = returns.rolling(period).skew()

            # Kurtosis
            features[f"kurtosis_{period}d"] = returns.rolling(period).kurt()

            # Jarque-Bera test statistic
            def jb_stat(x):
                if len(x) < 4:
                    return np.nan
                n = len(x)
                s = stats.skew(x)
                k = stats.kurtosis(x)
                return n / 6 * (s**2 + 0.25 * (k**2))

            features[f"jarque_bera_{period}d"] = returns.rolling(period).apply(jb_stat, raw=False)

        # Autocorrelation
        for lag in [1, 5, 10]:
            features[f"autocorr_lag{lag}"] = returns.rolling(60).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan, raw=False
            )

        # Hurst exponent (trend persistence)
        def hurst_exponent(ts, max_lag=20):
            """Calculate Hurst exponent"""
            if len(ts) < max_lag:
                return np.nan

            lags = range(2, min(max_lag, len(ts) // 2))
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]

            if not tau or all(t == 0 for t in tau):
                return np.nan

            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0

        features["hurst_exponent"] = close.rolling(60).apply(
            lambda x: hurst_exponent(x.values), raw=False
        )

        # Entropy (market uncertainty)
        def shannon_entropy(x, bins=10):
            """Calculate Shannon entropy"""
            if len(x) < bins:
                return np.nan
            hist, _ = np.histogram(x, bins=bins)
            probs = hist / len(x)
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))

        features["entropy"] = returns.rolling(60).apply(
            lambda x: shannon_entropy(x.values), raw=False
        )

        return features

    def _generate_transition_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate features for regime transitions

        Args:
            features: DataFrame with existing features

        Returns:
            DataFrame with transition features
        """
        transition = pd.DataFrame(index=features.index)

        # Volatility transitions
        if "volatility_20d" in features.columns:
            vol_change = features["volatility_20d"].pct_change(5)
            transition["vol_increasing"] = (vol_change > 0.2).astype(int)
            transition["vol_decreasing"] = (vol_change < -0.2).astype(int)
            transition["vol_spike"] = (vol_change > 0.5).astype(int)

        # Trend transitions
        if "ma_trend_20_50" in features.columns:
            transition["trend_change"] = features["ma_trend_20_50"].diff().abs()
            transition["trend_reversal"] = (features["ma_trend_20_50"].diff().abs() > 0).astype(int)

        # Volume transitions
        if "volume_ratio_20d" in features.columns:
            transition["volume_surge"] = (features["volume_ratio_20d"] > 2).astype(int)
            transition["volume_dry_up"] = (features["volume_ratio_20d"] < 0.5).astype(int)

        # Momentum transitions
        if "returns_5d" in features.columns and "returns_20d" in features.columns:
            mom_change = features["returns_5d"] - features["returns_20d"]
            transition["momentum_improving"] = (mom_change > 0.02).astype(int)
            transition["momentum_deteriorating"] = (mom_change < -0.02).astype(int)

        # Regime change indicators
        regime_features = ["high_vol_regime", "ma_trend_20_50", "high_volume_regime"]
        available_regime = [f for f in regime_features if f in features.columns]

        if available_regime:
            # Count regime changes
            for feat in available_regime:
                transition[f"{feat}_change"] = features[feat].diff().abs()

            # Composite regime change score
            transition["regime_change_score"] = sum(
                transition[f"{feat}_change"] for feat in available_regime
            ) / len(available_regime)

        return transition

    def get_regime_labels(self, features: pd.DataFrame, n_regimes: int = 4) -> pd.Series:
        """Generate regime labels for training (helper method)

        Args:
            features: DataFrame with regime features
            n_regimes: Number of regimes to identify

        Returns:
            Series with regime labels
        """
        # Select key features for regime identification
        key_features = ["volatility_20d", "returns_20d", "ma_trend_20_50", "volume_ratio_20d"]

        available_features = [f for f in key_features if f in features.columns]

        if not available_features:
            return pd.Series(0, index=features.index)

        # Normalize features
        X = features[available_features].ffill().fillna(0)
        X = (X - X.mean()) / (X.std() + 1e-10)

        # Simple k-means clustering for regime identification
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regimes = kmeans.fit_predict(X)

        return pd.Series(regimes, index=features.index)
