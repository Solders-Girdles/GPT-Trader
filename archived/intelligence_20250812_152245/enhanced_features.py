"""
Enhanced Feature Generation Framework for GPT-Trader Phase 1.

This module provides advanced feature engineering techniques:
- Wavelet transforms for multi-scale analysis
- Fourier transforms for cyclical pattern detection
- Polynomial feature combinations
- Technical pattern recognition
- Market microstructure features
- Alternative data integration

Extends the existing feature engineering pipeline with sophisticated transformations.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Optional advanced signal processing
try:
    import pywt

    HAS_WAVELETS = True
except ImportError:
    HAS_WAVELETS = False
    warnings.warn("PyWavelets not available. Install with: pip install PyWavelets")

try:
    from ta import add_all_ta_features

    HAS_TA = True
except ImportError:
    HAS_TA = False
    warnings.warn("ta library not available. Install with: pip install ta")

try:
    from sklearn.decomposition import PCA, FastICA
    from sklearn.feature_selection import VarianceThreshold

    HAS_SKLEARN_ADVANCED = True
except ImportError:
    HAS_SKLEARN_ADVANCED = False

from bot.utils.base import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class FeatureGenerationConfig(BaseConfig):
    """Configuration for enhanced feature generation."""

    # Wavelet transform parameters
    use_wavelets: bool = True
    wavelet_types: list[str] = field(default_factory=lambda: ["db4", "haar", "morlet"])
    wavelet_scales: list[int] = field(default_factory=lambda: [2, 4, 8, 16])

    # Fourier transform parameters
    use_fourier: bool = True
    fourier_periods: list[int] = field(default_factory=lambda: [5, 10, 20, 60, 252])
    fourier_components: int = 5

    # Polynomial features
    use_polynomial: bool = True
    polynomial_degree: int = 2
    polynomial_interaction_only: bool = True
    polynomial_include_bias: bool = False

    # Technical pattern recognition
    use_pattern_recognition: bool = True
    pattern_window_sizes: list[int] = field(default_factory=lambda: [10, 20, 50])

    # Market microstructure features
    use_microstructure: bool = True
    bid_ask_available: bool = False
    volume_profile: bool = True

    # Time-based features
    use_time_features: bool = True
    cyclical_encoding: bool = True
    calendar_effects: bool = True

    # Statistical features
    use_statistical_features: bool = True
    statistical_windows: list[int] = field(default_factory=lambda: [5, 10, 20, 50])

    # Alternative data features
    use_alternative_data: bool = False
    sentiment_data: bool = False
    economic_indicators: bool = False

    # Feature interaction detection
    use_feature_interactions: bool = True
    interaction_threshold: float = 0.1
    max_interactions: int = 50

    # Dimensionality reduction
    use_dimensionality_reduction: bool = True
    pca_components: int | None = None
    ica_components: int | None = None

    # Quality control
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    remove_highly_correlated: bool = True


class BaseFeatureGenerator(ABC):
    """Base class for feature generators."""

    def __init__(self, config: FeatureGenerationConfig) -> None:
        self.config = config
        self.feature_names_: list[str] | None = None

    @abstractmethod
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features from input data."""
        pass

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Get names of generated features."""
        pass


class WaveletFeatureGenerator(BaseFeatureGenerator):
    """Generate features using wavelet transforms."""

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate wavelet-based features."""
        if not HAS_WAVELETS:
            logger.warning("PyWavelets not available, skipping wavelet features")
            return pd.DataFrame(index=data.index)

        features = pd.DataFrame(index=data.index)

        # Apply wavelets to price columns
        price_columns = ["Open", "High", "Low", "Close"]
        available_columns = [col for col in price_columns if col in data.columns]

        for col in available_columns:
            series = data[col].dropna()
            if len(series) < 32:  # Minimum length for wavelets
                continue

            for wavelet in self.config.wavelet_types:
                try:
                    for scale in self.config.wavelet_scales:
                        # Continuous wavelet transform
                        coefficients, _ = pywt.cwt(series.values, scales=[scale], wavelet=wavelet)

                        # Extract features from coefficients
                        coeff_series = pd.Series(coefficients[0], index=series.index)

                        # Reindex to match original data
                        coeff_series = coeff_series.reindex(data.index)

                        # Statistical features of coefficients
                        feature_name = f"wavelet_{col}_{wavelet}_scale{scale}"
                        features[feature_name] = coeff_series

                        # Rolling statistics of coefficients
                        for window in [10, 20]:
                            features[f"{feature_name}_mean_{window}"] = coeff_series.rolling(
                                window
                            ).mean()
                            features[f"{feature_name}_std_{window}"] = coeff_series.rolling(
                                window
                            ).std()

                except Exception as e:
                    logger.warning(
                        f"Error generating wavelet features for {col} with {wavelet}: {e}"
                    )
                    continue

        self.feature_names_ = list(features.columns)
        return features

    def get_feature_names(self) -> list[str]:
        """Get wavelet feature names."""
        if self.feature_names_ is None:
            return []
        return self.feature_names_.copy()


class FourierFeatureGenerator(BaseFeatureGenerator):
    """Generate features using Fourier transforms."""

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Fourier-based features."""
        features = pd.DataFrame(index=data.index)

        # Apply FFT to price columns
        price_columns = ["Open", "High", "Low", "Close"]
        available_columns = [col for col in price_columns if col in data.columns]

        for col in available_columns:
            series = data[col].dropna()
            if len(series) < 64:  # Minimum length for meaningful FFT
                continue

            try:
                # Compute FFT
                fft_values = fft(series.values)
                frequencies = fftfreq(len(series))

                # Extract dominant frequencies
                power_spectrum = np.abs(fft_values) ** 2
                dominant_indices = np.argsort(power_spectrum)[-self.config.fourier_components :]

                for i, idx in enumerate(dominant_indices):
                    if idx == 0:  # Skip DC component
                        continue

                    freq = frequencies[idx]
                    amplitude = np.abs(fft_values[idx])
                    phase = np.angle(fft_values[idx])

                    # Create features
                    feature_base = f"fourier_{col}_comp{i}"
                    features[f"{feature_base}_freq"] = freq
                    features[f"{feature_base}_amplitude"] = amplitude
                    features[f"{feature_base}_phase"] = phase

                # Spectral features for specific periods
                for period in self.config.fourier_periods:
                    if period < len(series):
                        # Extract power at specific frequency
                        target_freq = 1.0 / period
                        freq_idx = np.argmin(np.abs(frequencies - target_freq))

                        power = power_spectrum[freq_idx]
                        features[f"fourier_{col}_power_{period}d"] = power

            except Exception as e:
                logger.warning(f"Error generating Fourier features for {col}: {e}")
                continue

        # Forward fill constant features
        for col in features.columns:
            if features[col].nunique() == 1:
                features[col] = features[col].fillna(method="ffill").fillna(0)
            else:
                features[col] = features[col].fillna(method="ffill")

        self.feature_names_ = list(features.columns)
        return features

    def get_feature_names(self) -> list[str]:
        """Get Fourier feature names."""
        if self.feature_names_ is None:
            return []
        return self.feature_names_.copy()


class PolynomialFeatureGenerator(BaseFeatureGenerator):
    """Generate polynomial and interaction features."""

    def __init__(self, config: FeatureGenerationConfig) -> None:
        super().__init__(config)
        self.poly_transformer = PolynomialFeatures(
            degree=config.polynomial_degree,
            interaction_only=config.polynomial_interaction_only,
            include_bias=config.polynomial_include_bias,
        )
        self.fitted = False

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate polynomial features."""
        # Select numeric columns for polynomial expansion
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # Limit to most important features to avoid explosion
        if len(numeric_columns) > 10:
            # Use variance as a simple importance measure
            variances = data[numeric_columns].var().sort_values(ascending=False)
            numeric_columns = variances.head(10).index.tolist()

        if not numeric_columns:
            return pd.DataFrame(index=data.index)

        # Handle missing values
        data_clean = data[numeric_columns].fillna(method="ffill").fillna(0)

        try:
            if not self.fitted:
                poly_features = self.poly_transformer.fit_transform(data_clean)
                self.fitted = True
            else:
                poly_features = self.poly_transformer.transform(data_clean)

            # Get feature names
            feature_names = self.poly_transformer.get_feature_names_out(numeric_columns)

            # Create DataFrame
            poly_df = pd.DataFrame(poly_features, index=data.index, columns=feature_names)

            # Remove original features (keep only interactions and higher-order terms)
            if self.config.polynomial_degree > 1:
                original_features = set(numeric_columns)
                interaction_features = [
                    col for col in poly_df.columns if col not in original_features
                ]
                poly_df = poly_df[interaction_features]

            self.feature_names_ = list(poly_df.columns)
            return poly_df

        except Exception as e:
            logger.warning(f"Error generating polynomial features: {e}")
            return pd.DataFrame(index=data.index)

    def get_feature_names(self) -> list[str]:
        """Get polynomial feature names."""
        if self.feature_names_ is None:
            return []
        return self.feature_names_.copy()


class PatternRecognitionGenerator(BaseFeatureGenerator):
    """Generate features based on technical analysis patterns."""

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate pattern recognition features."""
        features = pd.DataFrame(index=data.index)

        if "Close" not in data.columns:
            return features

        close = data["Close"]
        high = data.get("High", close)
        low = data.get("Low", close)
        volume = data.get("Volume", pd.Series(1, index=data.index))

        # Candlestick patterns
        features.update(self._candlestick_patterns(data))

        # Support and resistance levels
        features.update(self._support_resistance_features(high, low, close))

        # Trend patterns
        features.update(self._trend_patterns(close))

        # Volume patterns
        features.update(self._volume_patterns(close, volume))

        # Price action patterns
        features.update(self._price_action_patterns(data))

        self.feature_names_ = list(features.columns)
        return features

    def _candlestick_patterns(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Generate candlestick pattern features."""
        patterns = {}

        if not all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
            return patterns

        open_price = data["Open"]
        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        # Body and shadow sizes
        body_size = abs(close - open_price)
        upper_shadow = high - np.maximum(open_price, close)
        lower_shadow = np.minimum(open_price, close) - low
        total_range = high - low

        # Pattern indicators
        patterns["candle_body_ratio"] = body_size / (total_range + 1e-8)
        patterns["upper_shadow_ratio"] = upper_shadow / (total_range + 1e-8)
        patterns["lower_shadow_ratio"] = lower_shadow / (total_range + 1e-8)

        # Doji pattern (small body)
        patterns["doji_pattern"] = (body_size / (total_range + 1e-8) < 0.1).astype(int)

        # Hammer pattern (long lower shadow, small upper shadow)
        patterns["hammer_pattern"] = (
            (lower_shadow > 2 * body_size) & (upper_shadow < 0.5 * body_size)
        ).astype(int)

        # Shooting star pattern (long upper shadow, small lower shadow)
        patterns["shooting_star_pattern"] = (
            (upper_shadow > 2 * body_size) & (lower_shadow < 0.5 * body_size)
        ).astype(int)

        return patterns

    def _support_resistance_features(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> dict[str, pd.Series]:
        """Generate support and resistance features."""
        features = {}

        for window in self.config.pattern_window_sizes:
            # Rolling support and resistance
            resistance = high.rolling(window).max()
            support = low.rolling(window).min()

            # Distance from support/resistance
            features[f"resistance_distance_{window}"] = (resistance - close) / close
            features[f"support_distance_{window}"] = (close - support) / close

            # Support/resistance strength (how often price touched the level)
            resistance_touches = (high.rolling(window) >= resistance * 0.99).sum()
            support_touches = (low.rolling(window) <= support * 1.01).sum()

            features[f"resistance_strength_{window}"] = resistance_touches
            features[f"support_strength_{window}"] = support_touches

            # Breakout signals
            features[f"resistance_breakout_{window}"] = (close > resistance.shift(1)).astype(int)
            features[f"support_breakdown_{window}"] = (close < support.shift(1)).astype(int)

        return features

    def _trend_patterns(self, close: pd.Series) -> dict[str, pd.Series]:
        """Generate trend pattern features."""
        features = {}

        for window in self.config.pattern_window_sizes:
            # Linear regression slope (trend strength)
            def calculate_slope(series):
                if len(series) < 2:
                    return 0
                x = np.arange(len(series))
                slope, _, _, _, _ = stats.linregress(x, series)
                return slope

            features[f"trend_slope_{window}"] = close.rolling(window).apply(calculate_slope)

            # R-squared (trend quality)
            def calculate_r_squared(series):
                if len(series) < 3:
                    return 0
                x = np.arange(len(series))
                slope, intercept, r_value, _, _ = stats.linregress(x, series)
                return r_value**2

            features[f"trend_r2_{window}"] = close.rolling(window).apply(calculate_r_squared)

            # Higher highs and lower lows
            rolling_max = close.rolling(window).max()
            rolling_min = close.rolling(window).min()

            features[f"higher_high_{window}"] = (close > rolling_max.shift(1)).astype(int)
            features[f"lower_low_{window}"] = (close < rolling_min.shift(1)).astype(int)

        return features

    def _volume_patterns(self, close: pd.Series, volume: pd.Series) -> dict[str, pd.Series]:
        """Generate volume-based pattern features."""
        features = {}

        # Volume moving averages
        for window in [10, 20, 50]:
            vol_ma = volume.rolling(window).mean()
            features[f"volume_ratio_{window}"] = volume / vol_ma

            # Volume-price patterns
            price_change = close.pct_change()
            volume_change = volume.pct_change()

            # Volume confirmation
            features[f"volume_price_confirm_{window}"] = (
                (price_change > 0) & (volume > vol_ma)
            ).astype(int)

            # Volume divergence
            features[f"volume_divergence_{window}"] = abs(
                price_change.rolling(window).mean() - volume_change.rolling(window).mean()
            )

        return features

    def _price_action_patterns(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Generate price action pattern features."""
        features = {}

        if "Close" not in data.columns:
            return features

        close = data["Close"]
        returns = close.pct_change()

        # Consecutive patterns
        positive_days = (returns > 0).astype(int)
        negative_days = (returns < 0).astype(int)

        # Count consecutive positive/negative days
        for window in [5, 10, 20]:
            features[f"consecutive_positive_{window}"] = positive_days.rolling(window).sum()
            features[f"consecutive_negative_{window}"] = negative_days.rolling(window).sum()

        # Gap patterns
        if all(col in data.columns for col in ["Open", "Close"]):
            gap = data["Open"] - data["Close"].shift(1)
            gap_percent = gap / data["Close"].shift(1)

            features["gap_size"] = gap_percent
            features["gap_up"] = (gap_percent > 0.01).astype(int)
            features["gap_down"] = (gap_percent < -0.01).astype(int)

        return features

    def get_feature_names(self) -> list[str]:
        """Get pattern recognition feature names."""
        if self.feature_names_ is None:
            return []
        return self.feature_names_.copy()


class MicrostructureFeatureGenerator(BaseFeatureGenerator):
    """Generate market microstructure features."""

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate microstructure features."""
        features = pd.DataFrame(index=data.index)

        # Basic OHLCV features
        if all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
            features.update(self._ohlc_features(data))

        # Volume-based features
        if "Volume" in data.columns:
            features.update(self._volume_microstructure(data))

        # Price impact features
        features.update(self._price_impact_features(data))

        # Liquidity proxies
        features.update(self._liquidity_features(data))

        self.feature_names_ = list(features.columns)
        return features

    def _ohlc_features(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Generate OHLC-based microstructure features."""
        features = {}

        open_price = data["Open"]
        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        # Intraday returns
        features["overnight_return"] = open_price / close.shift(1) - 1
        features["intraday_return"] = close / open_price - 1
        features["high_low_ratio"] = (high - low) / close

        # Price position within the day's range
        features["close_position"] = (close - low) / (high - low + 1e-8)

        # True range components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        features["true_range_norm"] = true_range / close

        # Efficiency ratio (price move relative to total movement)
        for window in [10, 20]:
            price_change = abs(close - close.shift(window))
            volatility_sum = true_range.rolling(window).sum()
            features[f"efficiency_ratio_{window}"] = price_change / (volatility_sum + 1e-8)

        return features

    def _volume_microstructure(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Generate volume-based microstructure features."""
        features = {}

        volume = data["Volume"]
        close = data["Close"]
        returns = close.pct_change()

        # Volume-weighted average price (VWAP) approximation
        typical_price = (data["High"] + data["Low"] + data["Close"]) / 3

        for window in [10, 20]:
            vwap = (typical_price * volume).rolling(window).sum() / volume.rolling(window).sum()
            features[f"vwap_distance_{window}"] = (close - vwap) / vwap

        # Volume rate of change
        features["volume_roc"] = volume.pct_change()

        # Volume-return correlation
        for window in [10, 20]:
            features[f"volume_return_corr_{window}"] = returns.rolling(window).corr(
                volume.pct_change()
            )

        # On-balance volume
        obv = (volume * np.sign(returns)).cumsum()
        features["obv_norm"] = obv / obv.rolling(252).std()

        return features

    def _price_impact_features(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Generate price impact features."""
        features = {}

        close = data["Close"]
        returns = close.pct_change()
        volume = data.get("Volume", pd.Series(1, index=data.index))

        # Kyle's lambda (price impact)
        for window in [10, 20]:
            # Simple approximation: return volatility / volume
            return_vol = returns.rolling(window).std()
            avg_volume = volume.rolling(window).mean()
            features[f"price_impact_{window}"] = return_vol / (avg_volume + 1e-8)

        # Amihud illiquidity measure
        for window in [10, 20]:
            illiquidity = abs(returns) / (volume + 1e-8)
            features[f"amihud_illiquidity_{window}"] = illiquidity.rolling(window).mean()

        return features

    def _liquidity_features(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Generate liquidity proxy features."""
        features = {}

        close = data["Close"]
        high = data.get("High", close)
        low = data.get("Low", close)
        data.get("Volume", pd.Series(1, index=data.index))

        # Roll spread estimator
        returns = close.pct_change()
        for window in [10, 20]:
            autocovariance = returns.rolling(window + 1).cov(returns.shift(1))
            roll_spread = 2 * np.sqrt(np.maximum(-autocovariance, 0))
            features[f"roll_spread_{window}"] = roll_spread

        # High-low spread
        hl_spread = (high - low) / close
        features["hl_spread"] = hl_spread

        for window in [10, 20]:
            features[f"hl_spread_ma_{window}"] = hl_spread.rolling(window).mean()

        return features

    def get_feature_names(self) -> list[str]:
        """Get microstructure feature names."""
        if self.feature_names_ is None:
            return []
        return self.feature_names_.copy()


class TimeBasedFeatureGenerator(BaseFeatureGenerator):
    """Generate time-based features."""

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate time-based features."""
        features = pd.DataFrame(index=data.index)

        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("Data index is not DatetimeIndex, skipping time features")
            return features

        # Calendar features
        if self.config.calendar_effects:
            features.update(self._calendar_features(data.index))

        # Cyclical features
        if self.config.cyclical_encoding:
            features.update(self._cyclical_features(data.index))

        # Market regime time features
        features.update(self._market_time_features(data))

        self.feature_names_ = list(features.columns)
        return features

    def _calendar_features(self, index: pd.DatetimeIndex) -> dict[str, pd.Series]:
        """Generate calendar-based features."""
        features = {}

        # Basic time components
        features["year"] = pd.Series(index.year, index=index)
        features["month"] = pd.Series(index.month, index=index)
        features["day"] = pd.Series(index.day, index=index)
        features["dayofweek"] = pd.Series(index.dayofweek, index=index)
        features["dayofyear"] = pd.Series(index.dayofyear, index=index)
        features["week"] = pd.Series(index.isocalendar().week, index=index)
        features["quarter"] = pd.Series(index.quarter, index=index)

        # Market-specific features
        features["is_month_start"] = pd.Series(index.is_month_start.astype(int), index=index)
        features["is_month_end"] = pd.Series(index.is_month_end.astype(int), index=index)
        features["is_quarter_start"] = pd.Series(index.is_quarter_start.astype(int), index=index)
        features["is_quarter_end"] = pd.Series(index.is_quarter_end.astype(int), index=index)
        features["is_year_start"] = pd.Series(index.is_year_start.astype(int), index=index)
        features["is_year_end"] = pd.Series(index.is_year_end.astype(int), index=index)

        # Days since events
        features["days_since_month_start"] = pd.Series(
            (index - index.to_period("M").start_time).days, index=index
        )
        features["days_until_month_end"] = pd.Series(
            (index.to_period("M").end_time - index).days, index=index
        )

        return features

    def _cyclical_features(self, index: pd.DatetimeIndex) -> dict[str, pd.Series]:
        """Generate cyclical encoded features."""
        features = {}

        # Cyclical encoding for periodic features
        # Day of week (0-6)
        dow = index.dayofweek
        features["dayofweek_sin"] = pd.Series(np.sin(2 * np.pi * dow / 7), index=index)
        features["dayofweek_cos"] = pd.Series(np.cos(2 * np.pi * dow / 7), index=index)

        # Month (1-12)
        month = index.month
        features["month_sin"] = pd.Series(np.sin(2 * np.pi * month / 12), index=index)
        features["month_cos"] = pd.Series(np.cos(2 * np.pi * month / 12), index=index)

        # Day of year (1-365/366)
        doy = index.dayofyear
        features["dayofyear_sin"] = pd.Series(np.sin(2 * np.pi * doy / 365), index=index)
        features["dayofyear_cos"] = pd.Series(np.cos(2 * np.pi * doy / 365), index=index)

        # Hour (if available)
        if hasattr(index, "hour"):
            hour = index.hour
            features["hour_sin"] = pd.Series(np.sin(2 * np.pi * hour / 24), index=index)
            features["hour_cos"] = pd.Series(np.cos(2 * np.pi * hour / 24), index=index)

        return features

    def _market_time_features(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Generate market timing features."""
        features = {}

        index = data.index

        # Time since market events (approximation)
        # Assume market opens at 9:30 AM ET and closes at 4:00 PM ET
        market_open_hour = 9.5
        market_close_hour = 16.0

        if hasattr(index, "hour"):
            hour = pd.Series(index.hour + index.minute / 60.0, index=index)

            # Time until market open/close
            features["time_to_open"] = np.where(
                hour < market_open_hour, market_open_hour - hour, 24 + market_open_hour - hour
            )

            features["time_to_close"] = np.where(
                hour < market_close_hour, market_close_hour - hour, 24 + market_close_hour - hour
            )

            # Market session indicators
            features["is_premarket"] = ((hour >= 4) & (hour < market_open_hour)).astype(int)
            features["is_market_hours"] = (
                (hour >= market_open_hour) & (hour < market_close_hour)
            ).astype(int)
            features["is_aftermarket"] = ((hour >= market_close_hour) & (hour < 20)).astype(int)

        return features

    def get_feature_names(self) -> list[str]:
        """Get time-based feature names."""
        if self.feature_names_ is None:
            return []
        return self.feature_names_.copy()


class EnhancedFeatureFramework:
    """
    Comprehensive framework for enhanced feature generation.

    Coordinates multiple feature generators and provides integrated
    feature engineering pipeline.
    """

    def __init__(self, config: FeatureGenerationConfig) -> None:
        self.config = config
        self.generators: dict[str, BaseFeatureGenerator] = {}
        self.feature_names_: list[str] | None = None
        self.scaler: StandardScaler | None = None

        # Initialize generators based on configuration
        self._initialize_generators()

    def _initialize_generators(self) -> None:
        """Initialize feature generators based on configuration."""
        if self.config.use_wavelets:
            self.generators["wavelets"] = WaveletFeatureGenerator(self.config)

        if self.config.use_fourier:
            self.generators["fourier"] = FourierFeatureGenerator(self.config)

        if self.config.use_polynomial:
            self.generators["polynomial"] = PolynomialFeatureGenerator(self.config)

        if self.config.use_pattern_recognition:
            self.generators["patterns"] = PatternRecognitionGenerator(self.config)

        if self.config.use_microstructure:
            self.generators["microstructure"] = MicrostructureFeatureGenerator(self.config)

        if self.config.use_time_features:
            self.generators["time"] = TimeBasedFeatureGenerator(self.config)

        logger.info(
            f"Initialized {len(self.generators)} feature generators: {list(self.generators.keys())}"
        )

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate all enhanced features."""
        logger.info("Generating enhanced features...")

        all_features = pd.DataFrame(index=data.index)

        # Generate features from each generator
        for name, generator in self.generators.items():
            try:
                logger.info(f"Generating {name} features...")
                features = generator.generate_features(data)

                if not features.empty:
                    # Add prefix to avoid name conflicts
                    features.columns = [f"{name}_{col}" for col in features.columns]
                    all_features = pd.concat([all_features, features], axis=1)

                logger.info(f"Generated {len(features.columns)} {name} features")

            except Exception as e:
                logger.error(f"Error generating {name} features: {e}")
                continue

        # Apply quality control
        if not all_features.empty:
            all_features = self._apply_quality_control(all_features)

        # Store feature names
        self.feature_names_ = list(all_features.columns)

        logger.info(
            f"Enhanced feature generation completed. Total features: {len(all_features.columns)}"
        )

        return all_features

    def _apply_quality_control(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply quality control to generated features."""
        original_count = len(features.columns)

        # Remove features with low variance
        if self.config.variance_threshold > 0:
            variance_selector = VarianceThreshold(threshold=self.config.variance_threshold)

            # Handle missing values
            features_filled = features.fillna(method="ffill").fillna(0)

            try:
                mask = variance_selector.fit(features_filled).get_support()
                features = features.loc[:, mask]
                logger.info(
                    f"Removed {original_count - len(features.columns)} low-variance features"
                )
            except (ValueError, AttributeError):
                logger.warning("Could not apply variance threshold")

        # Remove highly correlated features
        if self.config.remove_highly_correlated:
            features = self._remove_highly_correlated_features(features)

        # Handle infinite values
        features = features.replace([np.inf, -np.inf], np.nan)

        # Forward fill and then fill remaining NaNs with 0
        features = features.fillna(method="ffill").fillna(0)

        return features

    def _remove_highly_correlated_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        try:
            # Calculate correlation matrix
            corr_matrix = features.corr().abs()

            # Find pairs of highly correlated features
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            # Find features to drop
            to_drop = [
                column
                for column in upper_triangle.columns
                if any(upper_triangle[column] > self.config.correlation_threshold)
            ]

            if to_drop:
                features = features.drop(columns=to_drop)
                logger.info(f"Removed {len(to_drop)} highly correlated features")

        except Exception as e:
            logger.warning(f"Could not remove correlated features: {e}")

        return features

    def get_feature_names(self) -> list[str]:
        """Get all generated feature names."""
        if self.feature_names_ is None:
            return []
        return self.feature_names_.copy()

    def get_generator_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary of all feature generators."""
        summary = {}

        for name, generator in self.generators.items():
            feature_names = generator.get_feature_names()
            summary[name] = {"n_features": len(feature_names), "feature_names": feature_names}

        return summary


def create_default_enhanced_features() -> EnhancedFeatureFramework:
    """Create default enhanced feature generation framework."""
    config = FeatureGenerationConfig(
        use_wavelets=HAS_WAVELETS,
        use_fourier=True,
        use_polynomial=True,
        use_pattern_recognition=True,
        use_microstructure=True,
        use_time_features=True,
        use_statistical_features=True,
    )

    return EnhancedFeatureFramework(config)
