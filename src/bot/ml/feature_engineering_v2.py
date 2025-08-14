"""
Optimized Feature Engineering
Phase 2.5 - Day 5

Reduces features from 200+ to ~50 essential ones based on importance and correlation analysis.
"""

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    VarianceThreshold,
)
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class FeatureCategory(Enum):
    """Feature categories for organization"""

    PRICE = "price"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    TREND = "trend"
    MARKET_MICROSTRUCTURE = "market_microstructure"


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""

    # Feature selection
    max_features: int = 50
    min_variance: float = 0.01
    correlation_threshold: float = 0.95

    # Technical indicators
    short_window: int = 5
    medium_window: int = 20
    long_window: int = 50

    # Scaling
    use_robust_scaler: bool = True  # Better for outliers

    # PCA
    pca_variance_ratio: float = 0.95  # Keep 95% of variance


class OptimizedFeatureEngineer:
    """
    Optimized feature engineering with reduced, high-quality features.

    Key improvements:
    - Reduced from 200+ to ~50 features
    - Removed highly correlated features
    - Focus on proven technical indicators
    - Proper feature scaling
    - Feature importance tracking
    """

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.scaler = None
        self.selected_features = []
        self.feature_importance = {}

        # Define essential features by category
        self.essential_features = {
            FeatureCategory.PRICE: [
                "returns_1d",
                "returns_5d",
                "returns_20d",
                "log_returns_1d",
                "log_returns_5d",
                "price_to_sma20",
                "price_to_sma50",
                "high_low_ratio",
                "close_to_high",
                "close_to_low",
            ],
            FeatureCategory.VOLUME: [
                "volume_ratio_5d",
                "volume_ratio_20d",
                "volume_trend",
                "obv_signal",
                "vwap_deviation",
            ],
            FeatureCategory.MOMENTUM: [
                "rsi_14",
                "rsi_divergence",
                "macd_signal",
                "macd_histogram",
                "stoch_k",
                "stoch_d",
                "williams_r",
                "roc_10",
            ],
            FeatureCategory.VOLATILITY: [
                "atr_14",
                "atr_ratio",
                "bollinger_width",
                "bollinger_position",
                "realized_volatility_5d",
                "realized_volatility_20d",
                "parkinson_volatility",
            ],
            FeatureCategory.TREND: [
                "adx_14",
                "adx_trend_strength",
                "ema_trend_strength",
                "sma_trend_alignment",
                "ichimoku_signal",
                "supertrend_signal",
            ],
            FeatureCategory.MARKET_MICROSTRUCTURE: [
                "bid_ask_spread",
                "order_flow_imbalance",
                "tick_rule",
                "effective_spread",
                "price_impact",
            ],
        }

        logger.info(
            f"OptimizedFeatureEngineer initialized with max {self.config.max_features} features"
        )

    def calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based features"""
        features = pd.DataFrame(index=df.index)

        # Returns
        features["returns_1d"] = df["close"].pct_change(1)
        features["returns_5d"] = df["close"].pct_change(5)
        features["returns_20d"] = df["close"].pct_change(20)

        # Log returns (more stable for modeling)
        features["log_returns_1d"] = np.log(df["close"] / df["close"].shift(1))
        features["log_returns_5d"] = np.log(df["close"] / df["close"].shift(5))

        # Price relative to moving averages
        sma_20 = df["close"].rolling(20).mean()
        sma_50 = df["close"].rolling(50).mean()
        features["price_to_sma20"] = df["close"] / sma_20 - 1
        features["price_to_sma50"] = df["close"] / sma_50 - 1

        # Price position in daily range
        features["high_low_ratio"] = (df["high"] - df["low"]) / df["close"]
        features["close_to_high"] = (df["high"] - df["close"]) / (df["high"] - df["low"]).replace(
            0, np.nan
        )
        features["close_to_low"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(
            0, np.nan
        )

        return features

    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features"""
        features = pd.DataFrame(index=df.index)

        # Volume ratios
        volume_ma_5 = df["volume"].rolling(5).mean()
        volume_ma_20 = df["volume"].rolling(20).mean()
        features["volume_ratio_5d"] = df["volume"] / volume_ma_5.replace(0, np.nan)
        features["volume_ratio_20d"] = df["volume"] / volume_ma_20.replace(0, np.nan)

        # Volume trend
        features["volume_trend"] = volume_ma_5 / volume_ma_20.replace(0, np.nan)

        # On-Balance Volume (OBV)
        obv = talib.OBV(df["close"].values, df["volume"].values)
        obv_ma = pd.Series(obv).rolling(20).mean()
        features["obv_signal"] = pd.Series(obv) / obv_ma.replace(0, np.nan) - 1

        # VWAP deviation
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
        features["vwap_deviation"] = df["close"] / vwap - 1

        return features

    def calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        features = pd.DataFrame(index=df.index)

        # RSI
        rsi = talib.RSI(df["close"].values, timeperiod=14)
        features["rsi_14"] = rsi / 100  # Normalize to [0, 1]

        # RSI divergence
        price_change = df["close"].pct_change(14)
        rsi_change = pd.Series(rsi).pct_change(14)
        features["rsi_divergence"] = np.sign(price_change) != np.sign(rsi_change)

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df["close"].values)
        features["macd_signal"] = np.sign(macd - macd_signal)
        features["macd_histogram"] = macd_hist / df["close"].values  # Normalize

        # Stochastic
        stoch_k, stoch_d = talib.STOCH(df["high"].values, df["low"].values, df["close"].values)
        features["stoch_k"] = stoch_k / 100
        features["stoch_d"] = stoch_d / 100

        # Williams %R
        williams_r = talib.WILLR(df["high"].values, df["low"].values, df["close"].values)
        features["williams_r"] = (williams_r + 100) / 100  # Normalize to [0, 1]

        # Rate of Change
        features["roc_10"] = talib.ROC(df["close"].values, timeperiod=10) / 100

        return features

    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility features"""
        features = pd.DataFrame(index=df.index)

        # ATR
        atr = talib.ATR(df["high"].values, df["low"].values, df["close"].values, timeperiod=14)
        features["atr_14"] = atr / df["close"].values  # Normalize
        features["atr_ratio"] = atr / pd.Series(atr).rolling(50).mean().replace(0, np.nan)

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df["close"].values)
        bb_width = upper - lower
        features["bollinger_width"] = bb_width / middle  # Normalized width
        features["bollinger_position"] = (df["close"].values - lower) / bb_width.clip(lower=0.0001)

        # Realized volatility
        returns = df["close"].pct_change()
        features["realized_volatility_5d"] = returns.rolling(5).std() * np.sqrt(252)
        features["realized_volatility_20d"] = returns.rolling(20).std() * np.sqrt(252)

        # Parkinson volatility (using high-low)
        hl_ratio = np.log(df["high"] / df["low"])
        features["parkinson_volatility"] = hl_ratio.rolling(20).apply(
            lambda x: np.sqrt(np.sum(x**2) / (4 * len(x) * np.log(2)))
        ) * np.sqrt(252)

        return features

    def calculate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend indicators"""
        features = pd.DataFrame(index=df.index)

        # ADX
        adx = talib.ADX(df["high"].values, df["low"].values, df["close"].values, timeperiod=14)
        features["adx_14"] = adx / 100  # Normalize
        features["adx_trend_strength"] = pd.cut(
            pd.Series(adx), bins=[0, 25, 50, 75, 100], labels=[0.25, 0.5, 0.75, 1.0]
        ).astype(float)

        # EMA trend
        ema_12 = talib.EMA(df["close"].values, timeperiod=12)
        ema_26 = talib.EMA(df["close"].values, timeperiod=26)
        features["ema_trend_strength"] = (ema_12 - ema_26) / df["close"].values

        # SMA alignment
        sma_10 = talib.SMA(df["close"].values, timeperiod=10)
        sma_20 = talib.SMA(df["close"].values, timeperiod=20)
        sma_50 = talib.SMA(df["close"].values, timeperiod=50)
        features["sma_trend_alignment"] = ((sma_10 > sma_20) & (sma_20 > sma_50)).astype(float)

        # Ichimoku (simplified)
        high_9 = df["high"].rolling(9).max()
        low_9 = df["low"].rolling(9).min()
        conversion_line = (high_9 + low_9) / 2

        high_26 = df["high"].rolling(26).max()
        low_26 = df["low"].rolling(26).min()
        base_line = (high_26 + low_26) / 2

        features["ichimoku_signal"] = np.sign(conversion_line - base_line)

        # Supertrend (simplified)
        hl_avg = (df["high"] + df["low"]) / 2
        atr = talib.ATR(df["high"].values, df["low"].values, df["close"].values, timeperiod=14)
        upper_band = hl_avg + (2 * atr)
        lower_band = hl_avg - (2 * atr)
        features["supertrend_signal"] = np.where(
            df["close"] > upper_band, 1, np.where(df["close"] < lower_band, -1, 0)
        )

        return features

    def calculate_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features"""
        features = pd.DataFrame(index=df.index)

        # Bid-ask spread (estimated from high-low)
        features["bid_ask_spread"] = 2 * (df["high"] - df["low"]) / (df["high"] + df["low"])

        # Order flow imbalance (estimated from close position in range)
        features["order_flow_imbalance"] = (2 * df["close"] - df["high"] - df["low"]) / (
            df["high"] - df["low"]
        ).replace(0, np.nan)

        # Tick rule (price direction)
        features["tick_rule"] = np.sign(df["close"].diff())

        # Effective spread (Roll's measure)
        price_changes = df["close"].diff()
        features["effective_spread"] = 2 * np.sqrt(
            np.abs(price_changes.rolling(20).cov(price_changes.shift(1)))
        )

        # Price impact (Kyle's lambda approximation)
        returns = df["close"].pct_change()
        volume_sqrt = np.sqrt(df["volume"])
        features["price_impact"] = returns.rolling(20).std() / volume_sqrt.rolling(
            20
        ).mean().replace(0, np.nan)

        return features

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all features from OHLCV data.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")

        # Calculate features by category
        feature_dfs = []

        # Price features
        price_features = self.calculate_price_features(df)
        feature_dfs.append(price_features)
        logger.debug(f"Created {len(price_features.columns)} price features")

        # Volume features
        if "volume" in df.columns:
            volume_features = self.calculate_volume_features(df)
            feature_dfs.append(volume_features)
            logger.debug(f"Created {len(volume_features.columns)} volume features")

        # Momentum features
        momentum_features = self.calculate_momentum_features(df)
        feature_dfs.append(momentum_features)
        logger.debug(f"Created {len(momentum_features.columns)} momentum features")

        # Volatility features
        volatility_features = self.calculate_volatility_features(df)
        feature_dfs.append(volatility_features)
        logger.debug(f"Created {len(volatility_features.columns)} volatility features")

        # Trend features
        trend_features = self.calculate_trend_features(df)
        feature_dfs.append(trend_features)
        logger.debug(f"Created {len(trend_features.columns)} trend features")

        # Market microstructure features
        micro_features = self.calculate_microstructure_features(df)
        feature_dfs.append(micro_features)
        logger.debug(f"Created {len(micro_features.columns)} microstructure features")

        # Combine all features
        all_features = pd.concat(feature_dfs, axis=1)

        # Handle missing values
        all_features = self.handle_missing_values(all_features)

        # Remove low variance features
        all_features = self.remove_low_variance_features(all_features)

        # Remove highly correlated features
        all_features = self.remove_correlated_features(all_features)

        # Select top K features if we have too many
        if len(all_features.columns) > self.config.max_features:
            all_features = self.select_top_features(all_features, df)

        # Scale features
        all_features = self.scale_features(all_features)

        # Store selected features
        self.selected_features = list(all_features.columns)

        logger.info(f"Feature engineering complete: {len(all_features.columns)} features")

        return all_features

    def handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        # Forward fill for time series continuity
        features = features.fillna(method="ffill", limit=5)

        # Fill remaining with median
        features = features.fillna(features.median())

        # Drop rows with too many missing values
        thresh = len(features.columns) * 0.5  # Need at least 50% non-null
        features = features.dropna(thresh=thresh)

        return features

    def remove_low_variance_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Remove features with low variance"""
        if len(features) == 0:
            return features

        selector = VarianceThreshold(threshold=self.config.min_variance)

        # Fit and transform
        features_array = selector.fit_transform(features.fillna(0))

        # Get selected feature names
        selected_features = features.columns[selector.get_support()]

        logger.debug(
            f"Removed {len(features.columns) - len(selected_features)} low variance features"
        )

        return pd.DataFrame(features_array, index=features.index, columns=selected_features)

    def remove_correlated_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features"""
        if len(features.columns) <= 1:
            return features

        # Calculate correlation matrix
        corr_matrix = features.corr().abs()

        # Find highly correlated pairs
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features to drop
        to_drop = [
            column
            for column in upper_tri.columns
            if any(upper_tri[column] > self.config.correlation_threshold)
        ]

        logger.debug(f"Removing {len(to_drop)} highly correlated features")

        return features.drop(columns=to_drop)

    def select_top_features(
        self, features: pd.DataFrame, original_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Select top K features based on importance"""

        # Create target variable (next day return)
        target = original_df["close"].pct_change().shift(-1)
        target = (target > 0).astype(int)  # Binary classification

        # Align features and target
        common_index = features.index.intersection(target.index)
        features_aligned = features.loc[common_index]
        target_aligned = target.loc[common_index].dropna()

        # Further align after dropping NaN
        common_index = features_aligned.index.intersection(target_aligned.index)
        features_aligned = features_aligned.loc[common_index]
        target_aligned = target_aligned.loc[common_index]

        if len(features_aligned) == 0:
            return features

        # Use Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(features_aligned.fillna(0), target_aligned)

        # Get feature importance
        importance = pd.Series(rf.feature_importances_, index=features.columns).sort_values(
            ascending=False
        )

        # Store importance scores
        self.feature_importance = importance.to_dict()

        # Select top features
        top_features = importance.head(self.config.max_features).index.tolist()

        logger.info(f"Selected top {len(top_features)} features based on importance")

        return features[top_features]

    def scale_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Scale features for ML models"""
        if self.config.use_robust_scaler:
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        # Fit and transform
        scaled_array = self.scaler.fit_transform(features.fillna(0))

        return pd.DataFrame(scaled_array, index=features.index, columns=features.columns)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        if not self.feature_importance:
            return pd.DataFrame()

        importance_df = pd.DataFrame.from_dict(
            self.feature_importance, orient="index", columns=["importance"]
        ).sort_values("importance", ascending=False)

        # Add feature category
        importance_df["category"] = importance_df.index.map(self._get_feature_category)

        return importance_df

    def _get_feature_category(self, feature_name: str) -> str:
        """Get category for a feature"""
        if "return" in feature_name or "price" in feature_name:
            return FeatureCategory.PRICE.value
        elif "volume" in feature_name or "obv" in feature_name or "vwap" in feature_name:
            return FeatureCategory.VOLUME.value
        elif "rsi" in feature_name or "macd" in feature_name or "stoch" in feature_name:
            return FeatureCategory.MOMENTUM.value
        elif "atr" in feature_name or "bollinger" in feature_name or "volatility" in feature_name:
            return FeatureCategory.VOLATILITY.value
        elif "adx" in feature_name or "ema" in feature_name or "sma" in feature_name:
            return FeatureCategory.TREND.value
        else:
            return FeatureCategory.MARKET_MICROSTRUCTURE.value

    def validate_features(self, features: pd.DataFrame) -> dict[str, Any]:
        """Validate feature quality"""
        validation_results = {
            "total_features": len(features.columns),
            "missing_ratio": features.isnull().sum().sum() / features.size,
            "low_variance_features": [],
            "highly_correlated_pairs": [],
            "feature_stats": {},
        }

        # Check for low variance
        for col in features.columns:
            if features[col].var() < self.config.min_variance:
                validation_results["low_variance_features"].append(col)

        # Check correlation
        corr_matrix = features.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        high_corr = np.where(upper_tri > self.config.correlation_threshold)
        for i, j in zip(high_corr[0], high_corr[1], strict=False):
            pair = (features.columns[i], features.columns[j], upper_tri.iloc[i, j])
            validation_results["highly_correlated_pairs"].append(pair)

        # Feature statistics
        validation_results["feature_stats"] = {
            "mean": features.mean().to_dict(),
            "std": features.std().to_dict(),
            "min": features.min().to_dict(),
            "max": features.max().to_dict(),
        }

        return validation_results


def create_feature_pipeline(config: FeatureConfig | None = None) -> OptimizedFeatureEngineer:
    """Create feature engineering pipeline"""
    return OptimizedFeatureEngineer(config)


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf

    # Get sample data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="2y")
    df.columns = [c.lower() for c in df.columns]

    # Create feature engineer
    engineer = create_feature_pipeline()

    # Engineer features
    features = engineer.engineer_features(df)

    print(f"Created {len(features.columns)} features")
    print(f"Feature shape: {features.shape}")

    # Get importance
    importance = engineer.get_feature_importance()
    print("\nTop 10 most important features:")
    print(importance.head(10))

    # Validate features
    validation = engineer.validate_features(features)
    print("\nValidation results:")
    print(f"Missing ratio: {validation['missing_ratio']:.2%}")
    print(f"Low variance features: {len(validation['low_variance_features'])}")
    print(f"Highly correlated pairs: {len(validation['highly_correlated_pairs'])}")
