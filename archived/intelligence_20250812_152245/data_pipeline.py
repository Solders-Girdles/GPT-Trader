from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


class LeakageFreePipeline:
    """Pipeline that prevents data leakage by fitting on train-only folds."""

    def __init__(self, n_splits: int = 5) -> None:
        self.tscv = TimeSeriesSplit(n_splits=int(n_splits))
        self.scaler = StandardScaler()
        self.feature_columns: list[str] | None = None
        self.is_fitted: bool = False

    def fit_transform(self, data: pd.DataFrame, target: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        if data.empty:
            return np.empty((0, 0)), np.empty((0,))
        self.feature_columns = [c for c in data.columns if c.startswith("feature_")]
        if not self.feature_columns:
            # Fallback: use all numeric columns except target if features not pre-named
            self.feature_columns = [c for c in data.select_dtypes(include=[np.number]).columns]

        # Fit on first fold's train indices
        train_idx, _ = next(self.tscv.split(data))
        train_data = data.iloc[train_idx][self.feature_columns]
        self.scaler.fit(train_data)
        self.is_fitted = True

        x_all = self.scaler.transform(data[self.feature_columns])
        y_all = target.values if hasattr(target, "values") else np.asarray(target)
        return x_all, y_all

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        if self.feature_columns is None:
            raise ValueError("Feature columns are not set")
        return self.scaler.transform(data[self.feature_columns])

    def save_pipeline(self, path: str) -> None:
        state = {
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(state, path)

    def load_pipeline(self, path: str) -> None:
        state = joblib.load(path)
        self.scaler = state["scaler"]
        self.feature_columns = state["feature_columns"]
        self.is_fitted = state["is_fitted"]


class FeatureRegistry:
    """Lightweight feature registry with schema and transformation hooks.

    Registers feature builders by name. Each builder takes a DataFrame and returns a DataFrame
    with one or more feature_* columns. Provides a compose() helper to assemble feature sets.
    """

    def __init__(self) -> None:
        self._builders: dict[str, Any] = {}

    def register(self, name: str, fn: Any) -> None:
        self._builders[name] = fn

    def available(self) -> list[str]:
        return sorted(self._builders.keys())

    def compose(self, builders: Iterable[str], df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for b in builders:
            if b in self._builders:
                add = self._builders[b](out)
                if isinstance(add, pd.DataFrame):
                    out = pd.concat([out, add], axis=1)
        return out


def build_basic_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["Close"].astype(float)
    out["feature_ret_1d"] = close.pct_change().fillna(0.0)
    out["feature_vol_%dd" % window] = close.pct_change().rolling(window).std().fillna(0.0)
    out["feature_ma_gap_%dd" % window] = (close / close.rolling(window).mean() - 1.0).fillna(0.0)
    return out


def build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["Close"].astype(float)
    out["feature_trend"] = (close > close.rolling(50).mean()).astype(float).fillna(0.0)
    out["feature_vol_regime"] = (
        (close.pct_change().rolling(20).std() > 0.02).astype(float).fillna(0.0)
    )
    return out


# --- Expanded feature builders ---


def build_return_features(
    df: pd.DataFrame, windows: Iterable[int] = (1, 5, 10, 20)
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df.get("Close")
    if c is None:
        return out
    c = c.astype(float)
    for w in windows:
        out[f"feature_ret_{w}d"] = c.pct_change(w).fillna(0.0)
    out["feature_logret_1d"] = (
        (c / c.shift(1)).apply(lambda x: 0.0 if x <= 0 else float(np.log(x))).fillna(0.0)
    )
    return out


def _true_range(df: pd.DataFrame) -> pd.Series:
    high = df.get("High").astype(float)
    low = df.get("Low").astype(float)
    close = df.get("Close").astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.fillna(0.0)


def build_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df.get("Close")
    if c is None:
        return out
    c = c.astype(float)
    for w in (10, 20, 60):
        out[f"feature_vol_{w}d"] = c.pct_change().rolling(w).std().fillna(0.0)
    # ATR and True Range
    tr = _true_range(df)
    out["feature_tr"] = tr
    out["feature_atr_14"] = tr.rolling(14).mean().fillna(0.0)
    # Parkinson volatility
    if "High" in df.columns and "Low" in df.columns:
        hl = (np.log(df["High"].astype(float) / df["Low"].astype(float)).pow(2)).rolling(20).mean()
        park = np.sqrt(hl / (4.0 * np.log(2.0)))
        out["feature_park_vol_20"] = park.fillna(0.0)
    return out


def build_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df.get("Close")
    if c is None:
        return out
    c = c.astype(float)
    for w in (20, 50, 200):
        ma = c.rolling(w).mean()
        out[f"feature_ma_gap_{w}"] = (c / ma - 1.0).fillna(0.0)
    # Bollinger z-score (20, 2)
    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    z = (c - ma20) / sd20.replace(0, np.nan)
    out["feature_bollinger_z_20_2"] = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


def build_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df.get("Close")
    if c is None:
        return out
    c = c.astype(float)
    # RSI (14)
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = (up / down).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    out["feature_rsi_14"] = rsi.fillna(0.0)
    # ROC
    for w in (5, 10, 20):
        out[f"feature_roc_{w}"] = (c / c.shift(w) - 1.0).fillna(0.0)
    # MACD (12, 26, 9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["feature_macd_line"] = macd.fillna(0.0)
    out["feature_macd_signal"] = signal.fillna(0.0)
    out["feature_macd_hist"] = (macd - signal).fillna(0.0)
    return out


def build_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    v = df.get("Volume")
    c = df.get("Close")
    if v is None or c is None:
        return out
    v = v.astype(float)
    c = c.astype(float)
    # Z-score volume
    v_ma = v.rolling(20).mean()
    v_sd = v.rolling(20).std().replace(0, np.nan)
    out["feature_vol_z_20"] = ((v - v_ma) / v_sd).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    # Dollar volume
    out["feature_dollar_vol"] = (v * c).fillna(0.0)
    # OBV
    direction = np.sign(c.diff().fillna(0.0))
    obv = (direction * v).cumsum().fillna(0.0)
    out["feature_obv"] = obv
    # PVT
    out["feature_pvt"] = ((c.diff() / c.shift(1)).fillna(0.0) * v).cumsum().fillna(0.0)
    return out


def build_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    high = df.get("High")
    low = df.get("Low")
    open_price = df.get("Open")
    close = df.get("Close")
    if any(x is None for x in (h, l, o, c)):
        return out
    high = high.astype(float)
    low = low.astype(float)
    open_price = open_price.astype(float)
    close = close.astype(float)
    out["feature_range_pct"] = (
        ((high - low) / close.replace(0, np.nan)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    )
    prev_close = close.shift(1)
    out["feature_gap_pct"] = (
        ((open_price - prev_close) / prev_close.replace(0, np.nan))
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )
    out["feature_hc_gap"] = (
        ((high - close) / close.replace(0, np.nan)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    )
    out["feature_lc_gap"] = (
        ((low - close) / close.replace(0, np.nan)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    )
    return out


def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if not isinstance(df.index, pd.DatetimeIndex):
        return out
    idx = df.index
    dow = idx.weekday
    mon = idx.month
    # Cyclical encoding for dow/month
    out["feature_dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["feature_dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["feature_mon_sin"] = np.sin(2 * np.pi * mon / 12.0)
    out["feature_mon_cos"] = np.cos(2 * np.pi * mon / 12.0)
    return out
