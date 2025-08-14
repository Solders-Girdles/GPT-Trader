from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegimeLabel:
    """Programmatic regime label with confidence."""

    regime_type: str
    confidence: float
    label_method: str  # 'rule_based', 'hmm', 'manual'
    features_used: list[str]
    timestamp: pd.Timestamp


class RegimeLabeler:
    """Create programmatic regime labels for benchmarking.

    Note: HMM-based labeling requires `hmmlearn` at inference. We import lazily
    to avoid import-time dependency errors when this module is not used.
    """

    def __init__(self, n_components: int = 4, random_state: int = 42) -> None:
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self._hmm_model = None

    def create_rule_based_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create simple rule-based regime labels from OHLC data.

        Expects columns including 'close'.
        """
        if data.empty or "close" not in data.columns:
            return pd.Series(dtype=str)

        returns = data["close"].pct_change().dropna()
        volatility = returns.rolling(20).std()
        momentum = data["close"].pct_change(20)

        labels = pd.Series(index=data.index, dtype=str)

        high_vol = volatility > volatility.quantile(0.8)
        low_vol = volatility < volatility.quantile(0.2)
        strong_trend = momentum.abs() > momentum.abs().quantile(0.8)

        labels[high_vol & strong_trend] = "volatile_trending"
        labels[high_vol & ~strong_trend] = "volatile_sideways"
        labels[low_vol & strong_trend] = "stable_trending"
        labels[low_vol & ~strong_trend] = "stable_sideways"

        return labels

    def _ensure_hmm(self) -> None:
        if self._hmm_model is None:
            from hmmlearn import hmm  # Lazy import

            self._hmm_model = hmm.GaussianHMM(
                n_components=self.n_components, random_state=self.random_state
            )

    def create_hmm_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create HMM-based regime labels using returns/vol features.

        Expects 'close' column.
        """
        if data.empty or "close" not in data.columns:
            return pd.Series(dtype=str)

        returns = data["close"].pct_change().dropna()
        vol20 = returns.rolling(20).std()
        mean20 = returns.rolling(20).mean()
        # Align lengths
        min_len = min(len(returns), len(vol20), len(mean20))
        r = returns.tail(min_len).values
        v = vol20.tail(min_len).values
        m = mean20.tail(min_len).values
        features = np.column_stack([r, v, m])

        self._ensure_hmm()
        self._hmm_model.fit(features)
        states = self._hmm_model.predict(features)

        regime_map = {
            0: "low_vol_regime",
            1: "high_vol_regime",
            2: "trending_regime",
            3: "mean_reversion_regime",
        }
        labeled = [regime_map.get(int(s), "unknown") for s in states]
        return pd.Series(labeled, index=returns.tail(min_len).index)

    def calculate_stability_metrics(
        self, predictions: pd.Series, labels: pd.Series
    ) -> dict[str, float]:
        """Calculate regime stability metrics between predicted and labeled regimes."""
        if predictions.empty:
            return {"avg_regime_duration": 0.0, "regime_entropy": 0.0, "transition_stability": 0.0}

        regime_changes = (predictions != predictions.shift()).sum()
        avg_regime_duration = float(len(predictions) / (regime_changes + 1))

        # Transition matrix stability
        transition_matrix = pd.crosstab(predictions.shift(), predictions, normalize="index")
        mat = transition_matrix.fillna(0.0).values
        regime_entropy = float(-np.sum(mat * np.log(mat + 1e-10)))

        return {
            "avg_regime_duration": avg_regime_duration,
            "regime_entropy": regime_entropy,
            "transition_stability": float(1 - mat.std()),
        }
