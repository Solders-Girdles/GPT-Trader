from __future__ import annotations

import numpy as np
from scipy import stats


class RobustConfidenceIntervals:
    """Calculate robust confidence intervals for financial data."""

    def __init__(self, block_size: int = 20) -> None:
        self.block_size = int(block_size)

    def normal_ci(self, returns: np.ndarray, confidence_level: float = 0.95) -> tuple[float, float]:
        returns = np.asarray(returns, dtype=float)
        if returns.size == 0:
            return (0.0, 0.0)
        mean_return = float(np.mean(returns))
        std_error = float(np.std(returns, ddof=1) / np.sqrt(max(len(returns), 1)))
        z_score = float(stats.norm.ppf((1 + confidence_level) / 2))
        margin = z_score * std_error
        return (mean_return - margin, mean_return + margin)

    def block_bootstrap_ci(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        n_bootstrap: int = 10_000,
    ) -> tuple[float, float]:
        returns = np.asarray(returns, dtype=float)
        if returns.size == 0:
            return (0.0, 0.0)
        block = max(self.block_size, 1)
        n_blocks = max(len(returns) // block, 1)
        bootstrap_means = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            idx = np.random.choice(n_blocks, size=n_blocks, replace=True)
            sample = []
            for b in idx:
                s = b * block
                e = s + block
                sample.extend(returns[s:e])
            bootstrap_means[i] = float(np.mean(sample)) if sample else 0.0
        alpha = 1 - confidence_level
        lo = float(np.percentile(bootstrap_means, (alpha / 2) * 100))
        hi = float(np.percentile(bootstrap_means, (1 - alpha / 2) * 100))
        return (lo, hi)

    def har_volatility_ci(
        self, returns: np.ndarray, confidence_level: float = 0.95
    ) -> tuple[float, float]:
        returns = np.asarray(returns, dtype=float)
        if returns.size == 0:
            return (0.0, 0.0)
        vol_5 = (
            float(np.std(returns[-5:], ddof=1))
            if len(returns) >= 5
            else float(np.std(returns, ddof=1))
        )
        vol_22 = float(np.std(returns[-22:], ddof=1)) if len(returns) >= 22 else vol_5
        vol_252 = float(np.std(returns[-252:], ddof=1)) if len(returns) >= 252 else vol_22
        har_vol = 0.1 + 0.4 * vol_5 + 0.3 * vol_22 + 0.2 * vol_252
        z_score = float(stats.norm.ppf((1 + confidence_level) / 2))
        margin = z_score * har_vol / np.sqrt(max(len(returns), 1))
        mean_return = float(np.mean(returns))
        return (mean_return - margin, mean_return + margin)

    def compare_ci_methods(
        self, returns: np.ndarray, confidence_level: float = 0.95
    ) -> dict[str, tuple[float, float]]:
        return {
            "normal_ci": self.normal_ci(returns, confidence_level),
            "block_bootstrap_ci": self.block_bootstrap_ci(returns, confidence_level),
            "har_volatility_ci": self.har_volatility_ci(returns, confidence_level),
        }
