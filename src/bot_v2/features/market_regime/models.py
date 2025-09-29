"""
ML models for market regime detection - LOCAL to this slice.

Simplified implementations for isolation - in production would use
proper libraries like sklearn, statsmodels, or specialized HMM packages.
"""

import numpy as np

from .types import MarketRegime, TrendRegime, VolatilityRegime


class HMMRegimeDetector:
    """
    Hidden Markov Model for regime detection.

    Simplified implementation - in production use specialized HMM library.
    """

    def __init__(self, n_states: int = 3, n_iterations: int = 100) -> None:
        self.n_states = n_states
        self.n_iterations = n_iterations

        # Model parameters
        self.transition_matrix = None  # State transition probabilities
        self.emission_means = None  # Mean returns for each state
        self.emission_vars = None  # Variance for each state
        self.initial_probs = None  # Initial state probabilities

        self.is_fitted = False

    def fit(self, returns: np.ndarray) -> None:
        """
        Fit HMM to return data using Baum-Welch algorithm.

        Simplified implementation.
        """
        len(returns)

        # Initialize parameters randomly
        self.transition_matrix = np.random.dirichlet(np.ones(self.n_states), self.n_states)
        self.emission_means = np.random.uniform(-0.02, 0.02, self.n_states)
        self.emission_vars = np.random.uniform(0.001, 0.01, self.n_states)
        self.initial_probs = np.ones(self.n_states) / self.n_states

        # Simplified EM algorithm
        for iteration in range(self.n_iterations):
            # E-step: Forward-backward algorithm (simplified)
            alpha, beta = self._forward_backward(returns)

            # M-step: Update parameters
            self._update_parameters(returns, alpha, beta)

            # Early stopping if converged (simplified check)
            if iteration > 10 and iteration % 10 == 0:
                break

        self.is_fitted = True

    def predict_regime(self, returns: np.ndarray) -> np.ndarray:
        """Predict regime sequence using Viterbi algorithm."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self._viterbi(returns)

    def get_current_regime(self, recent_returns: np.ndarray) -> int:
        """Get most likely current regime."""
        if len(recent_returns) == 0:
            return 0

        # Use forward algorithm to get current state probabilities
        alpha, _ = self._forward_backward(recent_returns)
        return np.argmax(alpha[-1])

    def _forward_backward(self, returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward-backward algorithm (simplified)."""
        n_obs = len(returns)

        # Forward pass
        alpha = np.zeros((n_obs, self.n_states))
        alpha[0] = self.initial_probs * self._emission_prob(returns[0])

        for t in range(1, n_obs):
            for j in range(self.n_states):
                alpha[t, j] = (
                    np.sum(alpha[t - 1] * self.transition_matrix[:, j])
                    * self._emission_prob(returns[t])[j]
                )

        # Backward pass
        beta = np.zeros((n_obs, self.n_states))
        beta[-1] = 1

        for t in range(n_obs - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.transition_matrix[i] * self._emission_prob(returns[t + 1]) * beta[t + 1]
                )

        return alpha, beta

    def _emission_prob(self, observation: float) -> np.ndarray:
        """Calculate emission probabilities for observation."""
        probs = np.zeros(self.n_states)
        for i in range(self.n_states):
            # Gaussian emission probability
            var = max(self.emission_vars[i], 1e-6)  # Avoid division by zero
            probs[i] = np.exp(-0.5 * (observation - self.emission_means[i]) ** 2 / var) / np.sqrt(
                2 * np.pi * var
            )
        return probs

    def _update_parameters(self, returns: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> None:
        """Update model parameters (M-step)."""
        n_obs = len(returns)

        # Calculate gamma (state probabilities)
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        # Update emission parameters
        for i in range(self.n_states):
            weights = gamma[:, i]
            weight_sum = np.sum(weights)

            if weight_sum > 0:
                self.emission_means[i] = np.sum(weights * returns) / weight_sum
                self.emission_vars[i] = (
                    np.sum(weights * (returns - self.emission_means[i]) ** 2) / weight_sum
                )

        # Update transition matrix (simplified)
        for i in range(self.n_states):
            for j in range(self.n_states):
                numerator = 0
                denominator = 0
                for t in range(n_obs - 1):
                    numerator += gamma[t, i] * gamma[t + 1, j]
                    denominator += gamma[t, i]

                if denominator > 0:
                    self.transition_matrix[i, j] = numerator / denominator

    def _viterbi(self, returns: np.ndarray) -> np.ndarray:
        """Viterbi algorithm for most likely state sequence."""
        n_obs = len(returns)

        # Initialize
        delta = np.zeros((n_obs, self.n_states))
        psi = np.zeros((n_obs, self.n_states), dtype=int)

        delta[0] = np.log(self.initial_probs) + np.log(self._emission_prob(returns[0]))

        # Forward pass
        for t in range(1, n_obs):
            for j in range(self.n_states):
                trans_scores = delta[t - 1] + np.log(self.transition_matrix[:, j])
                psi[t, j] = np.argmax(trans_scores)
                delta[t, j] = np.max(trans_scores) + np.log(self._emission_prob(returns[t])[j])

        # Backward pass
        states = np.zeros(n_obs, dtype=int)
        states[-1] = np.argmax(delta[-1])

        for t in range(n_obs - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states


class GARCHVolatilityModel:
    """
    GARCH model for volatility regime detection.

    Simplified implementation.
    """

    def __init__(self, p: int = 1, q: int = 1) -> None:
        self.p = p  # ARCH terms
        self.q = q  # GARCH terms

        # Parameters: omega, alpha, beta
        self.omega = 0.0001
        self.alpha = np.array([0.1])
        self.beta = np.array([0.8])

        self.is_fitted = False

    def fit(self, returns: np.ndarray) -> None:
        """Fit GARCH model to returns."""
        # Simplified parameter estimation
        # In production, use Maximum Likelihood Estimation

        # Initial estimates
        mean_return = np.mean(returns)
        residuals = returns - mean_return

        # Estimate unconditional variance
        unconditional_var = np.var(residuals)

        # Simple moment-based estimation
        self.omega = unconditional_var * 0.1
        self.alpha[0] = 0.1
        self.beta[0] = 0.8

        self.is_fitted = True

    def predict_volatility(self, returns: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Predict volatility for next periods."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Calculate conditional variances
        conditional_vars = self._calculate_conditional_variance(returns)

        # Forecast volatility
        forecasts = []
        last_var = conditional_vars[-1]
        last_return = returns[-1]

        for h in range(horizon):
            if h == 0:
                # One-step ahead forecast
                next_var = self.omega + self.alpha[0] * last_return**2 + self.beta[0] * last_var
            else:
                # Multi-step ahead (simplified)
                next_var = self.omega + (self.alpha[0] + self.beta[0]) * last_var
                last_var = next_var

            forecasts.append(np.sqrt(next_var * 252))  # Annualized volatility

        return np.array(forecasts)

    def classify_volatility_regime(self, current_vol: float) -> VolatilityRegime:
        """Classify volatility level into regime."""
        if current_vol < 0.15:
            return VolatilityRegime.LOW
        elif current_vol < 0.25:
            return VolatilityRegime.MEDIUM
        elif current_vol < 0.40:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME

    def _calculate_conditional_variance(self, returns: np.ndarray) -> np.ndarray:
        """Calculate conditional variance series."""
        n = len(returns)
        conditional_vars = np.zeros(n)

        # Initialize with unconditional variance
        conditional_vars[0] = np.var(returns)

        for t in range(1, n):
            conditional_vars[t] = (
                self.omega
                + self.alpha[0] * returns[t - 1] ** 2
                + self.beta[0] * conditional_vars[t - 1]
            )

        return conditional_vars


class RegimeEnsemble:
    """
    Ensemble of different regime detection models.

    Combines HMM, GARCH, and heuristic approaches.
    """

    def __init__(self) -> None:
        self.hmm_model = HMMRegimeDetector(n_states=4)
        self.garch_model = GARCHVolatilityModel()
        self.weights = {"hmm": 0.4, "garch": 0.3, "heuristic": 0.3}
        self.is_fitted = False

    def fit(self, returns: np.ndarray, prices: np.ndarray) -> None:
        """Fit all models in ensemble."""
        # Fit HMM to returns
        self.hmm_model.fit(returns)

        # Fit GARCH to returns
        self.garch_model.fit(returns)

        self.is_fitted = True

    def predict_regime(self, returns: np.ndarray, prices: np.ndarray) -> dict[str, any]:
        """Predict regime using ensemble approach."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted first")

        # HMM prediction
        hmm_regime = self.hmm_model.get_current_regime(returns[-20:])

        # GARCH volatility prediction
        garch_vol = self.garch_model.predict_volatility(returns[-20:], horizon=1)[0]
        vol_regime = self.garch_model.classify_volatility_regime(garch_vol)

        # Heuristic trend detection
        trend_regime = self._detect_trend_heuristic(prices[-60:])

        # Combine predictions
        final_regime = self._combine_predictions(hmm_regime, vol_regime, trend_regime)

        return {
            "regime": final_regime,
            "hmm_state": hmm_regime,
            "volatility_regime": vol_regime,
            "trend_regime": trend_regime,
            "predicted_volatility": garch_vol,
            "confidence": self._calculate_ensemble_confidence(hmm_regime, vol_regime, trend_regime),
        }

    def _detect_trend_heuristic(self, prices: np.ndarray) -> TrendRegime:
        """Simple trend detection using moving averages."""
        if len(prices) < 20:
            return TrendRegime.SIDEWAYS

        ma_5 = np.mean(prices[-5:])
        ma_20 = np.mean(prices[-20:])
        ma_60 = np.mean(prices[-60:]) if len(prices) >= 60 else ma_20

        # Calculate trend strength
        trend_5_20 = (ma_5 - ma_20) / ma_20
        trend_20_60 = (ma_20 - ma_60) / ma_60

        combined_trend = (trend_5_20 + trend_20_60) / 2

        # Classify trend
        if combined_trend > 0.05:
            return TrendRegime.STRONG_UPTREND if combined_trend > 0.15 else TrendRegime.UPTREND
        elif combined_trend < -0.05:
            return TrendRegime.STRONG_DOWNTREND if combined_trend < -0.15 else TrendRegime.DOWNTREND
        else:
            return TrendRegime.SIDEWAYS

    def _combine_predictions(
        self, hmm_state: int, vol_regime: VolatilityRegime, trend_regime: TrendRegime
    ) -> MarketRegime:
        """Combine individual model predictions into final regime."""

        # Map HMM states to market conditions (simplified)
        hmm_regimes = {0: "bull", 1: "sideways", 2: "bear", 3: "volatile"}

        hmm_regimes.get(hmm_state, "sideways")

        # Determine primary trend
        if trend_regime in [TrendRegime.STRONG_UPTREND, TrendRegime.UPTREND]:
            trend_signal = "bull"
        elif trend_regime in [TrendRegime.STRONG_DOWNTREND, TrendRegime.DOWNTREND]:
            trend_signal = "bear"
        else:
            trend_signal = "sideways"

        # Determine volatility level
        vol_high = vol_regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]

        # Crisis detection
        if vol_regime == VolatilityRegime.EXTREME and trend_signal == "bear":
            return MarketRegime.CRISIS

        # Combine signals
        if trend_signal == "bull":
            return MarketRegime.BULL_VOLATILE if vol_high else MarketRegime.BULL_QUIET
        elif trend_signal == "bear":
            return MarketRegime.BEAR_VOLATILE if vol_high else MarketRegime.BEAR_QUIET
        else:  # sideways
            return MarketRegime.SIDEWAYS_VOLATILE if vol_high else MarketRegime.SIDEWAYS_QUIET

    def _calculate_ensemble_confidence(
        self, hmm_state: int, vol_regime: VolatilityRegime, trend_regime: TrendRegime
    ) -> float:
        """Calculate confidence in ensemble prediction."""
        confidence = 0.5

        # Check agreement between models
        # This is simplified - in practice would be more sophisticated

        # Strong trend increases confidence
        if trend_regime in [TrendRegime.STRONG_UPTREND, TrendRegime.STRONG_DOWNTREND]:
            confidence += 0.2

        # Clear volatility regime increases confidence
        if vol_regime in [VolatilityRegime.LOW, VolatilityRegime.EXTREME]:
            confidence += 0.15

        # HMM confidence (simplified)
        confidence += 0.1

        return min(confidence, 1.0)
