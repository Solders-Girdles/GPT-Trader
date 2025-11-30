"""
Market regime detector using O(1) incremental algorithms.

Classifies market conditions into regimes (bull/bear/sideways x quiet/volatile)
using stateful indicators that update in constant time per tick.

Key algorithms:
- Trend: EMA slope and price position relative to long EMA
- Volatility: ATR percentile vs historical distribution
- Momentum: Rate of change over lookback window
- Crisis: Extreme volatility or drawdown detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from gpt_trader.features.live_trade.stateful_indicators import (
    OnlineEMA,
    RollingWindow,
    WelfordState,
)

from .indicators import (
    OnlineATR,
    OnlineBollingerBands,
    OnlineTrendStrength,
    RegimeTransitionMatrix,
)
from .models import RegimeConfig, RegimeState, RegimeType


@dataclass
class _SymbolRegimeState:
    """Per-symbol stateful indicators for regime detection.

    All updates are O(1) using incremental algorithms.
    """

    config: RegimeConfig

    # Trend indicators (EMAs)
    short_ema: OnlineEMA = field(init=False)
    long_ema: OnlineEMA = field(init=False)
    ema_slope_window: RollingWindow = field(init=False)  # For slope calculation

    # Volatility tracking
    returns_welford: WelfordState = field(default_factory=WelfordState)
    volatility_history: RollingWindow = field(init=False)  # Historical vol distribution

    # Momentum tracking
    price_window: RollingWindow = field(init=False)

    # Advanced indicators
    atr: OnlineATR = field(init=False)
    bollinger: OnlineBollingerBands = field(init=False)
    trend_strength: OnlineTrendStrength = field(init=False)
    atr_history: RollingWindow = field(init=False)  # For ATR percentile calculation

    # Price tracking
    prev_price: Decimal | None = None
    peak_price: Decimal | None = None  # For drawdown calculation
    tick_count: int = 0

    # Regime persistence
    current_regime: RegimeType = RegimeType.UNKNOWN
    regime_ticks: int = 0
    pending_regime: RegimeType | None = None
    pending_ticks: int = 0

    def __post_init__(self) -> None:
        """Initialize indicators based on config."""
        self.short_ema = OnlineEMA(period=self.config.short_ema_period)
        self.long_ema = OnlineEMA(period=self.config.long_ema_period)
        self.ema_slope_window = RollingWindow(max_size=5)  # 5-tick slope window
        self.volatility_history = RollingWindow(max_size=self.config.volatility_lookback)
        self.price_window = RollingWindow(max_size=self.config.momentum_period)

        # Advanced indicators
        self.atr = OnlineATR(period=self.config.atr_period)
        self.bollinger = OnlineBollingerBands(period=self.config.short_ema_period)
        self.trend_strength = OnlineTrendStrength(period=self.config.atr_period)
        self.atr_history = RollingWindow(max_size=self.config.volatility_lookback)

    def update(self, price: Decimal) -> None:
        """Update all indicators with new price. O(1) complexity."""
        self.tick_count += 1

        # Update EMAs
        self.short_ema.update(price)
        long_val = self.long_ema.update(price)

        # Track EMA slope (for trend direction)
        if long_val is not None:
            self.ema_slope_window.add(long_val)

        # Update volatility (absolute return)
        if self.prev_price is not None and self.prev_price > 0:
            ret = (price - self.prev_price) / self.prev_price
            self.returns_welford.update(ret)

            # Store current volatility reading (using return magnitude)
            abs_ret = abs(ret)
            self.volatility_history.add(abs_ret)

        # Update momentum window
        self.price_window.add(price)

        # Update advanced indicators
        atr_value = self.atr.update(price)
        if atr_value is not None:
            self.atr_history.add(atr_value)
        self.bollinger.update(price)
        self.trend_strength.update(price)

        # Update peak for drawdown calculation
        if self.peak_price is None or price > self.peak_price:
            self.peak_price = price

        self.prev_price = price

    def get_trend_score(self) -> float:
        """Calculate trend score from -1 (bearish) to +1 (bullish).

        Uses:
        1. Price position relative to long EMA
        2. Short EMA vs long EMA
        3. EMA slope direction
        """
        if not self.long_ema.initialized or self.long_ema.value is None:
            return 0.0

        long_ema = float(self.long_ema.value)
        if long_ema == 0:
            return 0.0

        scores: list[float] = []

        # 1. Price vs long EMA (weight: 0.4)
        if self.prev_price is not None:
            price_deviation = (float(self.prev_price) - long_ema) / long_ema
            price_score = max(-1.0, min(1.0, price_deviation / self.config.trend_threshold))
            scores.append(price_score * 0.4)

        # 2. Short EMA vs Long EMA (weight: 0.4)
        if self.short_ema.initialized and self.short_ema.value is not None:
            ema_diff = (float(self.short_ema.value) - long_ema) / long_ema
            ema_score = max(-1.0, min(1.0, ema_diff / self.config.trend_threshold))
            scores.append(ema_score * 0.4)

        # 3. EMA slope (weight: 0.2)
        if len(self.ema_slope_window) >= 2:
            oldest = float(self.ema_slope_window[0])
            newest = float(self.ema_slope_window[-1])
            if oldest > 0:
                slope = (newest - oldest) / oldest
                slope_score = max(-1.0, min(1.0, slope / (self.config.trend_threshold / 5)))
                scores.append(slope_score * 0.2)

        return sum(scores) if scores else 0.0

    def get_volatility_percentile(self) -> float:
        """Calculate current volatility percentile vs historical.

        Returns value from 0.0 (lowest) to 1.0 (highest).
        """
        if self.returns_welford.count < 10:
            return 0.5  # Default to middle

        current_vol = float(self.returns_welford.std_dev)
        if current_vol == 0:
            return 0.5

        # Compare to historical distribution
        if len(self.volatility_history) < 10:
            return 0.5

        # Count how many historical values are below current
        count_below = sum(1 for v in self.volatility_history.values if float(v) < current_vol)

        return count_below / len(self.volatility_history)

    def get_momentum_score(self) -> float:
        """Calculate momentum score from -1 to +1.

        Uses rate of change over momentum window.
        """
        if len(self.price_window) < 2:
            return 0.0

        oldest = float(self.price_window[0])
        newest = float(self.price_window[-1])

        if oldest == 0:
            return 0.0

        roc = (newest - oldest) / oldest

        # Normalize to -1 to +1 range (assuming 5% max expected change)
        return max(-1.0, min(1.0, roc / 0.05))

    def get_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_price is None or self.prev_price is None:
            return 0.0
        if self.peak_price == 0:
            return 0.0

        return float((self.peak_price - self.prev_price) / self.peak_price)

    def get_atr_percentile(self) -> float:
        """Get current ATR percentile vs historical.

        Returns value from 0.0 (lowest vol) to 1.0 (highest vol).
        More robust than return-based volatility for regime detection.
        """
        if not self.atr.is_ready or self.atr.value is None:
            return 0.5  # Default to middle

        current_atr = float(self.atr.value)
        if len(self.atr_history) < 10:
            return 0.5

        # Count how many historical ATR values are below current
        count_below = sum(1 for v in self.atr_history.values if float(v) < current_atr)
        return count_below / len(self.atr_history)

    def get_adx(self) -> float | None:
        """Get current ADX (trend strength) value.

        Returns 0-100 scale:
        - 0-25: Weak/no trend (ranging)
        - 25-50: Trending
        - 50-75: Strong trend
        - 75-100: Very strong trend
        """
        if not self.trend_strength.is_ready:
            return None

        result = self.trend_strength.update(self.prev_price or Decimal("0"))
        return result.get("adx")

    def get_squeeze_score(self) -> float:
        """Get Bollinger Band squeeze score.

        Returns 0.0-1.0:
        - High values indicate consolidation (squeeze)
        - Low values indicate expansion
        - Squeeze often precedes breakouts
        """
        if not self.bollinger.is_ready:
            return 0.5

        return self.bollinger.get_squeeze_score()

    def is_warmed_up(self) -> bool:
        """Check if enough data for valid regime detection."""
        return (
            self.long_ema.initialized
            and len(self.volatility_history) >= 20
            and self.tick_count >= self.config.long_ema_period
        )

    def serialize(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "short_ema": self.short_ema.serialize(),
            "long_ema": self.long_ema.serialize(),
            "ema_slope_window": self.ema_slope_window.serialize(),
            "returns_welford": self.returns_welford.serialize(),
            "volatility_history": self.volatility_history.serialize(),
            "price_window": self.price_window.serialize(),
            # Advanced indicators
            "atr": self.atr.serialize(),
            "bollinger": self.bollinger.serialize(),
            "trend_strength": self.trend_strength.serialize(),
            "atr_history": self.atr_history.serialize(),
            # Price tracking
            "prev_price": str(self.prev_price) if self.prev_price else None,
            "peak_price": str(self.peak_price) if self.peak_price else None,
            "tick_count": self.tick_count,
            "current_regime": self.current_regime.name,
            "regime_ticks": self.regime_ticks,
            "pending_regime": self.pending_regime.name if self.pending_regime else None,
            "pending_ticks": self.pending_ticks,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any], config: RegimeConfig) -> _SymbolRegimeState:
        """Restore state from serialized data."""
        state = cls(config=config)

        state.short_ema = OnlineEMA.deserialize(data["short_ema"])
        state.long_ema = OnlineEMA.deserialize(data["long_ema"])
        state.ema_slope_window = RollingWindow.deserialize(data["ema_slope_window"])
        state.returns_welford = WelfordState.deserialize(data["returns_welford"])
        state.volatility_history = RollingWindow.deserialize(data["volatility_history"])
        state.price_window = RollingWindow.deserialize(data["price_window"])

        # Restore advanced indicators (with backward compatibility)
        if "atr" in data:
            state.atr = OnlineATR.deserialize(data["atr"])
        if "bollinger" in data:
            state.bollinger = OnlineBollingerBands.deserialize(data["bollinger"])
        if "trend_strength" in data:
            state.trend_strength = OnlineTrendStrength.deserialize(data["trend_strength"])
        if "atr_history" in data:
            state.atr_history = RollingWindow.deserialize(data["atr_history"])

        state.prev_price = Decimal(data["prev_price"]) if data["prev_price"] else None
        state.peak_price = Decimal(data["peak_price"]) if data["peak_price"] else None
        state.tick_count = data["tick_count"]
        state.current_regime = RegimeType[data["current_regime"]]
        state.regime_ticks = data["regime_ticks"]
        state.pending_regime = (
            RegimeType[data["pending_regime"]] if data["pending_regime"] else None
        )
        state.pending_ticks = data["pending_ticks"]

        return state


class MarketRegimeDetector:
    """Market regime detector using O(1) incremental algorithms.

    Classifies market into regimes based on:
    - Trend direction (bull/bear/sideways)
    - Volatility level (quiet/volatile)
    - Crisis conditions (extreme volatility/drawdown)

    All operations are O(1) per price update for real-time use.

    Example:
        config = RegimeConfig()
        detector = MarketRegimeDetector(config)

        # Feed prices
        for price in prices:
            state = detector.update("BTC-USD", price)
            print(f"Regime: {state.regime}, Confidence: {state.confidence}")
    """

    def __init__(self, config: RegimeConfig | None = None):
        """Initialize detector with configuration.

        Args:
            config: Detection parameters. Uses defaults if None.
        """
        self.config = config or RegimeConfig()
        self._symbol_states: dict[str, _SymbolRegimeState] = {}
        # Per-symbol transition matrices for probabilistic forecasting
        self._transition_matrices: dict[str, RegimeTransitionMatrix] = {}

    def _get_or_create_state(self, symbol: str) -> _SymbolRegimeState:
        """Get or create symbol-specific state."""
        if symbol not in self._symbol_states:
            self._symbol_states[symbol] = _SymbolRegimeState(config=self.config)
            self._transition_matrices[symbol] = RegimeTransitionMatrix()
        return self._symbol_states[symbol]

    def _get_transition_matrix(self, symbol: str) -> RegimeTransitionMatrix:
        """Get transition matrix for symbol."""
        if symbol not in self._transition_matrices:
            self._transition_matrices[symbol] = RegimeTransitionMatrix()
        return self._transition_matrices[symbol]

    def update(self, symbol: str, price: Decimal) -> RegimeState:
        """Update regime detection with new price. O(1) complexity.

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            price: Current price

        Returns:
            Current regime state with confidence and metrics
        """
        state = self._get_or_create_state(symbol)
        state.update(price)

        # Not enough data yet
        if not state.is_warmed_up():
            return RegimeState.unknown()

        return self._classify_regime(state, symbol)

    def _classify_regime(self, state: _SymbolRegimeState, symbol: str | None = None) -> RegimeState:
        """Classify regime using statistical thresholds and advanced indicators."""
        # Calculate basic metrics
        trend_score = state.get_trend_score()
        volatility_percentile = state.get_volatility_percentile()
        momentum_score = state.get_momentum_score()
        drawdown = state.get_drawdown()

        # Calculate advanced metrics (use ATR percentile if available)
        atr_percentile = state.get_atr_percentile()
        adx = state.get_adx()
        squeeze_score = state.get_squeeze_score()

        # Use ATR-based volatility if available (more robust than return-based)
        if state.atr.is_ready:
            combined_vol = (volatility_percentile + atr_percentile) / 2
        else:
            combined_vol = volatility_percentile

        # Check for crisis conditions first
        is_crisis = self._check_crisis(state, combined_vol, drawdown)

        if is_crisis:
            new_regime = RegimeType.CRISIS
            confidence = 0.9
        else:
            # Normal regime classification with ADX enhancement
            new_regime = self._classify_normal_regime(trend_score, combined_vol, adx, squeeze_score)
            confidence = self._calculate_confidence(trend_score, combined_vol, adx, squeeze_score)

        # Apply regime persistence (smoothing)
        old_regime = state.current_regime
        final_regime, regime_age = self._apply_persistence(state, new_regime)

        # Update transition matrix if regime changed
        if symbol is not None and old_regime != final_regime:
            matrix = self._get_transition_matrix(symbol)
            matrix.record_transition(old_regime.name, final_regime.name)

        # Calculate transition probability (use matrix if available)
        transition_prob = self._estimate_transition_probability(state, new_regime, symbol)

        return RegimeState(
            regime=final_regime,
            confidence=confidence,
            trend_score=trend_score,
            volatility_percentile=combined_vol,
            momentum_score=momentum_score,
            regime_age_ticks=regime_age,
            transition_probability=transition_prob,
        )

    def _check_crisis(
        self,
        state: _SymbolRegimeState,
        volatility_percentile: float,
        drawdown: float,
    ) -> bool:
        """Check if market is in crisis conditions."""
        # High volatility crisis
        if volatility_percentile > 0.95:
            # Check if volatility is extreme (>3x normal)
            if state.returns_welford.count > 20:
                current_vol = float(state.returns_welford.std_dev)
                if state.volatility_history.mean is not None:
                    historical_vol = float(state.volatility_history.mean)
                    if (
                        historical_vol > 0
                        and current_vol > historical_vol * self.config.crisis_volatility_multiplier
                    ):
                        return True

        # Drawdown crisis
        if drawdown > self.config.crisis_drawdown_threshold:
            return True

        return False

    def _classify_normal_regime(
        self,
        trend_score: float,
        volatility_percentile: float,
        adx: float | None = None,
        squeeze_score: float = 0.5,
    ) -> RegimeType:
        """Classify into normal (non-crisis) regime using enhanced indicators.

        Args:
            trend_score: Trend direction score (-1 to 1)
            volatility_percentile: Combined volatility percentile (0 to 1)
            adx: ADX trend strength (0-100), None if not ready
            squeeze_score: Bollinger squeeze score (0-1), higher = tighter squeeze
        """
        # Determine trend direction using both trend score and ADX
        effective_threshold = self.config.trend_threshold

        # If ADX is available and shows weak trend, raise threshold
        if adx is not None:
            if adx < 20:  # Weak trend - require stronger signal
                effective_threshold *= 1.5
            elif adx > 40:  # Strong trend - lower threshold
                effective_threshold *= 0.7

        if trend_score > effective_threshold:
            trend = "bull"
        elif trend_score < -effective_threshold:
            trend = "bear"
        else:
            trend = "sideways"

        # Determine volatility level
        # High squeeze (consolidation) biases toward quiet regime
        vol_threshold_high = self.config.volatility_high_percentile
        vol_threshold_low = self.config.volatility_low_percentile

        # Adjust thresholds based on squeeze (consolidation detection)
        if squeeze_score > 0.7:  # Tight squeeze - market consolidating
            vol_threshold_high += 0.1  # Require higher vol to be "volatile"
        elif squeeze_score < 0.3:  # Expanding bands - market moving
            vol_threshold_low -= 0.1  # Lower threshold for "volatile"

        if volatility_percentile > vol_threshold_high:
            vol = "volatile"
        elif volatility_percentile < vol_threshold_low:
            vol = "quiet"
        else:
            # Middle ground - slight bias toward quiet for safety
            vol = "quiet"

        # Map to regime type
        regime_map = {
            ("bull", "quiet"): RegimeType.BULL_QUIET,
            ("bull", "volatile"): RegimeType.BULL_VOLATILE,
            ("bear", "quiet"): RegimeType.BEAR_QUIET,
            ("bear", "volatile"): RegimeType.BEAR_VOLATILE,
            ("sideways", "quiet"): RegimeType.SIDEWAYS_QUIET,
            ("sideways", "volatile"): RegimeType.SIDEWAYS_VOLATILE,
        }

        return regime_map[(trend, vol)]

    def _calculate_confidence(
        self,
        trend_score: float,
        volatility_percentile: float,
        adx: float | None = None,
        squeeze_score: float = 0.5,
    ) -> float:
        """Calculate confidence in regime classification.

        Higher confidence when signals are clear and ADX confirms trend strength.

        Args:
            trend_score: Trend direction score (-1 to 1)
            volatility_percentile: Combined volatility percentile (0 to 1)
            adx: ADX trend strength (0-100), None if not ready
            squeeze_score: Bollinger squeeze score (0-1)
        """
        # Trend clarity (how far from neutral)
        trend_clarity = abs(trend_score)

        # Volatility clarity (how far from middle percentile)
        vol_clarity = abs(volatility_percentile - 0.5) * 2

        # ADX boost: Higher ADX increases confidence in trend detection
        adx_boost = 0.0
        if adx is not None:
            # ADX > 25 indicates a trending market
            if adx > 40:
                adx_boost = 0.15  # Strong trend boost
            elif adx > 25:
                adx_boost = 0.08  # Moderate trend boost
            elif adx < 15:
                adx_boost = -0.1  # Weak/no trend penalty

        # Squeeze factor: High squeeze = less confident (breakout imminent)
        squeeze_penalty = 0.0
        if squeeze_score > 0.8:
            squeeze_penalty = -0.1  # Very tight squeeze = uncertainty

        # Combined confidence (weighted average)
        base_confidence = trend_clarity * 0.5 + vol_clarity * 0.3 + 0.2
        confidence = base_confidence + adx_boost + squeeze_penalty

        # Clamp to reasonable range
        return max(0.3, min(0.95, confidence))

    def _apply_persistence(
        self,
        state: _SymbolRegimeState,
        new_regime: RegimeType,
    ) -> tuple[RegimeType, int]:
        """Apply regime persistence to avoid whipsawing.

        Returns:
            Tuple of (final_regime, regime_age_ticks)
        """
        if new_regime == state.current_regime:
            # Same regime - increment age
            state.regime_ticks += 1
            state.pending_regime = None
            state.pending_ticks = 0
            return state.current_regime, state.regime_ticks

        # Different regime - check if should transition
        if state.pending_regime == new_regime:
            state.pending_ticks += 1
        else:
            state.pending_regime = new_regime
            state.pending_ticks = 1

        # Transition if pending long enough
        if state.pending_ticks >= self.config.min_regime_ticks:
            state.current_regime = new_regime
            state.regime_ticks = state.pending_ticks
            state.pending_regime = None
            state.pending_ticks = 0
            return new_regime, state.regime_ticks

        # Not enough confirmation - stay in current regime
        state.regime_ticks += 1
        return state.current_regime, state.regime_ticks

    def _estimate_transition_probability(
        self,
        state: _SymbolRegimeState,
        detected_regime: RegimeType,
        symbol: str | None = None,
    ) -> float:
        """Estimate probability of regime transition using historical data.

        Combines:
        - Current regime stability (how long in regime)
        - Pending transition progress
        - Historical transition probabilities from matrix

        Args:
            state: Symbol-specific regime state
            detected_regime: Currently detected regime (may differ from current)
            symbol: Symbol for looking up transition matrix
        """
        current_regime = state.current_regime

        if detected_regime == current_regime:
            # Stable regime - probability of staying decreases over time
            base_prob = max(0.05, 0.2 - state.regime_ticks * 0.01)

            # Use historical data if available
            if symbol is not None:
                matrix = self._get_transition_matrix(symbol)
                hist_dist = matrix.get_transition_distribution(current_regime.name)
                if hist_dist:
                    # Average with historical probability of any transition
                    hist_transition_prob = 1.0 - hist_dist.get(current_regime.name, 0.5)
                    return (base_prob + hist_transition_prob) / 2

            return base_prob

        if state.pending_regime is not None:
            # Building toward transition
            progress = state.pending_ticks / self.config.min_regime_ticks

            # Enhance with historical probability of this specific transition
            if symbol is not None:
                matrix = self._get_transition_matrix(symbol)
                hist_prob = matrix.get_transition_probability(
                    current_regime.name, detected_regime.name
                )
                if hist_prob > 0:
                    # Weight progress by historical likelihood
                    return min(0.95, progress * 0.7 + hist_prob * 0.3)

            return min(0.9, progress)

        return 0.3  # Some uncertainty

    def get_regime(self, symbol: str) -> RegimeState:
        """Get current regime for symbol without updating.

        Args:
            symbol: Trading symbol

        Returns:
            Current regime state (or unknown if no data)
        """
        if symbol not in self._symbol_states:
            return RegimeState.unknown()

        state = self._symbol_states[symbol]
        if not state.is_warmed_up():
            return RegimeState.unknown()

        return self._classify_regime(state)

    def serialize_state(self) -> dict[str, Any]:
        """Serialize all symbol states for persistence."""
        return {
            "config": self.config.to_dict(),
            "symbols": {symbol: state.serialize() for symbol, state in self._symbol_states.items()},
            "transition_matrices": {
                symbol: matrix.serialize() for symbol, matrix in self._transition_matrices.items()
            },
        }

    def deserialize_state(self, data: dict[str, Any]) -> None:
        """Restore state from serialized data."""
        # Config is already set, just restore symbol states
        symbols_data = data.get("symbols", {})
        for symbol, state_data in symbols_data.items():
            self._symbol_states[symbol] = _SymbolRegimeState.deserialize(state_data, self.config)

        # Restore transition matrices (with backward compatibility)
        matrices_data = data.get("transition_matrices", {})
        for symbol, matrix_data in matrices_data.items():
            self._transition_matrices[symbol] = RegimeTransitionMatrix.deserialize(matrix_data)

    def reset(self, symbol: str | None = None) -> None:
        """Reset state for a symbol or all symbols.

        Args:
            symbol: Symbol to reset, or None for all symbols
        """
        if symbol is not None:
            self._symbol_states.pop(symbol, None)
            self._transition_matrices.pop(symbol, None)
        else:
            self._symbol_states.clear()
            self._transition_matrices.clear()

    def get_transition_forecast(self, symbol: str) -> dict[str, float]:
        """Get forecasted transition probabilities for a symbol.

        Uses historical transition matrix to estimate likelihood of
        transitioning to each possible regime from current state.

        Args:
            symbol: Trading symbol

        Returns:
            Dict mapping regime names to transition probabilities
        """
        if symbol not in self._symbol_states:
            return {}

        state = self._symbol_states[symbol]
        matrix = self._get_transition_matrix(symbol)

        return matrix.get_transition_distribution(state.current_regime.name)

    def get_indicator_values(self, symbol: str) -> dict[str, Any]:
        """Get current values of all advanced indicators for a symbol.

        Useful for debugging and monitoring.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with indicator names and current values
        """
        if symbol not in self._symbol_states:
            return {}

        state = self._symbol_states[symbol]

        return {
            "atr": float(state.atr.value) if state.atr.value else None,
            "atr_percentile": state.get_atr_percentile(),
            "adx": state.get_adx(),
            "squeeze_score": state.get_squeeze_score(),
            "trend_score": state.get_trend_score(),
            "volatility_percentile": state.get_volatility_percentile(),
            "momentum_score": state.get_momentum_score(),
            "drawdown": state.get_drawdown(),
            "bollinger_ready": state.bollinger.is_ready,
            "trend_strength_ready": state.trend_strength.is_ready,
        }

    def get_atr_percentile(self, symbol: str) -> float | None:
        """Get ATR percentile for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            ATR percentile (0.0-1.0) or None if not ready
        """
        if symbol not in self._symbol_states:
            return None
        state = self._symbol_states[symbol]
        if not state.atr.is_ready:
            return None
        return state.get_atr_percentile()

    def get_adx(self, symbol: str) -> float | None:
        """Get ADX-style trend strength for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            ADX value (0-100) or None if not ready
        """
        if symbol not in self._symbol_states:
            return None
        state = self._symbol_states[symbol]
        return state.get_adx()

    def get_squeeze_score(self, symbol: str) -> float | None:
        """Get Bollinger squeeze score for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Squeeze score (0.0-1.0) or None if not ready
        """
        if symbol not in self._symbol_states:
            return None
        state = self._symbol_states[symbol]
        return state.get_squeeze_score()


__all__ = ["MarketRegimeDetector"]
