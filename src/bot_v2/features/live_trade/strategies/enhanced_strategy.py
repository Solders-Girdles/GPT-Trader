"""
Enhanced strategy with volatility and trend strength filters.

Wraps BaselinePerpsStrategy with additional filters:
- Volatility filter: Only trade when ATR is in normal range
- Trend strength filter: Require minimum ADX for MA crossover entries
- Dynamic stops: ATR-based stops instead of fixed percentage

Designed to beat baseline across trend/range/high-vol regimes.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.core.interfaces import Product
from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.features.live_trade.strategies.decisions import Action, Decision
from bot_v2.features.live_trade.strategies.indicators import (
    calculate_adx,
    calculate_atr,
    calculate_dynamic_stop,
    is_high_volatility,
    is_trending_market,
)
from bot_v2.features.live_trade.strategies.perps_baseline.config import StrategyConfig
from bot_v2.features.live_trade.strategies.perps_baseline.strategy import BaselinePerpsStrategy
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="enhanced_strategy")


class EnhancedStrategyConfig(StrategyConfig):
    """Extended config with enhancement parameters."""

    # Volatility filter settings
    enable_volatility_filter: bool = True
    atr_period: int = 14
    atr_high_threshold_multiplier: float = 2.0  # High vol if ATR > 2x average
    atr_low_threshold_multiplier: float = 0.5  # Low vol if ATR < 0.5x average

    # Trend strength filter settings
    enable_trend_filter: bool = True
    adx_period: int = 14
    adx_min_threshold: float = 20.0  # Require ADX > 20 for entries

    # Dynamic stop settings
    enable_dynamic_stops: bool = True
    atr_stop_multiplier: float = 2.0  # Stop distance = 2 * ATR
    min_stop_pct: float = 0.005  # 0.5% minimum
    max_stop_pct: float = 0.05  # 5% maximum


class EnhancedStrategy:
    """
    Enhanced MA crossover strategy with volatility and trend filters.

    Wraps BaselinePerpsStrategy and adds filtering logic:
    1. Volatility filter: Skip trades in extreme volatility (too high or too low)
    2. Trend filter: Require minimum ADX for trend-following entries
    3. Dynamic stops: Use ATR-based stops instead of fixed percentage

    Design philosophy:
    - Don't trade in choppy/ranging markets (low ADX)
    - Don't trade in extreme volatility spikes (high ATR)
    - Use wider stops in volatile periods, tighter in calm periods
    """

    def __init__(
        self,
        *,
        config: EnhancedStrategyConfig | None = None,
        risk_manager: LiveRiskManager | None = None,
        environment: str | None = None,
    ) -> None:
        """
        Initialize enhanced strategy.

        Args:
            config: Enhanced strategy configuration
            risk_manager: Risk manager for constraint checks
            environment: Optional label (e.g., "live", "backtest")
        """
        self.config = config or EnhancedStrategyConfig()
        self.risk_manager = risk_manager
        self.environment = environment or "live"

        # Wrapped baseline strategy
        self.baseline = BaselinePerpsStrategy(
            config=self.config,  # EnhancedStrategyConfig extends StrategyConfig
            risk_manager=risk_manager,
            environment=environment,
        )

        # Indicator state (for calculating avg ATR)
        self._recent_atr_values: dict[str, list[float]] = {}
        self._atr_window: int = 50  # Rolling window for average ATR

        logger.info(
            "EnhancedStrategy initialized | env=%s | vol_filter=%s | trend_filter=%s | dynamic_stops=%s",
            self.environment,
            self.config.enable_volatility_filter,
            self.config.enable_trend_filter,
            self.config.enable_dynamic_stops,
        )

    @property
    def state(self):
        """Expose baseline state for compatibility."""
        return self.baseline.state

    def decide(
        self,
        *,
        symbol: str,
        current_mark: Decimal,
        position_state: dict[str, Any] | None,
        recent_marks: Sequence[Decimal] | None,
        equity: Decimal,
        product: Product,
    ) -> Decision:
        """
        Generate trading decision with enhanced filters.

        Args:
            symbol: Trading symbol
            current_mark: Current mark price
            position_state: Current position or None
            recent_marks: Recent mark prices for indicators
            equity: Account equity
            product: Product metadata

        Returns:
            Decision with filters applied
        """
        # Get baseline decision
        baseline_decision = self.baseline.decide(
            symbol=symbol,
            current_mark=current_mark,
            position_state=position_state,
            recent_marks=recent_marks,
            equity=equity,
            product=product,
        )

        # If baseline says HOLD or position is already open, pass through
        # (filters only apply to new entries)
        has_position = position_state is not None and position_state.get("quantity", Decimal("0")) != Decimal("0")

        if baseline_decision.action == Action.HOLD or has_position:
            # For open positions, potentially adjust stops dynamically
            if has_position and self.config.enable_dynamic_stops:
                return self._maybe_adjust_dynamic_stop(
                    baseline_decision=baseline_decision,
                    symbol=symbol,
                    current_mark=current_mark,
                    recent_marks=recent_marks,
                    position_state=position_state,
                )
            return baseline_decision

        # Apply filters to new entry signals (BUY or SELL)
        if baseline_decision.action in (Action.BUY, Action.SELL):
            return self._filter_entry_signal(
                baseline_decision=baseline_decision,
                symbol=symbol,
                current_mark=current_mark,
                recent_marks=recent_marks,
            )

        return baseline_decision

    def reset(self, symbol: str | None = None) -> None:
        """Reset strategy state."""
        self.baseline.reset(symbol)
        if symbol:
            self._recent_atr_values.pop(symbol, None)
        else:
            self._recent_atr_values.clear()

    # ----------------------------------------------------------------- Filters

    def _filter_entry_signal(
        self,
        *,
        baseline_decision: Decision,
        symbol: str,
        current_mark: Decimal,
        recent_marks: Sequence[Decimal] | None,
    ) -> Decision:
        """
        Apply volatility and trend filters to entry signal.

        Returns:
            Original decision if filters pass, else HOLD decision
        """
        if not recent_marks or len(recent_marks) < max(self.config.atr_period, self.config.adx_period):
            logger.debug("Insufficient data for filters | symbol=%s", symbol)
            return baseline_decision

        # Calculate indicators (need OHLC approximations from marks)
        highs, lows, closes = self._approximate_ohlc(recent_marks, current_mark)

        # Volatility filter
        if self.config.enable_volatility_filter:
            if not self._passes_volatility_filter(symbol, highs, lows, closes):
                return Decision(
                    action=Action.HOLD,
                    reason=f"Volatility filter blocked entry (original: {baseline_decision.reason})",
                )

        # Trend strength filter
        if self.config.enable_trend_filter:
            if not self._passes_trend_filter(highs, lows, closes):
                return Decision(
                    action=Action.HOLD,
                    reason=f"Trend filter blocked entry (ADX too low, original: {baseline_decision.reason})",
                )

        logger.debug(
            "Entry signal passed filters | symbol=%s | action=%s",
            symbol,
            baseline_decision.action.value,
        )
        return baseline_decision

    def _passes_volatility_filter(
        self,
        symbol: str,
        highs: list[Decimal],
        lows: list[Decimal],
        closes: list[Decimal],
    ) -> bool:
        """
        Check if current volatility is in acceptable range.

        Returns:
            True if ATR is not too high or too low
        """
        current_atr = calculate_atr(highs, lows, closes, period=self.config.atr_period)
        if current_atr is None:
            return True  # Insufficient data - allow trade

        current_atr_float = float(current_atr)

        # Track ATR for rolling average
        if symbol not in self._recent_atr_values:
            self._recent_atr_values[symbol] = []

        self._recent_atr_values[symbol].append(current_atr_float)
        if len(self._recent_atr_values[symbol]) > self._atr_window:
            self._recent_atr_values[symbol].pop(0)

        # Calculate average ATR
        avg_atr = sum(self._recent_atr_values[symbol]) / len(self._recent_atr_values[symbol])

        # Check if volatility is extreme
        is_high_vol = is_high_volatility(
            current_atr,
            avg_atr=Decimal(str(avg_atr)),
            volatility_threshold=Decimal(str(self.config.atr_high_threshold_multiplier)),
        )

        is_low_vol = current_atr_float < avg_atr * self.config.atr_low_threshold_multiplier

        if is_high_vol:
            logger.info(
                "Volatility filter: HIGH volatility | symbol=%s | current_atr=%.4f | avg_atr=%.4f",
                symbol,
                current_atr_float,
                avg_atr,
            )
            return False

        if is_low_vol:
            logger.info(
                "Volatility filter: LOW volatility | symbol=%s | current_atr=%.4f | avg_atr=%.4f",
                symbol,
                current_atr_float,
                avg_atr,
            )
            return False

        return True

    def _passes_trend_filter(
        self,
        highs: list[Decimal],
        lows: list[Decimal],
        closes: list[Decimal],
    ) -> bool:
        """
        Check if trend strength is sufficient for MA crossover.

        Returns:
            True if ADX >= threshold
        """
        adx = calculate_adx(highs, lows, closes, period=self.config.adx_period)
        if adx is None:
            return True  # Insufficient data - allow trade

        adx_threshold = Decimal(str(self.config.adx_min_threshold))
        passes = is_trending_market(adx, adx_threshold=adx_threshold)

        if not passes:
            logger.info(
                "Trend filter: ADX too low | adx=%.2f | threshold=%.2f",
                float(adx),
                float(adx_threshold),
            )

        return passes

    def _maybe_adjust_dynamic_stop(
        self,
        *,
        baseline_decision: Decision,
        symbol: str,
        current_mark: Decimal,
        recent_marks: Sequence[Decimal] | None,
        position_state: dict[str, Any] | None,
    ) -> Decision:
        """
        Potentially adjust trailing stop based on current ATR.

        For now, just pass through baseline decision.
        Future: Could widen stops in volatile periods.
        """
        # Future enhancement: Adjust trailing stop percentage based on ATR
        # For Phase 2, we'll keep fixed trailing stops from baseline
        return baseline_decision

    # ----------------------------------------------------------------- Helpers

    def _approximate_ohlc(
        self,
        recent_marks: Sequence[Decimal],
        current_mark: Decimal,
    ) -> tuple[list[Decimal], list[Decimal], list[Decimal]]:
        """
        Approximate OHLC from mark prices.

        For indicator calculation, we approximate:
        - High = mark * 1.002 (0.2% above)
        - Low = mark * 0.998 (0.2% below)
        - Close = mark

        This is a simplification since we don't have true OHLC data
        in the live strategy decide() loop. Good enough for filters.
        """
        all_marks = list(recent_marks) + [current_mark]

        highs = [m * Decimal("1.002") for m in all_marks]
        lows = [m * Decimal("0.998") for m in all_marks]
        closes = list(all_marks)

        return highs, lows, closes


__all__ = ["EnhancedStrategy", "EnhancedStrategyConfig"]
