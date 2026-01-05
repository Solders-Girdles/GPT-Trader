"""
Stateful baseline perpetuals trading strategy with O(1) incremental updates.

This is a stateful version of BaselinePerpsStrategy that maintains internal
indicator state, providing:
- O(1) per-tick updates instead of O(n) recalculation
- State serialization for crash recovery
- Reduced memory usage (no full price history required)

The trading logic is identical to BaselinePerpsStrategy.
"""

from collections.abc import Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from gpt_trader.core import Product
from gpt_trader.features.live_trade.stateful_indicators import (
    OnlineRSI,
    OnlineSMA,
    RollingWindow,
)
from gpt_trader.features.live_trade.strategies.base import StatefulStrategyBase

from .strategy import (
    Action,
    BaseStrategyConfig,
    Decision,
    IndicatorState,
    StrategyConfig,
)

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.strategies.base import MarketDataContext


class StatefulBaselineStrategy(StatefulStrategyBase):
    """Stateful version of BaselinePerpsStrategy using O(1) incremental indicators.

    This strategy maintains internal state for all indicators:
    - OnlineSMA for short and long moving averages
    - OnlineRSI for relative strength index
    - RollingWindow for recent prices (crossover detection)

    State is updated incrementally with each price tick via update(),
    and can be serialized/deserialized for crash recovery.

    The trading logic (decide) is identical to the stateless version.
    """

    def __init__(
        self,
        config: BaseStrategyConfig | Any = None,
        risk_manager: Any = None,
    ):
        self.config: BaseStrategyConfig
        if config is None:
            self.config = StrategyConfig()
        elif isinstance(config, BaseStrategyConfig):
            self.config = config
        else:
            self.config = self._parse_config(config)

        self.risk_manager = risk_manager

        # Per-symbol indicator bundles
        self._indicators: dict[str, _SymbolIndicators] = {}

    def _parse_config(self, config: Any) -> StrategyConfig:
        """Parse configuration from various formats."""
        if hasattr(config, "__dict__"):
            kwargs = {}
            for field_name in StrategyConfig.__dataclass_fields__:
                if hasattr(config, field_name):
                    kwargs[field_name] = getattr(config, field_name)
            return StrategyConfig(**kwargs)
        return StrategyConfig()

    def _get_or_create_indicators(self, symbol: str) -> "_SymbolIndicators":
        """Get or create indicator bundle for a symbol."""
        if symbol not in self._indicators:
            self._indicators[symbol] = _SymbolIndicators(
                short_sma=OnlineSMA(period=self.config.short_ma_period),
                long_sma=OnlineSMA(period=self.config.long_ma_period),
                rsi=OnlineRSI(period=self.config.rsi_period),
                # Keep enough history for crossover detection
                price_history=RollingWindow(max_size=max(self.config.long_ma_period + 5, 30)),
            )
        return self._indicators[symbol]

    def update(self, symbol: str, price: Decimal) -> None:
        """Update indicators with new price. O(1) operation.

        Args:
            symbol: Trading pair symbol
            price: Latest mark/spot price
        """
        indicators = self._get_or_create_indicators(symbol)
        indicators.short_sma.update(price)
        indicators.long_sma.update(price)
        indicators.rsi.update(price)
        indicators.price_history.add(price)

    def serialize_state(self) -> dict[str, Any]:
        """Serialize all indicator state for persistence."""
        return {
            "indicators": {
                symbol: {
                    "short_sma": ind.short_sma.serialize(),
                    "long_sma": ind.long_sma.serialize(),
                    "rsi": ind.rsi.serialize(),
                    "price_history": ind.price_history.serialize(),
                }
                for symbol, ind in self._indicators.items()
            }
        }

    def deserialize_state(self, state: dict[str, Any]) -> None:
        """Restore indicator state from serialized data."""
        indicators_data = state.get("indicators", {})
        for symbol, data in indicators_data.items():
            self._indicators[symbol] = _SymbolIndicators(
                short_sma=OnlineSMA.deserialize(data["short_sma"]),
                long_sma=OnlineSMA.deserialize(data["long_sma"]),
                rsi=OnlineRSI.deserialize(data["rsi"]),
                price_history=RollingWindow.deserialize(data["price_history"]),
            )

    def decide(
        self,
        symbol: str,
        current_mark: Decimal,
        position_state: dict[str, Any] | None,
        recent_marks: Sequence[Decimal],
        equity: Decimal,
        product: Product | None,
        market_data: "MarketDataContext | None" = None,
    ) -> Decision:
        """Generate a trading decision using stateful indicators.

        Note: This method updates internal state AND makes a decision.
        The recent_marks parameter is used for backward compatibility
        but the strategy primarily uses its internal state.

        Args:
            symbol: Trading pair symbol
            current_mark: Current mark price
            position_state: Current position info (if any)
            recent_marks: Historical mark prices (for compatibility)
            equity: Account equity
            product: Product specification
            market_data: Optional enhanced market data (orderbook depth, trade flow)

        Returns:
            Decision with action, reason, confidence, and indicator state
        """
        # Update state with current price
        self.update(symbol, current_mark)

        # Check kill switch
        if self.config.kill_switch_enabled:
            return Decision(
                Action.HOLD,
                "Kill switch enabled",
                confidence=0.0,
                indicators={"kill_switch": True},
            )

        # Get indicator bundle
        indicators = self._get_or_create_indicators(symbol)

        # Check minimum data requirements
        min_data = max(self.config.long_ma_period, self.config.rsi_period + 1)
        if not indicators.is_ready(min_data):
            return Decision(
                Action.HOLD,
                "Insufficient data: warming up indicators",
                confidence=0.0,
                indicators={"warming_up": True, "required": min_data},
            )

        # Calculate indicator state from online indicators
        indicator_state = self._build_indicator_state(indicators, current_mark)

        # Determine if we have an existing position
        has_position = position_state is not None and position_state.get("quantity", 0) != 0
        position_side = position_state.get("side", "none") if position_state else "none"

        # Generate decision based on position state
        if has_position:
            return self._decide_with_position(
                symbol, current_mark, position_side, indicator_state, position_state
            )
        else:
            return self._decide_entry(symbol, current_mark, indicator_state)

    def _build_indicator_state(
        self, indicators: "_SymbolIndicators", current_price: Decimal
    ) -> IndicatorState:
        """Build IndicatorState from online indicator values."""
        state = IndicatorState()

        # Get current values from online indicators
        state.rsi = indicators.rsi.value
        state.short_ma = indicators.short_sma.value
        state.long_ma = indicators.long_sma.value

        # Determine RSI signal
        if state.rsi is not None:
            if state.rsi < self.config.rsi_oversold:
                state.rsi_signal = "oversold"
            elif state.rsi > self.config.rsi_overbought:
                state.rsi_signal = "overbought"
            else:
                state.rsi_signal = "neutral"

        # Determine trend from price vs long MA
        if state.long_ma is not None:
            if current_price > state.long_ma:
                state.trend = "bullish"
            elif current_price < state.long_ma:
                state.trend = "bearish"
            else:
                state.trend = "neutral"

        # Detect crossover from recent history
        state.crossover_signal = self._detect_crossover_from_history(indicators)

        return state

    def _detect_crossover_from_history(self, indicators: "_SymbolIndicators") -> str:
        """Detect MA crossover from price history.

        For stateful strategies, we need to track MA values over time
        to detect crossovers. This is a simplified approach using
        the current and previous relationship.
        """
        if not indicators.short_sma.is_ready or not indicators.long_sma.is_ready:
            return "none"

        short_ma = indicators.short_sma.value
        long_ma = indicators.long_sma.value

        if short_ma is None or long_ma is None:
            return "none"

        # Get previous values from history if available
        history = list(indicators.price_history.values)
        if len(history) < 3:
            return "none"

        # Calculate MA at previous point (approximation using history)
        # For a more accurate crossover, we'd need to store MA history
        prev_short = (
            sum(history[-self.config.short_ma_period - 1 : -1])
            / Decimal(self.config.short_ma_period)
            if len(history) > self.config.short_ma_period
            else None
        )
        prev_long = (
            sum(history[-self.config.long_ma_period - 1 : -1]) / Decimal(self.config.long_ma_period)
            if len(history) > self.config.long_ma_period
            else None
        )

        if prev_short is None or prev_long is None:
            return "none"

        # Detect crossover
        if prev_short <= prev_long and short_ma > long_ma:
            return "bullish"
        elif prev_short >= prev_long and short_ma < long_ma:
            return "bearish"

        return "none"

    def _decide_entry(
        self,
        symbol: str,
        current_mark: Decimal,
        indicators: IndicatorState,
    ) -> Decision:
        """Generate entry decision when no position exists."""
        signals: list[tuple[str, float]] = []
        reasons: list[str] = []

        # RSI signal
        if indicators.rsi_signal == "oversold":
            signals.append(("buy", self.config.rsi_weight))
            reasons.append(f"RSI oversold ({indicators.rsi:.1f})")
        elif indicators.rsi_signal == "overbought":
            signals.append(("sell", self.config.rsi_weight))
            reasons.append(f"RSI overbought ({indicators.rsi:.1f})")

        # Crossover signal
        if indicators.crossover_signal == "bullish":
            signals.append(("buy", self.config.crossover_weight))
            reasons.append("Bullish MA crossover")
        elif indicators.crossover_signal == "bearish":
            signals.append(("sell", self.config.crossover_weight))
            reasons.append("Bearish MA crossover")

        # Trend alignment
        if indicators.trend == "bullish":
            signals.append(("buy", self.config.trend_weight))
            reasons.append("Bullish trend")
        elif indicators.trend == "bearish":
            signals.append(("sell", self.config.trend_weight))
            reasons.append("Bearish trend")

        # Calculate net signal and confidence
        buy_weight = sum(w for d, w in signals if d == "buy")
        sell_weight = sum(w for d, w in signals if d == "sell")

        indicator_dict = self._indicators_to_dict(indicators)

        # Determine action based on weighted signals
        if buy_weight > sell_weight and buy_weight >= self.config.min_confidence:
            return Decision(
                Action.BUY,
                reason="; ".join(reasons) if reasons else "Buy signal",
                confidence=min(buy_weight, 1.0),
                indicators=indicator_dict,
            )
        elif sell_weight > buy_weight and sell_weight >= self.config.min_confidence:
            return Decision(
                Action.SELL,
                reason="; ".join(reasons) if reasons else "Sell signal",
                confidence=min(sell_weight, 1.0),
                indicators=indicator_dict,
            )

        return Decision(
            Action.HOLD,
            reason=("No clear signal" if not reasons else f"Weak signals: {'; '.join(reasons)}"),
            confidence=max(buy_weight, sell_weight),
            indicators=indicator_dict,
        )

    def _decide_with_position(
        self,
        symbol: str,
        current_mark: Decimal,
        position_side: str,
        indicators: IndicatorState,
        position_state: dict[str, Any] | None,
    ) -> Decision:
        """Generate decision when a position exists (exit or hold)."""
        indicator_dict = self._indicators_to_dict(indicators)
        reasons: list[str] = []

        # Check for exit signals based on position side
        if position_side == "long":
            exit_confidence = 0.0
            if indicators.rsi_signal == "overbought":
                exit_confidence += self.config.rsi_weight
                reasons.append(f"RSI overbought ({indicators.rsi:.1f})")
            if indicators.crossover_signal == "bearish":
                exit_confidence += self.config.crossover_weight
                reasons.append("Bearish MA crossover")
            if indicators.trend == "bearish":
                exit_confidence += self.config.trend_weight * 0.5
                reasons.append("Bearish trend developing")

            if exit_confidence >= self.config.min_confidence:
                return Decision(
                    Action.CLOSE,
                    reason="; ".join(reasons),
                    confidence=min(exit_confidence, 1.0),
                    indicators=indicator_dict,
                )

        elif position_side == "short":
            exit_confidence = 0.0
            if indicators.rsi_signal == "oversold":
                exit_confidence += self.config.rsi_weight
                reasons.append(f"RSI oversold ({indicators.rsi:.1f})")
            if indicators.crossover_signal == "bullish":
                exit_confidence += self.config.crossover_weight
                reasons.append("Bullish MA crossover")
            if indicators.trend == "bullish":
                exit_confidence += self.config.trend_weight * 0.5
                reasons.append("Bullish trend developing")

            if exit_confidence >= self.config.min_confidence:
                return Decision(
                    Action.CLOSE,
                    reason="; ".join(reasons),
                    confidence=min(exit_confidence, 1.0),
                    indicators=indicator_dict,
                )

        # Check stop loss / take profit
        if position_state:
            entry_price = position_state.get("entry_price")
            if entry_price is not None:
                entry = Decimal(str(entry_price))
                pnl_pct = (current_mark - entry) / entry
                if position_side == "short":
                    pnl_pct = -pnl_pct

                if pnl_pct <= -Decimal(str(self.config.stop_loss_pct)):
                    return Decision(
                        Action.CLOSE,
                        reason=f"Stop loss triggered ({pnl_pct:.2%})",
                        confidence=1.0,
                        indicators=indicator_dict,
                    )
                if pnl_pct >= Decimal(str(self.config.take_profit_pct)):
                    return Decision(
                        Action.CLOSE,
                        reason=f"Take profit triggered ({pnl_pct:.2%})",
                        confidence=1.0,
                        indicators=indicator_dict,
                    )

        return Decision(
            Action.HOLD,
            reason="Holding position",
            confidence=0.5,
            indicators=indicator_dict,
        )

    def _indicators_to_dict(self, indicators: IndicatorState) -> dict[str, Any]:
        """Convert indicator state to dictionary for decision output."""
        return {
            "rsi": float(indicators.rsi) if indicators.rsi is not None else None,
            "short_ma": (float(indicators.short_ma) if indicators.short_ma is not None else None),
            "long_ma": (float(indicators.long_ma) if indicators.long_ma is not None else None),
            "crossover_signal": indicators.crossover_signal,
            "trend": indicators.trend,
            "rsi_signal": indicators.rsi_signal,
            "stateful": True,  # Mark as stateful strategy
        }


class _SymbolIndicators:
    """Internal container for per-symbol indicator state."""

    __slots__ = ("short_sma", "long_sma", "rsi", "price_history")

    def __init__(
        self,
        short_sma: OnlineSMA,
        long_sma: OnlineSMA,
        rsi: OnlineRSI,
        price_history: RollingWindow,
    ):
        self.short_sma = short_sma
        self.long_sma = long_sma
        self.rsi = rsi
        self.price_history = price_history

    def is_ready(self, min_data: int) -> bool:
        """Check if indicators have enough data."""
        return (
            self.short_sma.is_ready
            and self.long_sma.is_ready
            and self.rsi.initialized
            and len(self.price_history) >= min_data
        )


# Convenience aliases
StatefulPerpsStrategy = StatefulBaselineStrategy


__all__ = ["StatefulBaselineStrategy", "StatefulPerpsStrategy"]
