"""Regime switching strategy.

Hard-switches between a trend-following strategy and a mean-reversion strategy
based on `MarketRegimeDetector` output.

Design goals:
- Backtestable with existing backtest runner.
- Safe defaults (HOLD on UNKNOWN, optional CLOSE on CRISIS).
- Avoid churn: once a position is opened by one side, keep that side "sticky"
  until the position closes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

from gpt_trader.features.intelligence.regime import MarketRegimeDetector, RegimeState, RegimeType
from gpt_trader.features.live_trade.interfaces import TradingStrategy
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.core import Product
    from gpt_trader.features.live_trade.strategies.base import MarketDataContext

logger = get_logger(__name__, component="regime_switcher")

_StrategyKind = Literal["trend", "mean_reversion"]


def _is_position_open(position_state: dict[str, Any] | None) -> bool:
    if not position_state:
        return False
    quantity = position_state.get("quantity", 0)
    try:
        return Decimal(str(quantity)) != 0
    except Exception:
        return bool(quantity)


def _bucket_for_regime(
    regime: RegimeType,
) -> Literal["trend", "mean_reversion", "crisis", "unknown"]:
    if regime in (
        RegimeType.BULL_QUIET,
        RegimeType.BULL_VOLATILE,
        RegimeType.BEAR_QUIET,
        RegimeType.BEAR_VOLATILE,
    ):
        return "trend"
    if regime in (RegimeType.SIDEWAYS_QUIET, RegimeType.SIDEWAYS_VOLATILE):
        return "mean_reversion"
    if regime == RegimeType.CRISIS:
        return "crisis"
    return "unknown"


def _with_regime_indicators(
    decision: Decision,
    *,
    regime_state: RegimeState,
    selected: _StrategyKind | None,
    sticky_used: bool,
    active: _StrategyKind | None,
) -> Decision:
    indicators = dict(decision.indicators or {})
    indicators.update(
        {
            "regime": regime_state.regime.name,
            "regime_confidence": regime_state.confidence,
            "regime_trend_score": regime_state.trend_score,
            "regime_volatility_percentile": regime_state.volatility_percentile,
            "regime_momentum_score": regime_state.momentum_score,
            "regime_age_ticks": regime_state.regime_age_ticks,
            "regime_transition_probability": regime_state.transition_probability,
            "regime_switcher_selected": selected,
            "regime_switcher_active": active,
            "regime_switcher_sticky_used": sticky_used,
        }
    )
    return Decision(
        action=decision.action,
        reason=decision.reason,
        confidence=decision.confidence,
        indicators=indicators,
    )


class RegimeSwitchingStrategy:
    """Switch between trend and mean-reversion strategies by detected regime."""

    def __init__(
        self,
        *,
        trend_strategy_factory: Callable[[], TradingStrategy],
        mean_reversion_strategy_factory: Callable[[], TradingStrategy],
        regime_detector: MarketRegimeDetector | None = None,
        required_lookback_bars: int = 64,
        sticky_positions: bool = True,
        close_on_crisis: bool = True,
    ) -> None:
        self._trend_factory = trend_strategy_factory
        self._mean_reversion_factory = mean_reversion_strategy_factory
        self._detector = regime_detector or MarketRegimeDetector()
        self._required_lookback_bars = max(1, int(required_lookback_bars))
        self._sticky_positions = sticky_positions
        self._close_on_crisis = close_on_crisis

        self._trend_by_symbol: dict[str, TradingStrategy] = {}
        self._mean_reversion_by_symbol: dict[str, TradingStrategy] = {}
        self._active_by_symbol: dict[str, _StrategyKind] = {}

        logger.info(
            "Initialized RegimeSwitchingStrategy",
            required_lookback_bars=self._required_lookback_bars,
            sticky_positions=self._sticky_positions,
            close_on_crisis=self._close_on_crisis,
        )

    def required_lookback_bars(self) -> int:
        return self._required_lookback_bars

    def _get_trend(self, symbol: str) -> TradingStrategy:
        strategy = self._trend_by_symbol.get(symbol)
        if strategy is None:
            strategy = self._trend_factory()
            self._trend_by_symbol[symbol] = strategy
        return strategy

    def _get_mean_reversion(self, symbol: str) -> TradingStrategy:
        strategy = self._mean_reversion_by_symbol.get(symbol)
        if strategy is None:
            strategy = self._mean_reversion_factory()
            self._mean_reversion_by_symbol[symbol] = strategy
        return strategy

    def decide(
        self,
        symbol: str,
        current_mark: Decimal,
        position_state: dict[str, Any] | None,
        recent_marks: Sequence[Decimal],
        equity: Decimal,
        product: Product | None,
        market_data: MarketDataContext | None = None,
        candles: Sequence[Any] | None = None,
    ) -> Decision:
        position_open = _is_position_open(position_state)
        regime_state = self._detector.update(symbol, current_mark)
        bucket = _bucket_for_regime(regime_state.regime)

        active = self._active_by_symbol.get(symbol)
        selected: _StrategyKind | None = None
        sticky_used = False

        if bucket == "crisis":
            if self._close_on_crisis and position_open:
                decision = Decision(
                    action=Action.CLOSE,
                    reason="Crisis regime: closing position",
                    confidence=1.0,
                    indicators={},
                )
            else:
                decision = Decision(
                    action=Action.HOLD,
                    reason="Crisis regime: holding",
                    confidence=0.0,
                    indicators={},
                )
            return _with_regime_indicators(
                decision,
                regime_state=regime_state,
                selected=None,
                sticky_used=False,
                active=active,
            )

        if bucket == "unknown":
            decision = Decision(
                action=Action.HOLD,
                reason="Regime unknown: holding",
                confidence=0.0,
                indicators={},
            )
            return _with_regime_indicators(
                decision,
                regime_state=regime_state,
                selected=None,
                sticky_used=False,
                active=active,
            )

        # Sticky position management: once a side opens a position, keep it in control
        # until the position closes. This avoids regime whipsaw on exits.
        if position_open and self._sticky_positions and active is not None:
            selected = active
            sticky_used = True
        else:
            selected = "trend" if bucket == "trend" else "mean_reversion"

        delegate = (
            self._get_trend(symbol) if selected == "trend" else self._get_mean_reversion(symbol)
        )
        delegated = delegate.decide(
            symbol=symbol,
            current_mark=current_mark,
            position_state=position_state,
            recent_marks=recent_marks,
            equity=equity,
            product=product,
            market_data=market_data,
            candles=candles,
        )

        # Track which side opened the current position (best-effort).
        if not position_open and delegated.action in (Action.BUY, Action.SELL):
            self._active_by_symbol[symbol] = selected
        elif position_open and delegated.action == Action.CLOSE:
            self._active_by_symbol.pop(symbol, None)

        return _with_regime_indicators(
            delegated,
            regime_state=regime_state,
            selected=selected,
            sticky_used=sticky_used,
            active=self._active_by_symbol.get(symbol),
        )

    def rehydrate(self, events: Sequence[dict[str, Any]]) -> int:
        processed = 0
        for strategy in [*self._trend_by_symbol.values(), *self._mean_reversion_by_symbol.values()]:
            if hasattr(strategy, "rehydrate"):
                processed += int(strategy.rehydrate(events))
        return processed
