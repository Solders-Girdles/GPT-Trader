"""Baseline strategy orchestration logic."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.core.interfaces import Product
from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.features.live_trade.strategies.decisions import Action, Decision
from bot_v2.features.live_trade.strategies.shared import (
    create_close_decision,
    create_entry_decision,
)
from bot_v2.utilities.logging_patterns import get_logger
from bot_v2.utilities.quantities import quantity_from

from .config import StrategyConfig
from .signals import StrategySignal, build_signal
from .state import StrategyState

logger = get_logger(__name__, component="live_trade_strategy")


class BaselinePerpsStrategy:
    """Simple MA crossover strategy with trailing stops for perpetuals."""

    def __init__(
        self,
        *,
        config: StrategyConfig | None = None,
        risk_manager: LiveRiskManager | None = None,
        environment: str | None = None,
        state: StrategyState | None = None,
    ) -> None:
        """
        Initialize strategy with configuration.

        Args:
            config: Strategy configuration
            risk_manager: Risk manager for constraint checks
            environment: Optional label for logging/context (e.g., "live", "simulated")
            state: Optional state container (primarily for testing)
        """
        self.config = config or StrategyConfig()
        self.risk_manager = risk_manager
        self.environment = environment or "live"
        self.state = state or StrategyState()

        logger.info(
            "BaselinePerpsStrategy initialized env=%s config=%s",
            self.environment,
            self.config,
        )

    # --------------------------------------------------------------------- API
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
        Generate trading decision based on strategy logic.

        Args:
            symbol: Trading symbol
            current_mark: Current mark price
            position_state: Current position (quantity, side, entry) or None
            recent_marks: Recent mark prices for MA calculation (excluding current mark)
            equity: Account equity for sizing
            product: Product metadata for rules
        """
        signal = build_signal(
            current_mark=Decimal(str(current_mark)),
            recent_marks=recent_marks,
            config=self.config,
        )

        logger.info(
            f"Strategy decision debug: symbol={symbol} "
            f"marks={len(recent_marks) if recent_marks else 0} "
            f"short_ma={signal.snapshot.short_ma} "
            f"long_ma={signal.snapshot.long_ma} "
            f"bullish={signal.snapshot.bullish_cross} "
            f"bearish={signal.snapshot.bearish_cross} "
            f"label={signal.label} "
            f"force={self.config.force_entry_on_trend}"
        )

        position_quantity = quantity_from(position_state, default=Decimal("0"))
        has_position = position_state is not None and position_quantity not in (None, Decimal("0"))

        if decision := self._maybe_reduce_only(symbol, position_state, has_position):
            return decision

        if has_position:
            return self._manage_open_position(
                symbol=symbol,
                signal=signal,
                position_state=position_state or {},
                current_mark=Decimal(str(current_mark)),
            )

        return self._evaluate_entry_opportunity(
            symbol=symbol,
            signal=signal,
            equity=Decimal(str(equity)),
            product=product,
        )

    def reset(self, symbol: str | None = None) -> None:
        """Reset strategy state (per symbol or globally)."""
        self.state.reset(symbol)

    # ----------------------------------------------------------------- Helpers
    def _maybe_reduce_only(
        self,
        symbol: str,
        position_state: dict[str, Any] | None,
        has_position: bool,
    ) -> Decision | None:
        if self.risk_manager and self.risk_manager.is_reduce_only_mode():
            if has_position and position_state:
                return create_close_decision(
                    symbol=symbol,
                    position_state=position_state,
                    position_adds=self.state.position_adds,
                    trailing_stops=self.state.trailing_stops,
                    reason="Reduce-only mode active",
                )
            return Decision(action=Action.HOLD, reason="Reduce-only mode - no new entries")
        return None

    def _evaluate_entry_opportunity(
        self,
        *,
        symbol: str,
        signal: StrategySignal,
        equity: Decimal,
        product: Product,
    ) -> Decision:
        if self.config.disable_new_entries:
            return Decision(action=Action.HOLD, reason="New entries disabled")

        if signal.is_bullish:
            return create_entry_decision(
                symbol=symbol,
                action=Action.BUY,
                equity=equity,
                product=product,
                position_fraction=self.config.position_fraction,
                target_leverage=self.config.target_leverage,
                max_trade_usd=self.config.max_trade_usd,
                position_adds=self.state.position_adds,
                trailing_stops=self.state.trailing_stops,
                reason="Bullish MA crossover",
            )

        if signal.is_bearish and self.config.enable_shorts:
            return create_entry_decision(
                symbol=symbol,
                action=Action.SELL,
                equity=equity,
                product=product,
                position_fraction=self.config.position_fraction,
                target_leverage=self.config.target_leverage,
                max_trade_usd=self.config.max_trade_usd,
                position_adds=self.state.position_adds,
                trailing_stops=self.state.trailing_stops,
                reason="Bearish MA crossover",
            )

        return Decision(action=Action.HOLD, reason="No signal")

    def _manage_open_position(
        self,
        *,
        symbol: str,
        signal: StrategySignal,
        position_state: dict[str, Any],
        current_mark: Decimal,
    ) -> Decision:
        pos_side = str(position_state.get("side", "")).lower()

        if self.config.disable_new_entries:
            if self._should_exit_on_signal(signal, pos_side):
                return self._close_position(
                    symbol=symbol,
                    position_state=position_state,
                    reason=f"Exit on {signal.label} signal",
                )
            return Decision(action=Action.HOLD, reason="New entries disabled")

        if self._should_exit_on_signal(signal, pos_side):
            exit_reason = (
                "Exit long on bearish signal"
                if pos_side == "long"
                else "Exit short on bullish signal"
            )
            return self._close_position(
                symbol=symbol,
                position_state=position_state,
                reason=exit_reason,
            )

        trailing_hit = self.state.update_trailing_stop(
            symbol=symbol,
            side=pos_side,
            current_price=current_mark,
            trailing_pct=Decimal(str(self.config.trailing_stop_pct)),
        )
        if trailing_hit:
            peak, stop_price = self.state.trailing_stops.get(symbol, (current_mark, current_mark))
            reason = (
                f"Trailing stop hit (peak: {peak}, stop: {stop_price}, current: {current_mark})"
            )
            return self._close_position(symbol=symbol, position_state=position_state, reason=reason)

        adds = self.state.position_adds.get(symbol, 0)
        if adds < self.config.max_adds:
            if pos_side == "long" and signal.is_bullish:
                # Future enhancement: add-to-position logic.
                logger.debug("Pyramiding opportunity detected for %s (long)", symbol)
            elif pos_side == "short" and signal.is_bearish and self.config.enable_shorts:
                logger.debug("Pyramiding opportunity detected for %s (short)", symbol)

        return Decision(action=Action.HOLD, reason="No signal")

    def _should_exit_on_signal(self, signal: StrategySignal, pos_side: str) -> bool:
        if pos_side == "long" and signal.is_bearish:
            return True
        if pos_side == "short" and signal.is_bullish:
            return True
        return False

    def _close_position(
        self,
        *,
        symbol: str,
        position_state: dict[str, Any],
        reason: str,
    ) -> Decision:
        return create_close_decision(
            symbol=symbol,
            position_state=position_state,
            position_adds=self.state.position_adds,
            trailing_stops=self.state.trailing_stops,
            reason=reason,
        )


__all__ = ["BaselinePerpsStrategy"]
