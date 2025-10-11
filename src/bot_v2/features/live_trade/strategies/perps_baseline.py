"""
Baseline perpetuals trading strategy with MA crossover and trailing stops.

Phase 6: Minimal, production-safe strategy for perps trading.
Uses RiskManager constraints and ProductCatalog rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.core.interfaces import Product
from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.features.live_trade.strategies.decisions import Action, Decision
from bot_v2.features.live_trade.strategies.shared import (
    calculate_ma_snapshot,
    create_close_decision,
    create_entry_decision,
    update_mark_window,
    update_trailing_stop,
)
from bot_v2.utilities.quantities import quantity_from

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for baseline strategy."""

    # MA parameters
    short_ma_period: int = 5
    long_ma_period: int = 20

    # Position management
    target_leverage: int = 2
    trailing_stop_pct: float = 0.01  # 1% trailing stop (fixed)
    # Simple, predictable sizing: percentage of equity per trade, with optional USD cap
    position_fraction: float = 0.05  # 5% of equity
    max_trade_usd: Decimal | None = None  # cap notional if set

    # Feature flags
    enable_shorts: bool = False
    max_adds: int = 0  # Disable pyramiding by default
    disable_new_entries: bool = False
    # Advanced entries (deprecated/no-op in simplified baseline)
    use_stop_entry: bool = False
    use_post_only: bool = False
    prefer_maker_orders: bool = False

    # Funding-awareness (deprecated for baseline; retained for backward compat, not used)
    funding_bias_bps: float = 0.0
    funding_block_long_bps: float = 0.0
    funding_block_short_bps: float = 0.0

    # Crossover robustness (optional)
    # Epsilon tolerance in basis points for crossover detection (0 = strict)
    ma_cross_epsilon_bps: Decimal = Decimal("0")
    # Bars to confirm crossover persistence (0 = no confirmation)
    ma_cross_confirm_bars: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyConfig:
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class BaselinePerpsStrategy:
    """
    Simple MA crossover strategy with trailing stops for perpetuals.

    - Enter long on bullish MA crossover
    - Enter short on bearish MA crossover (if enabled)
    - Exit on opposing crossover or trailing stop hit
    - Respects RiskManager constraints and reduce-only mode
    """

    def __init__(
        self,
        config: StrategyConfig | None = None,
        risk_manager: LiveRiskManager | None = None,
        environment: str | None = None,
    ) -> None:
        """
        Initialize strategy with configuration.

        Args:
            config: Strategy configuration
            risk_manager: Risk manager for constraint checks
            environment: Optional label for logging/context (e.g., "live", "simulated")
        """
        self.config = config or StrategyConfig()
        self.risk_manager = risk_manager
        self.environment = environment or "live"

        # In-memory state
        self.mark_windows: dict[str, list[Decimal]] = {}
        self.position_adds: dict[str, int] = {}  # Track adds per symbol
        self.trailing_stops: dict[str, tuple[Decimal, Decimal]] = {}  # symbol -> (peak, stop_price)

        logger.info(
            "BaselinePerpsStrategy initialized env=%s config=%s",
            self.environment,
            self.config,
        )

    def decide(
        self,
        symbol: str,
        current_mark: Decimal,
        position_state: dict[str, Any] | None,
        recent_marks: list[Decimal] | None,
        equity: Decimal,
        product: Product,
    ) -> Decision:
        """
        Generate trading decision based on strategy logic.

        Args:
            symbol: Trading symbol
            current_mark: Current mark price
            position_state: Current position (quantity, side, entry) or None
            recent_marks: Recent mark prices for MA calculation
            equity: Account equity for sizing
            product: Product metadata for rules

        Returns:
            Decision with action and parameters
        """
        # Update mark window
        marks = update_mark_window(
            self.mark_windows,
            symbol=symbol,
            current_mark=current_mark,
            short_period=self.config.short_ma_period,
            long_period=self.config.long_ma_period,
            recent_marks=recent_marks,
        )

        snapshot = calculate_ma_snapshot(
            marks,
            short_period=self.config.short_ma_period,
            long_period=self.config.long_ma_period,
            epsilon_bps=self.config.ma_cross_epsilon_bps,
            confirm_bars=self.config.ma_cross_confirm_bars,
        )
        signal = (
            "bullish"
            if snapshot.bullish_cross
            else "bearish" if snapshot.bearish_cross else "neutral"
        )

        # Check for existing position
        position_quantity = quantity_from(position_state, default=Decimal("0"))
        has_position = position_quantity not in (None, Decimal("0"))

        # Check if we're in reduce-only mode
        if self.risk_manager and self.risk_manager.is_reduce_only_mode():
            if has_position and position_state:
                # Can only close in reduce-only mode
                return create_close_decision(
                    symbol=symbol,
                    position_state=position_state,
                    position_adds=self.position_adds,
                    trailing_stops=self.trailing_stops,
                    reason="Reduce-only mode active",
                )
            return Decision(action=Action.HOLD, reason="Reduce-only mode - no new entries")

        # Check if new entries are disabled
        if self.config.disable_new_entries:
            if has_position and position_state:
                # Check for exit signals
                pos_side = str(position_state.get("side", "")).lower()

                if (pos_side == "long" and signal == "bearish") or (
                    pos_side == "short" and signal == "bullish"
                ):
                    return create_close_decision(
                        symbol=symbol,
                        position_state=position_state,
                        position_adds=self.position_adds,
                        trailing_stops=self.trailing_stops,
                        reason=f"Exit on {signal} signal",
                    )

            return Decision(action=Action.HOLD, reason="New entries disabled")

        # Funding awareness intentionally disabled in simplified baseline

        # Generate decision based on signal and position
        if not has_position:
            # No position - check for entry
            if signal == "bullish":
                return create_entry_decision(
                    symbol=symbol,
                    action=Action.BUY,
                    equity=equity,
                    product=product,
                    position_fraction=self.config.position_fraction,
                    target_leverage=self.config.target_leverage,
                    max_trade_usd=self.config.max_trade_usd,
                    position_adds=self.position_adds,
                    trailing_stops=self.trailing_stops,
                    reason="Bullish MA crossover",
                )
            elif signal == "bearish" and self.config.enable_shorts:
                return create_entry_decision(
                    symbol=symbol,
                    action=Action.SELL,
                    equity=equity,
                    product=product,
                    position_fraction=self.config.position_fraction,
                    target_leverage=self.config.target_leverage,
                    max_trade_usd=self.config.max_trade_usd,
                    position_adds=self.position_adds,
                    trailing_stops=self.trailing_stops,
                    reason="Bearish MA crossover",
                )
        elif position_state:
            # Has position - check for exit or add
            pos_side = str(position_state.get("side", "")).lower()

            # Check for exit signal
            if pos_side == "long" and signal == "bearish":
                return create_close_decision(
                    symbol=symbol,
                    position_state=position_state,
                    position_adds=self.position_adds,
                    trailing_stops=self.trailing_stops,
                    reason="Exit long on bearish signal",
                )
            elif pos_side == "short" and signal == "bullish":
                return create_close_decision(
                    symbol=symbol,
                    position_state=position_state,
                    position_adds=self.position_adds,
                    trailing_stops=self.trailing_stops,
                    reason="Exit short on bullish signal",
                )

            trailing_hit = update_trailing_stop(
                self.trailing_stops,
                symbol=symbol,
                side=pos_side,
                current_price=current_mark,
                trailing_pct=Decimal(str(self.config.trailing_stop_pct)),
            )
            if trailing_hit:
                peak, stop_price = self.trailing_stops.get(symbol, (current_mark, current_mark))
                return create_close_decision(
                    symbol=symbol,
                    position_state=position_state,
                    position_adds=self.position_adds,
                    trailing_stops=self.trailing_stops,
                    reason=(
                        f"Trailing stop hit (peak: {peak}, stop: {stop_price}, current: {current_mark})"
                    ),
                )

            # Check for adding to position (pyramiding)
            adds = self.position_adds.get(symbol, 0)
            if adds < self.config.max_adds:
                if pos_side == "long" and signal == "bullish":
                    # Could add to long, but keep simple for now
                    pass
                elif pos_side == "short" and signal == "bearish" and self.config.enable_shorts:
                    # Could add to short, but keep simple for now
                    pass

        return Decision(action=Action.HOLD, reason="No signal")

    def update_marks(self, symbol: str, marks: list[Decimal]) -> None:
        """Seed or update the internal mark window for a symbol.

        This helper is used by some tests to pre-populate recent prices
        without invoking decide() first.
        """
        if not marks:
            self.mark_windows.pop(symbol, None)
            return

        update_mark_window(
            self.mark_windows,
            symbol=symbol,
            current_mark=Decimal(str(marks[-1])),
            short_period=self.config.short_ma_period,
            long_period=self.config.long_ma_period,
            recent_marks=[Decimal(str(value)) for value in marks[:-1]],
        )

    def reset(self, symbol: str | None = None) -> None:
        """
        Reset strategy state.

        Args:
            symbol: Reset specific symbol or all if None
        """
        if symbol:
            if symbol in self.mark_windows:
                del self.mark_windows[symbol]
            if symbol in self.position_adds:
                del self.position_adds[symbol]
            if symbol in self.trailing_stops:
                del self.trailing_stops[symbol]
        else:
            self.mark_windows.clear()
            self.position_adds.clear()
            self.trailing_stops.clear()


# Functional wrapper for simple usage
def create_baseline_strategy(
    config: dict[str, Any] | None = None, risk_manager: LiveRiskManager | None = None
) -> BaselinePerpsStrategy:
    """
    Create a baseline perpetuals strategy.

    Args:
        config: Strategy configuration dict
        risk_manager: Risk manager instance

    Returns:
        Configured strategy instance
    """
    strategy_config = StrategyConfig.from_dict(config) if config else StrategyConfig()
    return BaselinePerpsStrategy(config=strategy_config, risk_manager=risk_manager)
