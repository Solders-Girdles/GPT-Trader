"""Enhanced baseline strategy for perpetuals trading with guard rails."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from ....features.strategy import (
    MarketConditionFilters,
    RiskGuards,
    StrategyEnhancements,
    create_conservative_filters,
    create_standard_risk_guards,
)
from ...brokerages.core.interfaces import Product
from ..risk import LiveRiskManager

__all__ = [
    "Action",
    "Decision",
    "StrategyFiltersConfig",
    "StrategyConfig",
    "PerpsBaselineEnhancedStrategy",
]

logger = logging.getLogger(__name__)


class Action(Enum):
    """Trading action decisions."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class Decision:
    """Enhanced strategy decision with rejection tracking."""

    action: Action
    target_notional: Decimal | None = None
    qty: Decimal | None = None
    leverage: int | None = None
    stop_params: dict[str, Any] | None = None
    reduce_only: bool = False
    reason: str = ""

    # Rejection tracking
    filter_rejected: bool = False
    guard_rejected: bool = False
    rejection_type: str | None = None  # 'spread', 'depth', 'volume', 'rsi', 'liq', 'slippage'


@dataclass
class StrategyFiltersConfig:
    """Configuration for market condition filters and risk guards."""

    # Market condition filters
    max_spread_bps: Decimal | None = Decimal("10")
    min_depth_l1: Decimal | None = Decimal("50000")
    min_depth_l10: Decimal | None = Decimal("200000")
    min_volume_1m: Decimal | None = Decimal("100000")
    require_rsi_confirmation: bool = True

    # Risk guards
    min_liquidation_buffer_pct: Decimal | None = Decimal("20")
    max_slippage_impact_bps: Decimal | None = Decimal("15")

    # RSI parameters
    rsi_period: int = 14
    rsi_oversold: Decimal = Decimal("30")
    rsi_overbought: Decimal = Decimal("70")

    def create_filters(self) -> MarketConditionFilters:
        """Create market condition filters from config."""
        return MarketConditionFilters(
            max_spread_bps=self.max_spread_bps,
            min_depth_l1=self.min_depth_l1,
            min_depth_l10=self.min_depth_l10,
            min_volume_1m=self.min_volume_1m,
            min_volume_5m=None,  # Optional
            rsi_oversold=self.rsi_oversold,
            rsi_overbought=self.rsi_overbought,
            require_rsi_confirmation=self.require_rsi_confirmation,
        )

    def create_guards(self) -> RiskGuards:
        """Create risk guards from config."""
        return RiskGuards(
            min_liquidation_buffer_pct=self.min_liquidation_buffer_pct,
            max_slippage_impact_bps=self.max_slippage_impact_bps,
        )


@dataclass
class StrategyConfig:
    """Enhanced configuration for baseline strategy."""

    # MA parameters
    short_ma_period: int = 5
    long_ma_period: int = 20
    ma_cross_epsilon_bps: Decimal = Decimal("1")  # Tolerance for crossover detection
    ma_cross_confirm_bars: int = 0  # Bars to confirm crossover (0 = no confirmation)

    # Position management
    target_leverage: int = 2
    trailing_stop_pct: float = 0.01  # 1% trailing stop

    # Feature flags
    enable_shorts: bool = False
    max_adds: int = 1  # Max positions per side
    disable_new_entries: bool = False

    # Filters and guards
    filters_config: StrategyFiltersConfig | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyConfig:
        """Create config from dictionary."""
        # Extract filters config if present
        filters_data = data.pop("filters_config", None)
        if filters_data:
            filters_config = StrategyFiltersConfig(**filters_data)
        else:
            filters_config = None

        return cls(
            **{k: v for k, v in data.items() if k in cls.__annotations__},
            filters_config=filters_config,
        )


class PerpsBaselineEnhancedStrategy:
    """Baseline MA crossover strategy with liquidity and risk enhancements."""

    def __init__(
        self,
        config: StrategyConfig | None = None,
        risk_manager: LiveRiskManager | None = None,
        event_store: Any | None = None,
        bot_id: str = "perps_bot",
    ) -> None:
        """
        Initialize enhanced strategy.

        Args:
            config: Strategy configuration
            risk_manager: Risk manager for constraint checks
            event_store: Event store for metrics tracking
        """
        self.config = config or StrategyConfig()
        self.risk_manager = risk_manager
        self.event_store = event_store
        self.bot_id = bot_id

        # Initialize filters and guards
        if self.config.filters_config:
            self.market_filters = self.config.filters_config.create_filters()
            self.risk_guards = self.config.filters_config.create_guards()
            self.enhancements = StrategyEnhancements(
                rsi_period=self.config.filters_config.rsi_period,
                rsi_confirmation_enabled=self.config.filters_config.require_rsi_confirmation,
            )
        else:
            # Use defaults
            self.market_filters = create_conservative_filters()
            self.risk_guards = create_standard_risk_guards()
            self.enhancements = StrategyEnhancements()

        # In-memory state
        self.mark_windows: dict[str, list[Decimal]] = {}
        self.position_adds: dict[str, int] = {}
        self.trailing_stops: dict[str, tuple[Decimal, Decimal]] = {}

        # Metrics tracking
        self.rejection_counts: dict[str, int] = {
            "filter_spread": 0,
            "filter_depth": 0,
            "filter_volume": 0,
            "filter_rsi": 0,
            "guard_liquidation": 0,
            "guard_slippage": 0,
            "stale_data": 0,
            "entries_accepted": 0,
        }

        logger.info(
            "PerpsBaselineEnhancedStrategy initialized with filters: %s",
            self.config.filters_config,
        )

    def decide(
        self,
        symbol: str,
        current_mark: Decimal,
        position_state: dict[str, Any] | None,
        recent_marks: list[Decimal],
        equity: Decimal,
        product: Product | None = None,
        market_snapshot: dict[str, Any] | None = None,
        is_stale: bool = False,
    ) -> Decision:
        """
        Generate enhanced trading decision with filters and guards.

        Args:
            symbol: Trading symbol
            current_mark: Current mark price
            position_state: Current position info
            recent_marks: Recent price history
            equity: Account equity
            product: Product metadata
            market_snapshot: Real-time market data from WebSocket
            is_stale: Whether market data is stale

        Returns:
            Enhanced decision with rejection tracking
        """
        # Update mark window
        if symbol not in self.mark_windows:
            self.mark_windows[symbol] = []

        self.mark_windows[symbol] = (recent_marks + [current_mark])[-50:]  # Keep last 50
        marks = self.mark_windows[symbol]

        # Check for stale data
        if is_stale:
            self._record_rejection("stale_data", symbol)
            if position_state:
                # Allow reduce-only exits even with stale data
                return Decision(
                    action=Action.HOLD,
                    reduce_only=True,
                    reason="Stale market data - reduce-only mode",
                )
            else:
                return Decision(
                    action=Action.HOLD,
                    reason="Stale market data - no new entries",
                    filter_rejected=True,
                    rejection_type="stale",
                )

        # Disabled new entries check
        if self.config.disable_new_entries and not position_state:
            return Decision(action=Action.HOLD, reason="New entries disabled")

        # Calculate MAs
        if len(marks) < self.config.long_ma_period:
            return Decision(action=Action.HOLD, reason=f"Need {self.config.long_ma_period} marks")

        short_ma = sum(marks[-self.config.short_ma_period :]) / self.config.short_ma_period
        long_ma = sum(marks[-self.config.long_ma_period :]) / self.config.long_ma_period

        # Robust crossover detection with epsilon tolerance
        eps = current_mark * (self.config.ma_cross_epsilon_bps / Decimal("10000"))
        cur_diff = short_ma - long_ma

        bullish_cross = False
        bearish_cross = False

        if len(marks) >= self.config.long_ma_period + 1:
            prev_marks = marks[:-1]
            prev_short = (
                sum(prev_marks[-self.config.short_ma_period :]) / self.config.short_ma_period
            )
            prev_long = sum(prev_marks[-self.config.long_ma_period :]) / self.config.long_ma_period
            prev_diff = prev_short - prev_long

            # Detect crosses with epsilon tolerance
            bullish_cross = (prev_diff <= eps) and (cur_diff > eps)
            bearish_cross = (prev_diff >= -eps) and (cur_diff < -eps)

            # Debug logging for crossover detection
            if bullish_cross or bearish_cross:
                cross_type = "Bullish" if bullish_cross else "Bearish"
                logger.debug(
                    f"{cross_type} cross detected: prev_diff={prev_diff:.4f}, "
                    f"cur_diff={cur_diff:.4f}, eps={eps:.4f}, "
                    f"confirm_bars={self.config.ma_cross_confirm_bars}"
                )

            # Optional confirmation bars
            if self.config.ma_cross_confirm_bars > 0 and (bullish_cross or bearish_cross):
                # Check if crossover persists for required bars
                if len(marks) >= self.config.long_ma_period + self.config.ma_cross_confirm_bars + 1:
                    confirmed = True
                    for i in range(1, self.config.ma_cross_confirm_bars + 1):
                        check_marks = marks[:-i]
                        check_short = (
                            sum(check_marks[-self.config.short_ma_period :])
                            / self.config.short_ma_period
                        )
                        check_long = (
                            sum(check_marks[-self.config.long_ma_period :])
                            / self.config.long_ma_period
                        )
                        check_diff = check_short - check_long

                        if bullish_cross and check_diff <= eps:
                            confirmed = False
                            break
                        elif bearish_cross and check_diff >= -eps:
                            confirmed = False
                            break

                    bullish_cross = bullish_cross and confirmed
                    bearish_cross = bearish_cross and confirmed
                else:
                    # Not enough history for confirmation
                    bullish_cross = bearish_cross = False

        # Calculate RSI if needed
        rsi = None
        if self.config.filters_config and self.config.filters_config.require_rsi_confirmation:
            rsi = self.enhancements.calculate_rsi(marks)

        # Generate base signal
        base_signal = None
        if bullish_cross and not position_state:
            base_signal = "buy"
        elif bearish_cross and self.config.enable_shorts and not position_state:
            base_signal = "sell"
        elif position_state:
            # Check exit conditions with same robust crossover logic
            # Recalculate crosses for exit (using same epsilon)
            exit_bullish = False
            exit_bearish = False

            if len(marks) >= self.config.long_ma_period + 1:
                prev_marks = marks[:-1]
                prev_short = (
                    sum(prev_marks[-self.config.short_ma_period :]) / self.config.short_ma_period
                )
                prev_long = (
                    sum(prev_marks[-self.config.long_ma_period :]) / self.config.long_ma_period
                )
                prev_diff = prev_short - prev_long
                cur_diff = short_ma - long_ma

                exit_bullish = (prev_diff <= eps) and (cur_diff > eps)
                exit_bearish = (prev_diff >= -eps) and (cur_diff < -eps)

                # Debug logging for exit crossovers
                if exit_bullish or exit_bearish:
                    cross_type = "Bullish" if exit_bullish else "Bearish"
                    logger.debug(
                        f"Exit {cross_type} cross detected: prev_diff={prev_diff:.4f}, "
                        f"cur_diff={cur_diff:.4f}, eps={eps:.4f}"
                    )

            if position_state["side"] == "long" and exit_bearish:
                return Decision(
                    action=Action.CLOSE, reduce_only=True, reason="Bearish crossover exit"
                )
            elif position_state["side"] == "short" and exit_bullish:
                return Decision(
                    action=Action.CLOSE, reduce_only=True, reason="Bullish crossover exit"
                )

            # Check trailing stop
            if self._check_trailing_stop(symbol, current_mark, position_state):
                return Decision(action=Action.CLOSE, reduce_only=True, reason="Trailing stop hit")

        # Apply market condition filters for new entries
        if base_signal and market_snapshot:
            # Check market conditions
            if base_signal == "buy":
                allow_entry, filter_reason = self.market_filters.should_allow_long_entry(
                    market_snapshot, rsi
                )
            else:  # sell
                allow_entry, filter_reason = self.market_filters.should_allow_short_entry(
                    market_snapshot, rsi
                )

            if not allow_entry:
                # Determine rejection type
                if "Spread" in filter_reason:
                    rejection_type = "spread"
                elif "depth" in filter_reason.lower():
                    rejection_type = "depth"
                elif "volume" in filter_reason.lower():
                    rejection_type = "volume"
                elif "RSI" in filter_reason:
                    rejection_type = "rsi"
                else:
                    rejection_type = "filter"

                self._record_rejection(f"filter_{rejection_type}", symbol)

                return Decision(
                    action=Action.HOLD,
                    reason=f"Market filter rejected: {filter_reason}",
                    filter_rejected=True,
                    rejection_type=rejection_type,
                )

        # Apply risk guards for new entries
        if base_signal:
            # Calculate position sizing
            target_notional = self._calculate_position_size(equity, current_mark, product)

            # Check liquidation distance
            if product and market_snapshot:
                safe_liq, liq_reason = self.risk_guards.check_liquidation_distance(
                    entry_price=current_mark,
                    position_size=target_notional / current_mark,
                    leverage=Decimal(self.config.target_leverage),
                    account_equity=equity,
                )

                if not safe_liq:
                    self._record_rejection("guard_liquidation", symbol)
                    return Decision(
                        action=Action.HOLD,
                        reason=f"Liquidation guard: {liq_reason}",
                        guard_rejected=True,
                        rejection_type="liquidation",
                    )

                # Check slippage impact
                safe_slip, slip_reason = self.risk_guards.check_slippage_impact(
                    order_size=target_notional, market_snapshot=market_snapshot
                )

                if not safe_slip:
                    self._record_rejection("guard_slippage", symbol)
                    return Decision(
                        action=Action.HOLD,
                        reason=f"Slippage guard: {slip_reason}",
                        guard_rejected=True,
                        rejection_type="slippage",
                    )

            # All checks passed - generate entry
            self._record_acceptance(symbol)

            return Decision(
                action=Action.BUY if base_signal == "buy" else Action.SELL,
                target_notional=target_notional,
                leverage=self.config.target_leverage,
                reason=f"{'Bullish' if base_signal == 'buy' else 'Bearish'} MA crossover with RSI confirmation",
            )

        # Default hold with more detail
        ma_diff = short_ma - long_ma
        return Decision(
            action=Action.HOLD, reason=f"No signal (MA diff: {ma_diff:.2f}, eps: {eps:.2f})"
        )

    def _check_trailing_stop(
        self, symbol: str, current_price: Decimal, position_state: dict[str, Any]
    ) -> bool:
        """Check if trailing stop is hit."""
        if symbol not in self.trailing_stops:
            # Initialize trailing stop
            self.trailing_stops[symbol] = (
                current_price,
                current_price * (1 - self.config.trailing_stop_pct),
            )
            return False

        peak, stop_price = self.trailing_stops[symbol]

        # Update peak and stop for long positions
        if position_state["side"] == "long":
            if current_price > peak:
                peak = current_price
                stop_price = peak * (1 - self.config.trailing_stop_pct)
                self.trailing_stops[symbol] = (peak, stop_price)

            return current_price <= stop_price

        # Update for short positions
        if position_state["side"] == "short":
            if current_price < peak:
                peak = current_price
                stop_price = peak * (1 + self.config.trailing_stop_pct)
                self.trailing_stops[symbol] = (peak, stop_price)

            return current_price >= stop_price

        return False

    def _calculate_position_size(
        self, equity: Decimal, current_mark: Decimal, product: Product | None
    ) -> Decimal:
        """Calculate target notional position size.

        Returns:
            Target notional value in USD
        """
        # Base notional: fraction of equity with leverage
        target_notional = equity * Decimal("0.1") * self.config.target_leverage

        # Apply product constraints if available
        if product and hasattr(product, "min_size"):
            min_notional = product.min_size * current_mark
            if target_notional < min_notional:
                target_notional = min_notional * Decimal("1.1")  # 10% buffer

        return target_notional

    def _record_rejection(self, rejection_type: str, symbol: str) -> None:
        """Record rejection metrics."""
        if rejection_type in self.rejection_counts:
            self.rejection_counts[rejection_type] += 1

        if self.event_store:
            self.event_store.append_metric(
                self.bot_id,
                {
                    "type": "strategy_rejection",
                    "symbol": symbol,
                    "rejection_type": rejection_type,
                    "timestamp": datetime.now().isoformat(),
                },
            )

        logger.info(f"Rejection recorded: {rejection_type} for {symbol}")

    def _record_acceptance(self, symbol: str) -> None:
        """Record entry acceptance."""
        self.rejection_counts["entries_accepted"] += 1

        if self.event_store:
            self.event_store.append_metric(
                self.bot_id,
                {
                    "type": "strategy_acceptance",
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                },
            )

    def get_metrics(self) -> dict[str, Any]:
        """Get strategy metrics including rejection counts."""
        total_rejections = sum(
            v for k, v in self.rejection_counts.items() if k != "entries_accepted"
        )

        return {
            "rejection_counts": self.rejection_counts.copy(),
            "total_rejections": total_rejections,
            "entries_accepted": self.rejection_counts["entries_accepted"],
            "acceptance_rate": (
                self.rejection_counts["entries_accepted"]
                / (total_rejections + self.rejection_counts["entries_accepted"])
                if (total_rejections + self.rejection_counts["entries_accepted"]) > 0
                else 0
            ),
        }
