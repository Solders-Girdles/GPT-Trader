"""
Strategy decision pipeline for trading decisions.

Extracts decision logic (guards, filters, exits, entries, position sizing) from strategy
classes for focused testing and reusability. Orchestrates the decision flow from signal
evaluation to action selection with proper guard/filter enforcement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.core.interfaces import Product
from bot_v2.features.live_trade.risk import LiveRiskManager, PositionSizingContext
from bot_v2.features.live_trade.strategies.decisions import Action, Decision
from bot_v2.features.live_trade.strategies.strategy_signals import SignalSnapshot
from bot_v2.features.strategy_tools import MarketConditionFilters, RiskGuards

logger = logging.getLogger(__name__)


def _to_decimal(value: Decimal | float | int) -> Decimal:
    """Convert value to Decimal safely."""
    return value if isinstance(value, Decimal) else Decimal(str(value))


@dataclass
class DecisionContext:
    """
    Context for strategy decision making.

    Encapsulates all inputs needed for the decision pipeline.
    """

    # Symbol and signal
    symbol: str
    signal: SignalSnapshot
    current_mark: Decimal

    # Account and position state
    equity: Decimal
    position_state: dict[str, Any] | None

    # Market data
    market_snapshot: dict[str, Any] | None
    is_stale: bool
    marks: list[Decimal]

    # Product metadata
    product: Product | None

    # Configuration
    enable_shorts: bool
    disable_new_entries: bool
    target_leverage: int
    trailing_stop_pct: float
    long_ma_period: int
    short_ma_period: int
    epsilon: Decimal

    # Current trailing stop state (peak, stop_price) for this symbol
    trailing_stop_state: tuple[Decimal, Decimal] | None = None


@dataclass
class DecisionResult:
    """
    Result from decision pipeline.

    Encapsulates the trading decision plus updated state.
    """

    decision: Decision
    updated_trailing_stop: tuple[Decimal, Decimal] | None = None
    rejection_type: str | None = None  # For metrics tracking


class StrategyDecisionPipeline:
    """
    Decision pipeline for trading strategies.

    Responsibilities:
    - Early guard checks (stale data, disabled entries, insufficient data)
    - Market filter evaluation (spread, depth, volume, RSI)
    - Position exit logic (opposing signals, trailing stops)
    - Entry signal processing (risk guards, position sizing)
    - Trailing stop management

    Stateless - all state passed via DecisionContext.
    """

    def __init__(
        self,
        market_filters: MarketConditionFilters,
        risk_guards: RiskGuards,
        risk_manager: LiveRiskManager | None = None,
    ) -> None:
        """
        Initialize decision pipeline.

        Args:
            market_filters: Market condition filters for entry validation
            risk_guards: Risk guards for safety checks
            risk_manager: Optional risk manager for position sizing
        """
        self.market_filters = market_filters
        self.risk_guards = risk_guards
        self.risk_manager = risk_manager

    def evaluate(self, context: DecisionContext) -> DecisionResult:
        """
        Evaluate trading decision for given context.

        Pipeline flow:
        1. Check early guards (stale data, disabled entries, insufficient data)
        2. Calculate signals (already done - passed in context)
        3. Evaluate position exits (crossovers, trailing stops)
        4. Determine entry signal (if flat)
        5. Apply market filters
        6. Process entry (risk guards, position sizing)
        7. Return hold decision if no action

        Args:
            context: Decision context with signal, position, market data

        Returns:
            DecisionResult with trading decision and updated state
        """
        # Step 1: Early guards
        early_decision = self._check_early_guards(context)
        if early_decision:
            return early_decision

        # Step 2: Position exits (opposing signals, trailing stops)
        exit_decision = self._evaluate_position_exits(context)
        if exit_decision:
            return exit_decision

        # Step 3: Entry signal determination
        base_signal = self._determine_entry_signal(context)

        # Step 4: Market filters
        filter_decision = self._apply_market_filters(context, base_signal)
        if filter_decision:
            return filter_decision

        # Step 5: Entry processing (risk guards, position sizing)
        entry_decision = self._process_entry_signal(context, base_signal)
        if entry_decision:
            return entry_decision

        # Step 6: Default hold
        return DecisionResult(
            decision=Decision(
                action=Action.HOLD,
                reason=f"No signal (MA diff: {float(context.signal.ma_diff):.2f}, eps: {float(context.epsilon):.2f})",
            )
        )

    def _check_early_guards(self, context: DecisionContext) -> DecisionResult | None:
        """Check early exit conditions before signal evaluation."""
        # Stale data guard
        if context.is_stale:
            if context.position_state:
                decision = Decision(
                    action=Action.HOLD,
                    reduce_only=True,
                    reason="Stale market data - reduce-only mode",
                )
            else:
                decision = Decision(
                    action=Action.HOLD,
                    reason="Stale market data - no new entries",
                    filter_rejected=True,
                    rejection_type="stale",
                )
            return DecisionResult(decision=decision, rejection_type="stale_data")

        # Disabled entries guard
        if context.disable_new_entries and not context.position_state:
            return DecisionResult(
                decision=Decision(action=Action.HOLD, reason="New entries disabled")
            )

        # Insufficient data guard
        if len(context.marks) < context.long_ma_period:
            return DecisionResult(
                decision=Decision(action=Action.HOLD, reason=f"Need {context.long_ma_period} marks")
            )

        return None

    def _evaluate_position_exits(self, context: DecisionContext) -> DecisionResult | None:
        """Check exit crossovers and trailing stops for existing positions."""
        if not context.position_state:
            return None

        # Check opposing crossover exits
        exit_decision = self._check_crossover_exits(context)
        if exit_decision:
            return exit_decision

        # Check trailing stop
        trailing_result = self._check_trailing_stop(context)
        if trailing_result:
            return trailing_result

        return None

    def _check_crossover_exits(self, context: DecisionContext) -> DecisionResult | None:
        """Check for opposing crossover exits."""
        if len(context.marks) < context.long_ma_period + 1:
            return None

        # Calculate previous MA crossover state
        prev_marks = context.marks[:-1]
        prev_short = sum(prev_marks[-context.short_ma_period :], Decimal("0")) / Decimal(
            context.short_ma_period
        )
        prev_long = sum(prev_marks[-context.long_ma_period :], Decimal("0")) / Decimal(
            context.long_ma_period
        )
        prev_diff = prev_short - prev_long
        cur_diff = context.signal.short_ma - context.signal.long_ma

        # Detect exit crossovers
        exit_bullish = (prev_diff <= context.epsilon) and (cur_diff > context.epsilon)
        exit_bearish = (prev_diff >= -context.epsilon) and (cur_diff < -context.epsilon)

        if exit_bullish or exit_bearish:
            cross_type = "Bullish" if exit_bullish else "Bearish"
            logger.debug(
                "Exit %s cross detected: prev_diff=%.4f, cur_diff=%.4f, eps=%.4f",
                cross_type,
                float(prev_diff),
                float(cur_diff),
                float(context.epsilon),
            )

        # Exit logic
        side = context.position_state.get("side") if context.position_state else None
        if side == "long" and exit_bearish:
            return DecisionResult(
                decision=Decision(
                    action=Action.CLOSE, reduce_only=True, reason="Bearish crossover exit"
                )
            )
        if side == "short" and exit_bullish:
            return DecisionResult(
                decision=Decision(
                    action=Action.CLOSE, reduce_only=True, reason="Bullish crossover exit"
                )
            )

        return None

    def _check_trailing_stop(self, context: DecisionContext) -> DecisionResult | None:
        """Check if trailing stop is hit and update stop state."""
        if not context.position_state:
            return None

        trailing_pct = _to_decimal(context.trailing_stop_pct)
        current_price = context.current_mark

        # Initialize trailing stop if needed
        if context.trailing_stop_state is None:
            updated_stop = (
                current_price,
                current_price * (Decimal("1") - trailing_pct),
            )
            return DecisionResult(decision=None, updated_trailing_stop=updated_stop)

        peak, stop_price = context.trailing_stop_state
        side = context.position_state["side"]

        # Update trailing stop for long positions
        if side == "long":
            if current_price > peak:
                peak = current_price
                stop_price = peak * (Decimal("1") - trailing_pct)

            if current_price <= stop_price:
                return DecisionResult(
                    decision=Decision(
                        action=Action.CLOSE, reduce_only=True, reason="Trailing stop hit"
                    ),
                    updated_trailing_stop=(peak, stop_price),
                )

            return DecisionResult(decision=None, updated_trailing_stop=(peak, stop_price))

        # Update trailing stop for short positions
        if side == "short":
            if current_price < peak:
                peak = current_price
                stop_price = peak * (Decimal("1") + trailing_pct)

            if current_price >= stop_price:
                return DecisionResult(
                    decision=Decision(
                        action=Action.CLOSE, reduce_only=True, reason="Trailing stop hit"
                    ),
                    updated_trailing_stop=(peak, stop_price),
                )

            return DecisionResult(decision=None, updated_trailing_stop=(peak, stop_price))

        return None

    def _determine_entry_signal(self, context: DecisionContext) -> str | None:
        """Determine potential entry direction when flat."""
        if context.position_state:
            return None

        if context.signal.bullish_cross:
            return "buy"

        if context.signal.bearish_cross and context.enable_shorts:
            return "sell"

        return None

    def _apply_market_filters(
        self, context: DecisionContext, base_signal: str | None
    ) -> DecisionResult | None:
        """Apply market condition filters for prospective entries."""
        if not base_signal or context.market_snapshot is None:
            return None

        # Check filters
        if base_signal == "buy":
            allow_entry, filter_reason = self.market_filters.should_allow_long_entry(
                context.market_snapshot, context.signal.rsi
            )
        else:
            allow_entry, filter_reason = self.market_filters.should_allow_short_entry(
                context.market_snapshot, context.signal.rsi
            )

        if allow_entry:
            return None

        # Determine rejection type for metrics
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

        return DecisionResult(
            decision=Decision(
                action=Action.HOLD,
                reason=f"Market filter rejected: {filter_reason}",
                filter_rejected=True,
                rejection_type=rejection_type,
            ),
            rejection_type=f"filter_{rejection_type}",
        )

    def _process_entry_signal(
        self, context: DecisionContext, base_signal: str | None
    ) -> DecisionResult | None:
        """Run risk guards and size the position when an entry exists."""
        if not base_signal:
            return None

        # Calculate base position size
        target_notional = self._calculate_position_size(context)

        # Apply risk manager position sizing if available
        if self.risk_manager and base_signal:
            sizing_context = PositionSizingContext(
                symbol=context.symbol,
                side=base_signal,
                equity=context.equity,
                current_price=context.current_mark,
                strategy_name="PerpsBaselineEnhancedStrategy",
                method=getattr(self.risk_manager.config, "position_sizing_method", "intelligent"),
                current_position_quantity=(
                    Decimal(str(context.position_state.get("quantity", "0")))
                    if context.position_state
                    else Decimal("0")
                ),
                target_leverage=Decimal(str(context.target_leverage)),
                product=context.product,
                strategy_multiplier=float(
                    getattr(self.risk_manager.config, "position_sizing_multiplier", 1.0)
                ),
            )
            advice = self.risk_manager.size_position(sizing_context)
            target_notional = advice.target_notional

            if advice.reduce_only and target_notional == 0:
                return DecisionResult(
                    decision=Decision(
                        action=Action.HOLD,
                        reason=advice.reason or "Position sizing prevented entry",
                        guard_rejected=True,
                        rejection_type="position_sizing",
                    ),
                    rejection_type="risk_position_sizing",
                )

        # Apply risk guards
        if context.product and context.market_snapshot:
            # Liquidation distance guard
            safe_liq, liq_reason = self.risk_guards.check_liquidation_distance(
                entry_price=context.current_mark,
                position_size=target_notional / context.current_mark,
                leverage=Decimal(context.target_leverage),
                account_equity=context.equity,
            )

            if not safe_liq:
                return DecisionResult(
                    decision=Decision(
                        action=Action.HOLD,
                        reason=f"Liquidation guard: {liq_reason}",
                        guard_rejected=True,
                        rejection_type="liquidation",
                    ),
                    rejection_type="guard_liquidation",
                )

            # Slippage impact guard
            safe_slip, slip_reason = self.risk_guards.check_slippage_impact(
                order_size=target_notional, market_snapshot=context.market_snapshot
            )

            if not safe_slip:
                return DecisionResult(
                    decision=Decision(
                        action=Action.HOLD,
                        reason=f"Slippage guard: {slip_reason}",
                        guard_rejected=True,
                        rejection_type="slippage",
                    ),
                    rejection_type="guard_slippage",
                )

        # All guards passed - return entry decision
        return DecisionResult(
            decision=Decision(
                action=Action.BUY if base_signal == "buy" else Action.SELL,
                target_notional=target_notional,
                leverage=context.target_leverage,
                reason=f"{'Bullish' if base_signal == 'buy' else 'Bearish'} MA crossover with RSI confirmation",
            ),
            rejection_type="entries_accepted",  # For metrics tracking
        )

    def _calculate_position_size(self, context: DecisionContext) -> Decimal:
        """
        Calculate target notional position size.

        Returns:
            Target notional value in USD
        """
        # Base notional: 10% of equity with leverage
        target_notional = context.equity * Decimal("0.1") * Decimal(str(context.target_leverage))

        # Apply product constraints if available
        if context.product and hasattr(context.product, "min_size"):
            min_notional = context.product.min_size * context.current_mark
            if target_notional < min_notional:
                target_notional = min_notional * Decimal("1.1")  # 10% buffer

        return target_notional
