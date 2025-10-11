"""Enhanced baseline strategy for perpetuals trading with guard rails."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
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
from bot_v2.features.strategy_tools import (
    MarketConditionFilters,
    RiskGuards,
    StrategyEnhancements,
    create_conservative_filters,
    create_standard_risk_guards,
)

__all__ = [
    "Action",
    "Decision",
    "StrategyFiltersConfig",
    "StrategyConfig",
    "PerpsBaselineEnhancedStrategy",
]

logger = logging.getLogger(__name__)


def _to_decimal(value: Decimal | float | int) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


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


@dataclass
class SignalSnapshot:
    """Summary of technical signals used for decisions."""

    short_ma: Decimal
    long_ma: Decimal
    epsilon: Decimal
    bullish_cross: bool
    bearish_cross: bool
    rsi: Decimal | None

    @property
    def ma_diff(self) -> Decimal:
        return self.short_ma - self.long_ma


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
        marks = self._prepare_trading_data(symbol, recent_marks, current_mark)

        early_decision = self._check_early_guards(
            symbol=symbol,
            is_stale=is_stale,
            position_state=position_state,
            marks=marks,
        )
        if early_decision:
            return early_decision

        signal = self._calculate_trading_signals(marks, current_mark)

        exit_decision = self._evaluate_position_exits(
            symbol=symbol,
            signal=signal,
            marks=marks,
            current_mark=current_mark,
            position_state=position_state,
        )
        if exit_decision:
            return exit_decision

        base_signal = self._determine_entry_signal(signal, position_state)

        filter_decision = self._apply_market_filters(
            symbol=symbol,
            base_signal=base_signal,
            rsi=signal.rsi,
            market_snapshot=market_snapshot,
        )
        if filter_decision:
            return filter_decision

        entry_decision = self._process_entry_signal(
            symbol=symbol,
            base_signal=base_signal,
            equity=equity,
            current_mark=current_mark,
            product=product,
            market_snapshot=market_snapshot,
            position_state=position_state,
        )
        if entry_decision:
            return entry_decision

        entry_reason = (
            "Bullish MA crossover with RSI confirmation"
            if base_signal == "buy"
            else "Bearish MA crossover with RSI confirmation" if base_signal == "sell" else None
        )
        if base_signal and entry_reason:
            return create_entry_decision(
                symbol=symbol,
                action=Action.BUY if base_signal == "buy" else Action.SELL,
                equity=equity,
                product=product or self._build_default_product(symbol),
                position_fraction=0.1,
                target_leverage=self.config.target_leverage,
                max_trade_usd=None,
                position_adds=self.position_adds,
                trailing_stops=self.trailing_stops,
                reason=entry_reason,
            )

        return self._build_hold_decision(signal)

        if entry_decision:
            return entry_decision

        return self._build_hold_decision(signal)

    def _prepare_trading_data(
        self, symbol: str, recent_marks: list[Decimal], current_mark: Decimal
    ) -> list[Decimal]:
        """Maintain mark history and return the updated window."""
        return update_mark_window(
            self.mark_windows,
            symbol=symbol,
            current_mark=current_mark,
            short_period=self.config.short_ma_period,
            long_period=self.config.long_ma_period,
            recent_marks=recent_marks,
            buffer=30,
        )

    def _check_early_guards(
        self,
        *,
        symbol: str,
        is_stale: bool,
        position_state: dict[str, Any] | None,
        marks: list[Decimal],
    ) -> Decision | None:
        """Run early exit checks before signal evaluation."""
        if is_stale:
            self._record_rejection("stale_data", symbol)
            if position_state:
                return Decision(
                    action=Action.HOLD,
                    reduce_only=True,
                    reason="Stale market data - reduce-only mode",
                )
            return Decision(
                action=Action.HOLD,
                reason="Stale market data - no new entries",
                filter_rejected=True,
                rejection_type="stale",
            )

        if self.config.disable_new_entries and not position_state:
            return Decision(action=Action.HOLD, reason="New entries disabled")

        if len(marks) < self.config.long_ma_period:
            return Decision(action=Action.HOLD, reason=f"Need {self.config.long_ma_period} marks")

        return None

    def _calculate_trading_signals(
        self, marks: list[Decimal], current_mark: Decimal
    ) -> SignalSnapshot:
        """Compute moving averages, crossovers, and supporting indicators."""
        snapshot = calculate_ma_snapshot(
            marks,
            short_period=self.config.short_ma_period,
            long_period=self.config.long_ma_period,
            epsilon_bps=self.config.ma_cross_epsilon_bps,
            confirm_bars=self.config.ma_cross_confirm_bars,
        )

        if snapshot.bullish_cross or snapshot.bearish_cross:
            cross_type = "Bullish" if snapshot.bullish_cross else "Bearish"
            logger.debug(
                "%s cross detected: short=%.4f long=%.4f eps=%.4f",
                cross_type,
                float(snapshot.short_ma),
                float(snapshot.long_ma),
                float(snapshot.epsilon),
            )

        rsi = None
        if self.config.filters_config and self.config.filters_config.require_rsi_confirmation:
            rsi_raw = self.enhancements.calculate_rsi(marks)
            rsi = Decimal(str(rsi_raw)) if rsi_raw is not None else None

        return SignalSnapshot(
            short_ma=snapshot.short_ma,
            long_ma=snapshot.long_ma,
            epsilon=snapshot.epsilon,
            bullish_cross=snapshot.bullish_cross,
            bearish_cross=snapshot.bearish_cross,
            rsi=rsi,
        )

    def _evaluate_position_exits(
        self,
        *,
        symbol: str,
        signal: SignalSnapshot,
        marks: list[Decimal],
        current_mark: Decimal,
        position_state: dict[str, Any] | None,
    ) -> Decision | None:
        """Check exit crossovers and trailing stops for existing positions."""
        if not position_state:
            return None

        exit_bullish = False
        exit_bearish = False

        if len(marks) >= self.config.long_ma_period + 1:
            prev_marks = marks[:-1]
            prev_short = sum(prev_marks[-self.config.short_ma_period :], Decimal("0")) / Decimal(
                self.config.short_ma_period
            )
            prev_long = sum(prev_marks[-self.config.long_ma_period :], Decimal("0")) / Decimal(
                self.config.long_ma_period
            )
            prev_diff = prev_short - prev_long
            cur_diff = signal.short_ma - signal.long_ma

            exit_bullish = (prev_diff <= signal.epsilon) and (cur_diff > signal.epsilon)
            exit_bearish = (prev_diff >= -signal.epsilon) and (cur_diff < -signal.epsilon)

            if exit_bullish or exit_bearish:
                cross_type = "Bullish" if exit_bullish else "Bearish"
                logger.debug(
                    "Exit %s cross detected: prev_diff=%.4f, cur_diff=%.4f, eps=%.4f",
                    cross_type,
                    float(prev_diff),
                    float(cur_diff),
                    float(signal.epsilon),
                )

        side = position_state.get("side") if position_state else None
        if side == "long" and exit_bearish:
            return create_close_decision(
                symbol=symbol,
                position_state=position_state,
                position_adds=self.position_adds,
                trailing_stops=self.trailing_stops,
                reason="Bearish crossover exit",
            )
        if side == "short" and exit_bullish:
            return create_close_decision(
                symbol=symbol,
                position_state=position_state,
                position_adds=self.position_adds,
                trailing_stops=self.trailing_stops,
                reason="Bullish crossover exit",
            )

        if position_state and self._check_trailing_stop(symbol, current_mark, position_state):
            return create_close_decision(
                symbol=symbol,
                position_state=position_state,
                position_adds=self.position_adds,
                trailing_stops=self.trailing_stops,
                reason="Trailing stop hit",
            )

        return None

    def _determine_entry_signal(
        self, signal: SignalSnapshot, position_state: dict[str, Any] | None
    ) -> str | None:
        """Decide on potential entry direction when flat."""
        if position_state:
            return None
        if signal.bullish_cross:
            return "buy"
        if signal.bearish_cross and self.config.enable_shorts:
            return "sell"
        return None

    def _apply_market_filters(
        self,
        *,
        symbol: str,
        base_signal: str | None,
        rsi: Decimal | None,
        market_snapshot: dict[str, Any] | None,
    ) -> Decision | None:
        """Apply market condition filters for prospective entries."""
        if not base_signal or market_snapshot is None:
            return None

        if base_signal == "buy":
            allow_entry, filter_reason = self.market_filters.should_allow_long_entry(
                market_snapshot, rsi
            )
        else:
            allow_entry, filter_reason = self.market_filters.should_allow_short_entry(
                market_snapshot, rsi
            )

        if allow_entry:
            return None

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

    def _build_hold_decision(self, signal: SignalSnapshot) -> Decision:
        """Produce the default hold decision with diagnostics."""
        return Decision(
            action=Action.HOLD,
            reason=f"No signal (MA diff: {float(signal.ma_diff):.2f}, eps: {float(signal.epsilon):.2f})",
        )

    def _check_trailing_stop(
        self, symbol: str, current_price: Decimal, position_state: dict[str, Any]
    ) -> bool:
        """Check if trailing stop is hit."""
        side = position_state.get("side", "").lower()
        return update_trailing_stop(
            self.trailing_stops,
            symbol=symbol,
            side=side,
            current_price=current_price,
            trailing_pct=_to_decimal(self.config.trailing_stop_pct),
        )

    def _calculate_position_size(
        self, equity: Decimal, current_mark: Decimal, product: Product | None
    ) -> Decimal:
        """Calculate target notional position size.

        Returns:
            Target notional value in USD
        """
        # Base notional: fraction of equity with leverage
        target_notional = equity * Decimal("0.1") * Decimal(str(self.config.target_leverage))

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
