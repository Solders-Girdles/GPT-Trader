"""Baseline MA crossover strategy with liquidity and risk enhancements."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from functools import lru_cache
from typing import Any

from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.features.live_trade.strategies.decisions import Action, Decision
from bot_v2.features.live_trade.strategies.shared import (
    create_close_decision,
    create_entry_decision,
)
from bot_v2.features.strategy_tools import (
    create_conservative_filters,
    create_standard_risk_guards,
)
from bot_v2.utilities.logging_patterns import get_logger

from .config import StrategyConfig, StrategyFiltersConfig
from .signals import StrategySignal, build_signal
from .state import StrategyState

logger = get_logger(__name__, component="live_trade_strategy")


def _to_decimal(value: Decimal | float | int) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


class PerpsBaselineEnhancedStrategy:
    """Baseline MA crossover strategy with liquidity and risk enhancements."""

    def __init__(
        self,
        *,
        config: StrategyConfig | None = None,
        risk_manager: LiveRiskManager | None = None,
        event_store: Any | None = None,
        bot_id: str = "coinbase_trader",
        state: StrategyState | None = None,
    ) -> None:
        """
        Initialize enhanced strategy.

        Args:
            config: Strategy configuration
            risk_manager: Risk manager for constraint checks
            event_store: Event store for metrics tracking
            bot_id: Identifier for telemetry batching
            state: Optional state container (primarily for testing/runtime hydration)
        """
        self.config = config or StrategyConfig()
        self.risk_manager = risk_manager
        self.event_store = event_store
        self.bot_id = bot_id
        self.state = state or StrategyState()

        filters_config: StrategyFiltersConfig | None = self.config.filters_config
        if filters_config:
            self.market_filters = filters_config.create_filters()
            self.risk_guards = filters_config.create_guards()
        else:
            self.market_filters = create_conservative_filters()
            self.risk_guards = create_standard_risk_guards()

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

        signal = build_signal(
            marks=marks,
            config=self.config,
        )

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
            signal=signal,
            base_signal=base_signal,
            equity=equity,
            current_mark=current_mark,
            product=product,
            market_snapshot=market_snapshot,
            position_state=position_state,
        )
        if entry_decision:
            return entry_decision

        return self._build_hold_decision(signal)

    def reset(self, symbol: str | None = None) -> None:
        """Reset strategy state (per symbol or globally)."""
        self.state.reset(symbol)

    # ----------------------------------------------------------------- Helpers
    def _prepare_trading_data(
        self, symbol: str, recent_marks: list[Decimal], current_mark: Decimal
    ) -> list[Decimal]:
        """Maintain mark history and return the updated window."""
        return self.state.update_mark_window(
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

    def _evaluate_position_exits(
        self,
        *,
        symbol: str,
        signal: StrategySignal,
        marks: list[Decimal],
        current_mark: Decimal,
        position_state: dict[str, Any] | None,
    ) -> Decision | None:
        """Check exit crossovers and trailing stops for existing positions."""
        if not position_state:
            return None

        exit_bullish = False
        exit_bearish = False
        snapshot = signal.snapshot

        if len(marks) >= self.config.long_ma_period + 1:
            prev_marks = marks[:-1]
            prev_short = sum(prev_marks[-self.config.short_ma_period :], Decimal("0")) / Decimal(
                self.config.short_ma_period
            )
            prev_long = sum(prev_marks[-self.config.long_ma_period :], Decimal("0")) / Decimal(
                self.config.long_ma_period
            )
            prev_diff = prev_short - prev_long
            cur_diff = snapshot.short_ma - snapshot.long_ma

            exit_bullish = (prev_diff <= snapshot.epsilon) and (cur_diff > snapshot.epsilon)
            exit_bearish = (prev_diff >= -snapshot.epsilon) and (cur_diff < -snapshot.epsilon)

            if exit_bullish or exit_bearish:
                cross_type = "Bullish" if exit_bullish else "Bearish"
                logger.debug(
                    "Exit %s cross detected: prev_diff=%.4f, cur_diff=%.4f, eps=%.4f",
                    cross_type,
                    float(prev_diff),
                    float(cur_diff),
                    float(snapshot.epsilon),
                )

        side = (position_state.get("side") or "").lower()
        if side == "long" and exit_bearish:
            return create_close_decision(
                symbol=symbol,
                position_state=position_state,
                position_adds=self.state.position_adds,
                trailing_stops=self.state.trailing_stops,
                reason="Bearish crossover exit",
            )
        if side == "short" and exit_bullish:
            return create_close_decision(
                symbol=symbol,
                position_state=position_state,
                position_adds=self.state.position_adds,
                trailing_stops=self.state.trailing_stops,
                reason="Bullish crossover exit",
            )

        trailing_hit = self.state.update_trailing_stop(
            symbol=symbol,
            side=side,
            current_price=current_mark,
            trailing_pct=_to_decimal(self.config.trailing_stop_pct),
        )
        if trailing_hit:
            return create_close_decision(
                symbol=symbol,
                position_state=position_state,
                position_adds=self.state.position_adds,
                trailing_stops=self.state.trailing_stops,
                reason="Trailing stop hit",
            )

        return None

    def _determine_entry_signal(
        self, signal: StrategySignal, position_state: dict[str, Any] | None
    ) -> str | None:
        """Decide on potential entry direction when flat."""
        if position_state:
            return None
        if signal.is_bullish:
            return "buy"
        if signal.is_bearish and self.config.enable_shorts:
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

    def _process_entry_signal(
        self,
        *,
        symbol: str,
        signal: StrategySignal,
        base_signal: str | None,
        equity: Decimal,
        current_mark: Decimal,
        product: Product | None,
        market_snapshot: dict[str, Any] | None,
        position_state: dict[str, Any] | None,
    ) -> Decision | None:
        """Run guard checks and create an entry decision when appropriate."""
        if position_state or not base_signal:
            return None

        product_meta = product or self._build_default_product(symbol)
        guard_decision = self._apply_risk_guards(
            symbol=symbol,
            base_signal=base_signal,
            equity=equity,
            current_mark=current_mark,
            product=product_meta,
            market_snapshot=market_snapshot,
        )
        if guard_decision:
            return guard_decision

        reason = "Bullish MA crossover" if base_signal == "buy" else "Bearish MA crossover"
        if signal.rsi is not None:
            reason += f" with RSI {float(signal.rsi):.2f}"

        action = Action.BUY if base_signal == "buy" else Action.SELL

        self._record_acceptance(symbol)
        return create_entry_decision(
            symbol=symbol,
            action=action,
            equity=equity,
            product=product_meta,
            position_fraction=0.1,
            target_leverage=self.config.target_leverage,
            max_trade_usd=None,
            position_adds=self.state.position_adds,
            trailing_stops=self.state.trailing_stops,
            reason=reason,
        )

    def _apply_risk_guards(
        self,
        *,
        symbol: str,
        base_signal: str,
        equity: Decimal,
        current_mark: Decimal,
        product: Product,
        market_snapshot: dict[str, Any] | None,
    ) -> Decision | None:
        """Evaluate risk guards prior to submitting an entry."""
        if not market_snapshot:
            return None

        target_notional = self._calculate_position_size(equity, current_mark, product)

        ok_liq, liq_reason = self.risk_guards.check_liquidation_distance(
            entry_price=current_mark,
            position_size=target_notional,
            leverage=Decimal(str(max(self.config.target_leverage, 1))),
            account_equity=equity,
        )
        if not ok_liq:
            self._record_rejection("guard_liquidation", symbol)
            return Decision(
                action=Action.HOLD,
                reason=f"Risk guard rejected entry: {liq_reason}",
                guard_rejected=True,
                rejection_type="liquidation",
            )

        ok_slip, slip_reason = self.risk_guards.check_slippage_impact(
            order_size=target_notional,
            market_snapshot=market_snapshot,
        )
        if not ok_slip:
            self._record_rejection("guard_slippage", symbol)
            return Decision(
                action=Action.HOLD,
                reason=f"Risk guard rejected entry: {slip_reason}",
                guard_rejected=True,
                rejection_type="slippage",
            )

        return None

    def _build_hold_decision(self, signal: StrategySignal) -> Decision:
        """Produce the default hold decision with diagnostics."""
        snapshot = signal.snapshot
        return Decision(
            action=Action.HOLD,
            reason=f"No signal (MA diff: {float(snapshot.diff):.2f}, eps: {float(snapshot.epsilon):.2f})",
        )

    def _calculate_position_size(
        self, equity: Decimal, current_mark: Decimal, product: Product | None
    ) -> Decimal:
        """Calculate target notional position size."""
        target_notional = equity * Decimal("0.1") * Decimal(str(self.config.target_leverage))

        if product and getattr(product, "min_size", None):
            min_notional = product.min_size * current_mark
            if target_notional < min_notional:
                target_notional = min_notional * Decimal("1.1")

        return target_notional

    def _record_rejection(self, rejection_type: str, symbol: str) -> None:
        """Record rejection metrics."""
        self.state.record_rejection(rejection_type)

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

        logger.info("Rejection recorded: %s for %s", rejection_type, symbol)

    def _record_acceptance(self, symbol: str) -> None:
        """Record entry acceptance."""
        self.state.record_acceptance()

        if self.event_store:
            self.event_store.append_metric(
                self.bot_id,
                {
                    "type": "strategy_acceptance",
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                },
            )

    def _build_default_product(self, symbol: str) -> Product:
        """Build a minimal Product stub when metadata is unavailable."""
        target_leverage = int(self.config.target_leverage)
        return _default_product_stub(symbol, target_leverage)

    def get_metrics(self) -> dict[str, Any]:
        """Get strategy metrics including rejection counts."""
        return self.state.get_metrics()


__all__ = ["PerpsBaselineEnhancedStrategy"]


@lru_cache(maxsize=128)
def _default_product_stub(symbol: str, target_leverage: int) -> Product:
    """Return a cached minimal product descriptor for fallback sizing."""
    parts = symbol.split("-")
    base_asset = parts[0] if parts else symbol
    quote_asset = parts[-1] if len(parts) > 1 else "USD"

    return Product(
        symbol=symbol,
        base_asset=base_asset,
        quote_asset=quote_asset,
        market_type=MarketType.PERPETUAL,
        min_size=Decimal("0"),
        step_size=Decimal("0.0001"),
        min_notional=None,
        price_increment=Decimal("0.01"),
        leverage_max=target_leverage,
        expiry=None,
        contract_size=None,
        funding_rate=None,
        next_funding_time=None,
    )
