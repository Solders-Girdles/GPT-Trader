"""
Enhanced baseline strategy for perpetuals trading with guard rails.

This module provides a modular MA crossover strategy with:
- Signal calculation via StrategySignalCalculator (moving averages, crossovers, RSI)
- Decision logic via StrategyDecisionPipeline (guards, filters, exits, entries, sizing)
- Facade pattern for clean integration and state management
- Comprehensive filtering and risk guards for safe trading
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.core.interfaces import Product
from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.features.live_trade.strategies.decision_pipeline import (
    DecisionContext,
    DecisionResult,
    StrategyDecisionPipeline,
)
from bot_v2.features.live_trade.strategies.decisions import Action, Decision
from bot_v2.features.live_trade.strategies.strategy_signals import (
    SignalSnapshot,
    StrategySignalCalculator,
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
    "SignalSnapshot",
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


class PerpsBaselineEnhancedStrategy:
    """
    Baseline MA crossover strategy with liquidity and risk enhancements.

    Architecture:
    - Delegates signal calculation to StrategySignalCalculator
    - Delegates decision logic to StrategyDecisionPipeline
    - Manages state (mark windows, trailing stops, metrics)
    - Facade pattern for clean integration

    Decision flow:
    1. Prepare data (maintain mark history)
    2. Calculate signals (MA crossovers, RSI)
    3. Evaluate decision via pipeline (guards, filters, exits, entries, sizing)
    4. Apply result (update trailing stops, record metrics)
    """

    def __init__(
        self,
        config: StrategyConfig | None = None,
        risk_manager: LiveRiskManager | None = None,
        event_store: Any | None = None,
        bot_id: str = "perps_bot",
        signal_calculator: StrategySignalCalculator | None = None,
        decision_pipeline: StrategyDecisionPipeline | None = None,
    ) -> None:
        """
        Initialize enhanced strategy.

        Args:
            config: Strategy configuration
            risk_manager: Risk manager for constraint checks
            event_store: Event store for metrics tracking
            bot_id: Bot identifier for logging
            signal_calculator: Optional signal calculator (for testing/injection)
            decision_pipeline: Optional decision pipeline (for testing/injection)
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

        # Initialize signal calculator
        if signal_calculator:
            self.signal_calculator = signal_calculator
        else:
            self.signal_calculator = StrategySignalCalculator(
                short_ma_period=self.config.short_ma_period,
                long_ma_period=self.config.long_ma_period,
                ma_cross_epsilon_bps=self.config.ma_cross_epsilon_bps,
                ma_cross_confirm_bars=self.config.ma_cross_confirm_bars,
                rsi_calculator=self.enhancements,
            )

        # Initialize decision pipeline
        if decision_pipeline:
            self.decision_pipeline = decision_pipeline
        else:
            self.decision_pipeline = StrategyDecisionPipeline(
                market_filters=self.market_filters,
                risk_guards=self.risk_guards,
                risk_manager=self.risk_manager,
            )

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
        # Prepare data (maintain mark history)
        marks = self._prepare_trading_data(symbol, recent_marks, current_mark)

        # Calculate signal
        signal = self._calculate_trading_signals(marks, current_mark)

        # Build decision context
        context = DecisionContext(
            symbol=symbol,
            signal=signal,
            current_mark=current_mark,
            equity=equity,
            position_state=position_state,
            market_snapshot=market_snapshot,
            is_stale=is_stale,
            marks=marks,
            product=product,
            enable_shorts=self.config.enable_shorts,
            disable_new_entries=self.config.disable_new_entries,
            target_leverage=self.config.target_leverage,
            trailing_stop_pct=self.config.trailing_stop_pct,
            long_ma_period=self.config.long_ma_period,
            short_ma_period=self.config.short_ma_period,
            epsilon=signal.epsilon,
            trailing_stop_state=self.trailing_stops.get(symbol),
        )

        # Evaluate decision via pipeline
        result = self.decision_pipeline.evaluate(context)

        # Apply result (update state, record metrics)
        self._apply_decision_result(result, symbol)

        # Return decision
        return result.decision

    def _apply_decision_result(self, result: DecisionResult, symbol: str) -> None:
        """
        Apply decision result to strategy state.

        Updates trailing stops and records rejection metrics.

        Args:
            result: Decision result from pipeline
            symbol: Trading symbol
        """
        # Update trailing stops if changed
        if result.updated_trailing_stop is not None:
            self.trailing_stops[symbol] = result.updated_trailing_stop

        # Record metrics
        if result.rejection_type:
            if result.rejection_type == "entries_accepted":
                self._record_acceptance(symbol)
            else:
                self._record_rejection(result.rejection_type, symbol)

    def _prepare_trading_data(
        self, symbol: str, recent_marks: list[Decimal], current_mark: Decimal
    ) -> list[Decimal]:
        """Maintain mark history and return the updated window."""
        window = self.mark_windows.setdefault(symbol, [])
        updated_window = (recent_marks + [current_mark])[-50:]
        if window != updated_window:
            self.mark_windows[symbol] = updated_window
        return self.mark_windows[symbol]

    def _calculate_trading_signals(
        self, marks: list[Decimal], current_mark: Decimal
    ) -> SignalSnapshot:
        """Compute moving averages, crossovers, and supporting indicators (delegates to calculator)."""
        # Determine if RSI is required
        require_rsi = (
            self.config.filters_config is not None
            and self.config.filters_config.require_rsi_confirmation
        )

        # Delegate to signal calculator
        return self.signal_calculator.calculate_signals(
            marks, current_mark, require_rsi=require_rsi
        )

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
