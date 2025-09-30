"""
Risk management for perpetuals live trading.

Phase 5: Leverage-aware sizing and runtime guards.
Complete isolation - no strategy logic, pure risk controls.

This module now serves as a delegation facade to focused submodules.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.brokerages.core.interfaces import Product
from bot_v2.features.live_trade.risk.position_sizing import (
    ImpactAssessment,
    ImpactRequest,
    PositionSizer,
    PositionSizingAdvice,
    PositionSizingContext,
)
from bot_v2.features.live_trade.risk.pre_trade_checks import (
    PreTradeValidator,
    ValidationError,
)
from bot_v2.features.live_trade.risk.runtime_monitoring import RuntimeMonitor
from bot_v2.features.live_trade.risk.state_management import (
    RiskRuntimeState,
    StateManager,
)
from bot_v2.features.live_trade.risk_runtime import CircuitBreakerOutcome
from bot_v2.persistence.event_store import EventStore

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    "LiveRiskManager",
    "ValidationError",
    "PositionSizingContext",
    "PositionSizingAdvice",
    "ImpactRequest",
    "ImpactAssessment",
    "RiskRuntimeState",
]


class LiveRiskManager:
    """Risk management for perpetuals live trading.

    Enforces leverage limits, liquidation buffers, exposure caps,
    and runtime guards for safe trading.

    This class now delegates to focused helper modules for better organization.
    """

    def __init__(
        self,
        config: RiskConfig | None = None,
        event_store: EventStore | None = None,
        risk_info_provider: Callable[[str], dict[str, Any]] | None = None,
        position_size_estimator: (
            Callable[[PositionSizingContext], PositionSizingAdvice] | None
        ) = None,
        impact_estimator: Callable[[ImpactRequest], ImpactAssessment] | None = None,
    ):
        """
        Initialize risk manager with configuration.

        Args:
            config: Risk configuration (defaults loaded from env)
            event_store: Event store for risk metrics/events
            risk_info_provider: Provider for symbol-specific risk info
            position_size_estimator: Optional dynamic position sizing calculator
            impact_estimator: Optional callable returning market impact assessments
        """
        self.config = config or RiskConfig.from_env()
        self.event_store = event_store or EventStore()

        # Time provider for testability
        self._now_provider = lambda: datetime.utcnow()

        # Shared state
        self.last_mark_update: dict[str, datetime] = {}

        # Initialize helper modules
        self.state_manager = StateManager(
            config=self.config,
            event_store=self.event_store,
            now_provider=self._now_provider,
        )

        self.position_sizer = PositionSizer(
            config=self.config,
            event_store=self.event_store,
            position_size_estimator=position_size_estimator,
            impact_estimator=impact_estimator,
            is_reduce_only_mode=self.state_manager.is_reduce_only_mode,
        )

        self.pre_trade_validator = PreTradeValidator(
            config=self.config,
            event_store=self.event_store,
            risk_info_provider=risk_info_provider,
            impact_estimator=impact_estimator,
            is_reduce_only_mode=self.state_manager.is_reduce_only_mode,
            now_provider=self._now_provider,
            last_mark_update=self.last_mark_update,
        )

        self.runtime_monitor = RuntimeMonitor(
            config=self.config,
            event_store=self.event_store,
            set_reduce_only_mode=self.state_manager.set_reduce_only_mode,
            now_provider=self._now_provider,
            last_mark_update=self.last_mark_update,
        )

        # Expose commonly accessed state attributes for backward compatibility
        self._state = self.state_manager._state
        self.daily_pnl = self.state_manager.daily_pnl
        self.start_of_day_equity = self.state_manager.start_of_day_equity
        self.positions = self.runtime_monitor.positions
        self.circuit_breaker_state = self.runtime_monitor.circuit_breaker_state

        # Private attributes
        self._risk_info_provider = risk_info_provider
        self._position_size_estimator = position_size_estimator
        self._impact_estimator = impact_estimator
        self._state_listener: Callable[[RiskRuntimeState], None] | None = None

        # Log configuration
        if hasattr(self.config, "to_dict"):
            logger.info(f"LiveRiskManager initialized with config: {self.config.to_dict()}")
        else:
            logger.info("LiveRiskManager initialized with a non-standard config object")

    # ========== State Management Delegation ==========

    def is_reduce_only_mode(self) -> bool:
        """Check if reduce-only mode is active."""
        return self.state_manager.is_reduce_only_mode()

    def set_reduce_only_mode(self, enabled: bool, reason: str = "") -> None:
        """Toggle reduce-only mode."""
        self.state_manager.set_reduce_only_mode(enabled, reason)
        # Update local reference for backward compatibility
        self._state = self.state_manager._state

    def set_state_listener(self, listener: Callable[[RiskRuntimeState], None] | None) -> None:
        """Register a listener for state changes."""
        self.state_manager.set_state_listener(listener)
        self._state_listener = listener

    def reset_daily_tracking(self, current_equity: Decimal) -> None:
        """Reset daily tracking at start of new day."""
        self.state_manager.reset_daily_tracking(current_equity)
        # Update local references for backward compatibility
        self.daily_pnl = self.state_manager.daily_pnl
        self.start_of_day_equity = self.state_manager.start_of_day_equity

    # ========== Position Sizing Delegation ==========

    def size_position(self, context: PositionSizingContext) -> PositionSizingAdvice:
        """Calculate position size using dynamic estimator or fallback logic."""
        return self.position_sizer.size_position(context)

    def set_impact_estimator(
        self, estimator: Callable[[ImpactRequest], ImpactAssessment] | None
    ) -> None:
        """Install or clear the market-impact estimator hook."""
        self.position_sizer.set_impact_estimator(estimator)
        self.pre_trade_validator._impact_estimator = estimator
        self._impact_estimator = estimator

    # ========== Pre-Trade Validation Delegation ==========

    def pre_trade_validate(
        self,
        symbol: str,
        side: str,
        qty: Decimal | None = None,
        price: Decimal | None = None,
        product: Product | None = None,
        equity: Decimal | None = None,
        current_positions: dict[str, Any] | None = None,
        *,
        quantity: Decimal | None = None,
    ) -> None:
        """Validate order against all risk limits before placement."""
        self.pre_trade_validator.pre_trade_validate(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            product=product,
            equity=equity,
            current_positions=current_positions,
            quantity=quantity,
        )

    def validate_leverage(
        self,
        symbol: str,
        qty: Decimal | None = None,
        price: Decimal | None = None,
        product: Product | None = None,
        equity: Decimal | None = None,
        *,
        quantity: Decimal | None = None,
    ) -> None:
        """Validate that order doesn't exceed leverage limits."""
        self.pre_trade_validator.validate_leverage(
            symbol=symbol,
            qty=qty,
            price=price,
            product=product,
            equity=equity,
            quantity=quantity,
        )

    def validate_liquidation_buffer(
        self,
        symbol: str,
        qty: Decimal | None = None,
        price: Decimal | None = None,
        product: Product | None = None,
        equity: Decimal | None = None,
        *,
        quantity: Decimal | None = None,
    ) -> None:
        """Ensure adequate buffer from liquidation after trade."""
        self.pre_trade_validator.validate_liquidation_buffer(
            symbol=symbol,
            qty=qty,
            price=price,
            product=product,
            equity=equity,
            quantity=quantity,
        )

    def validate_exposure_limits(
        self,
        symbol: str,
        notional: Decimal,
        equity: Decimal,
        current_positions: dict[str, Any] | None = None,
    ) -> None:
        """Validate per-symbol and total exposure limits."""
        self.pre_trade_validator.validate_exposure_limits(
            symbol=symbol,
            notional=notional,
            equity=equity,
            current_positions=current_positions,
        )

    def validate_slippage_guard(
        self,
        symbol: str,
        side: str,
        qty: Decimal | None = None,
        expected_price: Decimal | None = None,
        mark_or_quote: Decimal | None = None,
        *,
        quantity: Decimal | None = None,
    ) -> None:
        """Optional slippage guard based on spread."""
        self.pre_trade_validator.validate_slippage_guard(
            symbol=symbol,
            side=side,
            qty=qty,
            expected_price=expected_price,
            mark_or_quote=mark_or_quote,
            quantity=quantity,
        )

    def set_risk_info_provider(self, provider: Callable[[str], dict[str, Any]]) -> None:
        """Set a provider callable that returns exchange risk info for a symbol."""
        self._risk_info_provider = provider
        self.pre_trade_validator._risk_info_provider = provider

    # ========== Runtime Monitoring Delegation ==========

    def track_daily_pnl(
        self, current_equity: Decimal, positions_pnl: dict[str, dict[str, Decimal]]
    ) -> bool:
        """Track daily PnL and trigger reduce-only if breaching limit."""
        # Initialize start of day if needed
        if self.state_manager.start_of_day_equity == 0:
            self.state_manager.start_of_day_equity = current_equity
            self.start_of_day_equity = current_equity
            return False

        triggered, updated_pnl = self.runtime_monitor.track_daily_pnl(
            current_equity=current_equity,
            positions_pnl=positions_pnl,
            daily_pnl=self.state_manager.daily_pnl,
            start_of_day_equity=self.state_manager.start_of_day_equity,
        )

        # Update state manager's daily_pnl
        self.state_manager.daily_pnl = updated_pnl
        self.daily_pnl = updated_pnl

        return triggered

    def check_liquidation_buffer(
        self, symbol: str, position_data: dict[str, Any], equity: Decimal
    ) -> bool:
        """Monitor liquidation buffer for position."""
        result = self.runtime_monitor.check_liquidation_buffer(symbol, position_data, equity)
        # Update local reference for backward compatibility
        self.positions = self.runtime_monitor.positions
        return result

    def check_mark_staleness(self, symbol: str, mark_timestamp: datetime | None = None) -> bool:
        """Check whether mark data for symbol is stale."""
        return self.runtime_monitor.check_mark_staleness(symbol, mark_timestamp)

    def append_risk_metrics(self, equity: Decimal, positions: dict[str, Any]) -> None:
        """Append periodic risk metrics snapshot to the event store."""
        self.runtime_monitor.append_risk_metrics(
            equity=equity,
            positions=positions,
            daily_pnl=self.state_manager.daily_pnl,
            start_of_day_equity=self.state_manager.start_of_day_equity,
            is_reduce_only_mode=self.state_manager.is_reduce_only_mode(),
        )

    def check_correlation_risk(self, positions: dict[str, Any]) -> bool:
        """Check portfolio correlation and concentration risk."""
        return self.runtime_monitor.check_correlation_risk(positions)

    def check_volatility_circuit_breaker(
        self, symbol: str, recent_marks: list[Decimal]
    ) -> CircuitBreakerOutcome:
        """Check rolling volatility and trigger progressive circuit breakers."""
        outcome = self.runtime_monitor.check_volatility_circuit_breaker(symbol, recent_marks)
        # Update local reference for backward compatibility
        self.circuit_breaker_state = self.runtime_monitor.circuit_breaker_state
        return outcome

    # ========== Internal Helpers ==========

    def _now(self) -> datetime:
        """Get current time (for testability)."""
        return self._now_provider()
