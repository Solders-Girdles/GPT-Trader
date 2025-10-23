"""
Risk management for perpetuals live trading.

Phase 5: Leverage-aware sizing and runtime guards.
Complete isolation - no strategy logic, pure risk controls.

This module now serves as a delegation facade to focused submodules.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, MutableMapping
from datetime import datetime
from decimal import Decimal
from threading import RLock
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
from bot_v2.features.live_trade.risk.state_management import (
    RiskRuntimeState,
    StateManager,
)
from bot_v2.features.live_trade.risk_runtime import CircuitBreakerOutcome, RuntimeMonitor
from bot_v2.orchestration.state_manager import ReduceOnlyModeSource
from bot_v2.persistence.event_store import EventStore
from bot_v2.utilities.datetime_helpers import normalize_to_utc, utc_now
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="live_trade_risk")

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


class MarkTimestampRegistry(MutableMapping[str, datetime | None]):
    """Thread-safe registry for tracking latest mark timestamps per symbol."""

    def __init__(self, now_provider: Callable[[], datetime] | None = None) -> None:
        self._data: dict[str, datetime | None] = {}
        self._lock = RLock()
        self._now_provider = now_provider or utc_now

    def __getitem__(self, key: str) -> datetime | None:
        with self._lock:
            return self._data[key]

    def __setitem__(self, key: str, value: datetime | None) -> None:
        with self._lock:
            self._data[key] = value

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._data

    def keys(self) -> tuple[str, ...]:  # type: ignore[override]
        with self._lock:
            return tuple(self._data.keys())

    def items(self) -> tuple[tuple[str, datetime | None], ...]:  # type: ignore[override]
        with self._lock:
            return tuple(self._data.items())

    def values(self) -> tuple[datetime | None, ...]:  # type: ignore[override]
        with self._lock:
            return tuple(self._data.values())

    def get(self, key: str, default: datetime | None = None) -> datetime | None:  # type: ignore[override]
        with self._lock:
            return self._data.get(key, default)

    def clear(self) -> None:  # type: ignore[override]
        with self._lock:
            self._data.clear()

    def snapshot(self) -> dict[str, datetime | None]:
        """Return a shallow copy of the registry."""
        with self._lock:
            return dict(self._data)

    def update_timestamp(self, symbol: str, timestamp: datetime | None = None) -> datetime:
        """Record the latest mark timestamp for a symbol."""
        ts_source = timestamp or self._now_provider()
        normalized = normalize_to_utc(ts_source)
        with self._lock:
            self._data[symbol] = normalized
        return normalized


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
        self.last_mark_update: MarkTimestampRegistry = MarkTimestampRegistry(
            now_provider=self._now_provider
        )

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

        # Store reference to centralized state manager for injection into runtime monitor
        self._centralized_state_manager = None

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

    def record_mark_update(self, symbol: str, timestamp: datetime | None = None) -> datetime:
        """Update the latest mark timestamp for ``symbol``."""
        return self.last_mark_update.update_timestamp(symbol, timestamp)

    def mark_timestamp_for(self, symbol: str) -> datetime | None:
        """Return the last recorded mark timestamp for ``symbol``."""
        return self.last_mark_update.get(symbol)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name == "_now_provider":
            state_manager = getattr(self, "state_manager", None)
            if state_manager is not None:
                state_manager._now_provider = value
            pre_trade_validator = getattr(self, "pre_trade_validator", None)
            if pre_trade_validator is not None:
                pre_trade_validator._now_provider = value
            runtime_monitor = getattr(self, "runtime_monitor", None)
            if runtime_monitor is not None:
                runtime_monitor._now_provider = value

    def set_event_store(self, event_store: EventStore) -> None:
        """Rebind all subcomponents to a shared event store."""
        self.event_store = event_store
        if hasattr(self.state_manager, "event_store"):
            self.state_manager.event_store = event_store
        if hasattr(self.position_sizer, "event_store"):
            self.position_sizer.event_store = event_store
        if hasattr(self.pre_trade_validator, "event_store"):
            self.pre_trade_validator.event_store = event_store
        if hasattr(self.runtime_monitor, "event_store"):
            self.runtime_monitor.event_store = event_store

    # ========== State Management Delegation ==========

    def is_reduce_only_mode(self) -> bool:
        """Check if reduce-only mode is active."""
        return self.state_manager.is_reduce_only_mode()

    def set_reduce_only_mode(self, enabled: bool, reason: str = "") -> None:
        """Toggle reduce-only mode."""
        # Check if there's a centralized state manager available

        # Try to find the bot instance that contains this risk manager
        # This is a temporary solution - ideally the state manager should be injected
        centralized_manager = None

        # First check if we have a reference to the centralized manager
        if hasattr(self, "_centralized_state_manager"):
            centralized_manager = self._centralized_state_manager
        else:
            # Try to find it through the event store (not ideal but works for backward compatibility)
            # This will be improved when we properly inject the dependency
            pass

        if centralized_manager is not None:
            centralized_manager.set_reduce_only_mode(
                enabled=enabled,
                reason=reason,
                source=ReduceOnlyModeSource.RISK_MANAGER,
                metadata={"context": "risk_manager"},
            )
        else:
            # Fallback to local state manager
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
