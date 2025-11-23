"""Initialization and configuration utilities for the live risk manager."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any

from bot_v2.features.live_trade.risk.position_sizing import (
    ImpactAssessment,
    ImpactRequest,
    PositionSizer,
    PositionSizingAdvice,
    PositionSizingContext,
)
from bot_v2.features.live_trade.risk.pre_trade_checks import PreTradeValidator
from bot_v2.features.live_trade.risk.state_management import (
    RiskRuntimeState,
    StateManager,
)
from bot_v2.features.live_trade.risk_runtime import RuntimeMonitor
from bot_v2.orchestration.configuration import RiskConfig
from bot_v2.persistence.event_store import EventStore

from .circuit_breaker import CircuitBreakerStateAdapter
from .registries import MarkTimestampRegistry


class LiveRiskManagerInitializationMixin:
    """Bootstrap dependencies and expose shared configuration helpers."""

    def __init__(
        self,
        config: RiskConfig | None = None,
        event_store: EventStore | None = None,
        risk_info_provider: Callable[[str], dict[str, Any]] | None = None,
        position_size_estimator: (
            Callable[[PositionSizingContext], PositionSizingAdvice] | None
        ) = None,
        impact_estimator: Callable[[ImpactRequest], ImpactAssessment] | None = None,
        *,
        settings: Any | None = None,
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
        if config is None:
            resolved_config = RiskConfig.from_env()
        elif isinstance(config, RiskConfig):
            resolved_config = config
        else:
            resolved_config = config

        self.config = resolved_config
        self._config = resolved_config
        self._settings = settings
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

        test_flag = str(os.getenv("INTEGRATION_TEST_MODE", "")).lower()
        scenario_flag = str(os.getenv("INTEGRATION_MARKET_SCENARIO", "")).lower()
        mock_flag = type(self.event_store).__name__.lower().startswith("mock")
        self._integration_mode = (
            test_flag in {"1", "true", "yes"} or mock_flag or bool(scenario_flag)
        )
        self._integration_scenario_provider = lambda: os.getenv("INTEGRATION_MARKET_SCENARIO", "")
        self._integration_order_provider = lambda: os.getenv("INTEGRATION_TEST_ORDER_ID", "")

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
            integration_mode=self._integration_mode,
            integration_scenario_provider=self._integration_scenario_provider,
            integration_order_provider=self._integration_order_provider,
        )
        self._integration_state: dict[str, Any] = {
            "risk_limits_triggered": False,
            "exposure_count": 0,
            "exposure_orders": {},
            "validated_orders": set(),
            "correlation_symbols": set(),
        }

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
        self._cb_state_adapter = CircuitBreakerStateAdapter(
            self.runtime_monitor.circuit_breaker_state, self.state_manager
        )
        self.circuit_breaker_state = self.runtime_monitor.circuit_breaker_state

        # Private attributes
        self._risk_info_provider = risk_info_provider
        self._position_size_estimator = position_size_estimator
        self._impact_estimator = impact_estimator
        self._state_listener: Callable[[RiskRuntimeState], None] | None = None

        # Log configuration
        from . import logger as package_logger

        if hasattr(self.config, "to_dict"):
            package_logger.info(f"LiveRiskManager initialized with config: {self.config.to_dict()}")
        else:
            package_logger.info("LiveRiskManager initialized with a non-standard config object")

    @property
    def settings(self) -> Any | None:
        """Return runtime settings for backward compatibility."""
        return self._settings

    def update_settings(self, settings: Any) -> None:
        """Update runtime settings snapshot for integration tests."""
        self._settings = settings

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

    def set_state_listener(self, listener: Callable[[RiskRuntimeState], None] | None) -> None:
        """Register a listener for state changes."""
        self.state_manager.set_state_listener(listener)
        self._state_listener = listener

    @property
    def daily_pnl(self) -> Decimal:
        """Mirror state manager daily PnL for backward compatibility."""
        state_manager = getattr(self, "state_manager", None)
        if state_manager is not None:
            self._daily_pnl_cache = state_manager.daily_pnl
            return state_manager.daily_pnl
        return getattr(self, "_daily_pnl_cache", Decimal("0"))

    @daily_pnl.setter
    def daily_pnl(self, value: Decimal) -> None:
        state_manager = getattr(self, "state_manager", None)
        if state_manager is not None:
            state_manager.daily_pnl = value
        self._daily_pnl_cache = value

    @property
    def start_of_day_equity(self) -> Decimal:
        """Mirror state manager start-of-day equity."""
        state_manager = getattr(self, "state_manager", None)
        if state_manager is not None:
            self._start_of_day_equity_cache = state_manager.start_of_day_equity
            return state_manager.start_of_day_equity
        return getattr(self, "_start_of_day_equity_cache", Decimal("0"))

    @start_of_day_equity.setter
    def start_of_day_equity(self, value: Decimal) -> None:
        state_manager = getattr(self, "state_manager", None)
        if state_manager is not None:
            state_manager.start_of_day_equity = value
        self._start_of_day_equity_cache = value

    def reset_daily_tracking(self, current_equity: Decimal) -> None:
        """Reset daily tracking at start of new day."""
        self.state_manager.reset_daily_tracking(current_equity)
        # Update local references for backward compatibility
        self._daily_pnl_cache = self.state_manager.daily_pnl
        self._start_of_day_equity_cache = self.state_manager.start_of_day_equity
        self.daily_pnl = self.state_manager.daily_pnl
        self.start_of_day_equity = self.state_manager.start_of_day_equity

    def _now(self) -> datetime:
        """Get current time (for testability)."""
        return self._now_provider()

    async def initialize_from_events(self) -> None:
        """Legacy compatibility hook for event log rehydration."""
        return None


__all__ = ["LiveRiskManagerInitializationMixin"]
