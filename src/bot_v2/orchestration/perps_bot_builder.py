"""Builder pattern for PerpsBot to improve testability and maintainability."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from bot_v2.monitoring.configuration_guardian import ConfigurationGuardian
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.perps_bootstrap import PerpsBootstrapResult, prepare_perps_bot
from bot_v2.orchestration.service_registry import ServiceRegistry, empty_registry

logger = logging.getLogger(__name__)


class PerpsBotBuilder:
    """Builder for PerpsBot instances with improved separation of concerns.
    
    This builder breaks down the complex PerpsBot initialization into focused steps:
    1. Configuration setup
    2. Runtime state initialization  
    3. Storage bootstrapping
    4. Service construction
    5. Monitoring setup
    6. Final initialization
    """
    
    def __init__(self) -> None:
        self._config: BotConfig | None = None
        self._registry: ServiceRegistry | None = None
        self._bootstrap_result: PerpsBootstrapResult | None = None
        self._storage_context: Any | None = None
        self._baseline_snapshot: Any | None = None
        
    def with_config(self, config: BotConfig) -> PerpsBotBuilder:
        """Set the bot configuration."""
        self._config = config
        return self
        
    def with_registry(self, registry: ServiceRegistry) -> PerpsBotBuilder:
        """Set the service registry (for dependency injection in tests)."""
        self._registry = registry
        return self
        
    def prepare_runtime_environment(self) -> PerpsBotBuilder:
        """Prepare runtime paths and bootstrap dependencies."""
        if self._config is None:
            raise ValueError("Configuration must be set before preparing runtime environment")
            
        self._bootstrap_result = prepare_perps_bot(self._config)
        return self
        
    def bootstrap_storage(self) -> PerpsBotBuilder:
        """Bootstrap storage components."""
        if self._bootstrap_result is None:
            raise ValueError("Runtime environment must be prepared before bootstrapping storage")
            
        self._storage_context = self._bootstrap_result.storage_context
        return self
        
    def create_baseline_snapshot(self) -> PerpsBotBuilder:
        """Create baseline snapshot for configuration drift detection."""
        if self._config is None:
            raise ValueError("Configuration must be set before creating baseline snapshot")
            
        config_dict = self._extract_config_dict()
        active_symbols = list(self._config.symbols) if self._config.symbols else []
        positions = []  # Empty at startup
        account_equity = None  # Not known yet
        broker_type = "mock" if self._config.mock_broker else "live"
        
        self._baseline_snapshot = ConfigurationGuardian.create_baseline_snapshot(
            config_dict=config_dict,
            active_symbols=active_symbols,
            positions=positions,
            account_equity=account_equity,
            profile=self._config.profile,
            broker_type=broker_type,
        )
        return self
        
    def build(self) -> "PerpsBot":
        """Build the PerpsBot instance."""
        if self._config is None:
            raise ValueError("Configuration must be set before building")
        if self._bootstrap_result is None:
            raise ValueError("Runtime environment must be prepared before building")
        if self._storage_context is None:
            raise ValueError("Storage must be bootstrapped before building")
        if self._baseline_snapshot is None:
            raise ValueError("Baseline snapshot must be created before building")
            
        # Import here to avoid circular imports
        from bot_v2.orchestration.perps_bot import PerpsBot
        
        # Create bot with pre-built dependencies
        bot = PerpsBot.__new__(PerpsBot)
        
        # Set basic attributes
        bot.bot_id = "perps_bot"
        bot.start_time = datetime.now(UTC)
        bot.running = False
        
        # Set configuration and registry
        bot.config = self._config
        bot.registry = self._registry or self._bootstrap_result.registry
        bot.config_controller = self._bootstrap_result.config_controller
        bot._session_guard = self._bootstrap_result.session_guard
        
        # Set symbols and derivatives flag
        bot.symbols = list(self._config.symbols or [])
        if not bot.symbols:
            logger.warning("No symbols configured; continuing with empty symbol list")
        bot._derivatives_enabled = bool(getattr(self._config, "derivatives_enabled", False))
        
        # Set storage components
        bot.event_store = self._storage_context.event_store
        bot.orders_store = self._storage_context.orders_store
        
        # Set baseline snapshot and guardian
        bot.baseline_snapshot = self._baseline_snapshot
        bot.configuration_guardian = ConfigurationGuardian(self._baseline_snapshot)
        
        # Initialize runtime state
        bot._init_runtime_state()
        
        # Construct services
        bot._construct_services()
        
        # Bootstrap runtime coordinator
        bot.runtime_coordinator.bootstrap()
        
        # Initialize remaining services
        bot._init_accounting_services()
        bot._init_market_services()
        bot._start_streaming_if_configured()
        
        return bot
        
    def _extract_config_dict(self) -> dict[str, Any]:
        """Extract configuration dictionary for baseline snapshot."""
        if self._config is None:
            raise ValueError("Configuration must be set")
            
        return {
            "profile": self._config.profile,
            "dry_run": self._config.dry_run,
            "symbols": list(self._config.symbols) if self._config.symbols else [],
            "derivatives_enabled": getattr(self._config, "derivatives_enabled", False),
            "update_interval": self._config.update_interval,
            "short_ma": self._config.short_ma,
            "long_ma": self._config.long_ma,
            "target_leverage": self._config.target_leverage,
            "trailing_stop_pct": self._config.trailing_stop_pct,
            "enable_shorts": self._config.enable_shorts,
            "max_position_size": self._config.max_position_size,
            "max_leverage": self._config.max_leverage,
            "reduce_only_mode": self._config.reduce_only_mode,
            "mock_broker": self._config.mock_broker,
            "mock_fills": self._config.mock_fills,
            "enable_order_preview": self._config.enable_order_preview,
            "account_telemetry_interval": self._config.account_telemetry_interval,
            "trading_window_start": self._config.trading_window_start,
            "trading_window_end": self._config.trading_window_end,
            "trading_days": self._config.trading_days,
            "daily_loss_limit": self._config.daily_loss_limit,
            "time_in_force": self._config.time_in_force,
            "perps_enable_streaming": getattr(self._config, "perps_enable_streaming", False),
            "perps_stream_level": getattr(self._config, "perps_stream_level", 1),
            "perps_paper_trading": getattr(self._config, "perps_paper_trading", False),
            "perps_force_mock": getattr(self._config, "perps_force_mock", False),
            "perps_position_fraction": getattr(self._config, "perps_position_fraction", 1.0),
            "perps_skip_startup_reconcile": getattr(self._config, "perps_skip_startup_reconcile", False),
        }


def create_perps_bot(
    config: BotConfig,
    registry: ServiceRegistry | None = None,
) -> "PerpsBot":
    """Factory function to create a PerpsBot using the builder pattern.
    
    Args:
        config: Bot configuration
        registry: Optional service registry for dependency injection
        
    Returns:
        Configured PerpsBot instance
    """
    builder = PerpsBotBuilder()
    
    if registry:
        builder.with_registry(registry)
        
    return (
        builder
        .with_config(config)
        .prepare_runtime_environment()
        .bootstrap_storage()
        .create_baseline_snapshot()
        .build()
    )


# Factory for test environments
def create_test_perps_bot(
    config: BotConfig,
    registry: ServiceRegistry | None = None,
    mock_storage: bool = True,
    mock_baseline: bool = True,
) -> "PerpsBot":
    """Factory function to create a PerpsBot for testing with mocks.
    
    Args:
        config: Bot configuration
        registry: Optional service registry for dependency injection
        mock_storage: Whether to use mock storage components
        mock_baseline: Whether to create a mock baseline snapshot
        
    Returns:
        Configured PerpsBot instance for testing
    """
    builder = PerpsBotBuilder()
    
    if registry:
        builder.with_registry(registry)
        
    builder = builder.with_config(config).prepare_runtime_environment()
    
    if not mock_storage:
        builder = builder.bootstrap_storage()
        
    if not mock_baseline:
        builder = builder.create_baseline_snapshot()
        
    return builder.build()
