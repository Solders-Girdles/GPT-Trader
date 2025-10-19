"""
Comprehensive unit tests for StrategyCoordinator.

Tests strategy cycle orchestration, mark updates, trading cycle execution,
configuration drift handling, and symbol processing flows.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest
from tests.unit.bot_v2.orchestration.helpers import ScenarioBuilder

from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.coordinators.strategy import StrategyCoordinator
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.service_registry import ServiceRegistry


@pytest.fixture
def base_context():
    """Base coordinator context for strategy coordinator tests."""
    config = BotConfig(profile=Profile.PROD, symbols=["BTC-PERP"], dry_run=False)
    runtime_state = PerpsBotRuntimeState(["BTC-PERP"])

    broker = Mock()
    risk_manager = Mock()
    orders_store = Mock()
    event_store = Mock()

    registry = ServiceRegistry(config=config)

    context = CoordinatorContext(
        config=config,
        registry=registry,
        event_store=event_store,
        orders_store=orders_store,
        broker=broker,
        risk_manager=risk_manager,
        symbols=("BTC-PERP",),
        bot_id="test-bot",
        runtime_state=runtime_state,
        config_controller=Mock(),
        strategy_orchestrator=Mock(),
        set_running_flag=lambda _: None,
    )
    return context


@pytest.fixture
def coordinator(base_context):
    """StrategyCoordinator instance."""
    return StrategyCoordinator(base_context)


class TestStrategyCoordinatorInitialization:
    """Test StrategyCoordinator initialization."""

    def test_initialization_sets_context(self, coordinator, base_context):
        """Test coordinator initializes with context."""
        assert coordinator.context == base_context
        assert coordinator.name == "strategy"

    def test_initialize_returns_updated_context(self, coordinator, base_context):
        """Test initialize returns context (no-op for strategy coordinator)."""
        result = coordinator.initialize(base_context)
        assert result == base_context


class TestSymbolProcessorHelpers:
    """Test symbol processor helper methods."""

    def test_symbol_processor_returns_orchestrator(self, coordinator, base_context):
        """Test symbol_processor property returns strategy orchestrator."""
        base_context.strategy_orchestrator = Mock()
        coordinator.update_context(base_context)

        processor = coordinator.symbol_processor
        assert processor == base_context.strategy_orchestrator

    def test_set_symbol_processor_updates_internal(self, coordinator):
        """Test set_symbol_processor updates internal processor."""
        new_processor = Mock()
        coordinator.set_symbol_processor(new_processor)
        assert coordinator._symbol_processor == new_processor

    def test_process_symbol_expects_context_detection(self, coordinator):
        """Test _process_symbol_expects_context detects signature requirements."""
        # Mock processor with context parameters
        processor = Mock()
        processor.process_symbol = Mock()
        coordinator.set_symbol_processor(processor)

        # Should detect no context needed for simple signature
        expects_context = coordinator._process_symbol_expects_context()
        assert expects_context is False

        # Mock processor requiring context
        def process_with_context(symbol, balances=None, position_map=None):
            pass

        processor.process_symbol = process_with_context
        coordinator.set_symbol_processor(processor)

        expects_context = coordinator._process_symbol_expects_context()
        assert expects_context is True


class TestTradingCycle:
    """Test run_cycle and trading cycle orchestration."""

    @pytest.mark.asyncio
    async def test_run_cycle_executes_full_flow(self, coordinator, base_context):
        """Test run_cycle executes complete trading cycle."""
        # Setup mocks
        base_context.broker.list_balances = AsyncMock(return_value=[])
        base_context.broker.list_positions = AsyncMock(return_value=[])
        base_context.broker.get_account_info = AsyncMock(return_value=None)
        base_context.system_monitor = Mock()
        base_context.system_monitor.log_status = AsyncMock()
        base_context.session_guard = Mock()
        base_context.session_guard.should_trade = Mock(return_value=True)

        coordinator.update_context(base_context)

        # Mock update_marks to avoid quote fetching
        coordinator.update_marks = AsyncMock()

        # Mock configuration validation
        coordinator._validate_configuration_and_handle_drift = AsyncMock(return_value=True)
        coordinator._execute_trading_cycle = AsyncMock()

        await coordinator.run_cycle()

        coordinator.update_marks.assert_called_once()
        coordinator._execute_trading_cycle.assert_called_once()
        base_context.system_monitor.log_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_cycle_skips_when_session_guard_blocks(self, coordinator, base_context):
        """Test run_cycle skips trading when session guard blocks."""
        base_context.session_guard = Mock()
        base_context.session_guard.should_trade = Mock(return_value=False)
        base_context.system_monitor = Mock()
        base_context.system_monitor.log_status = AsyncMock()

        coordinator.update_context(base_context)

        await coordinator.run_cycle()

        base_context.system_monitor.log_status.assert_called_once()
        # Should not execute trading cycle

    @pytest.mark.asyncio
    async def test_run_cycle_handles_config_drift(self, coordinator, base_context):
        """Test run_cycle handles configuration drift detection."""
        base_context.broker.list_balances = AsyncMock(return_value=[])
        base_context.broker.list_positions = AsyncMock(return_value=[])
        base_context.broker.get_account_info = AsyncMock(return_value=None)

        coordinator.update_context(base_context)
        coordinator.update_marks = AsyncMock()
        coordinator._validate_configuration_and_handle_drift = AsyncMock(return_value=False)

        await coordinator.run_cycle()

        # Should return early without executing trading cycle
        coordinator._validate_configuration_and_handle_drift.assert_called_once()


class TestMarkUpdates:
    """Test mark update functionality."""

    @pytest.mark.asyncio
    async def test_update_marks_fetches_quotes(self, coordinator, base_context):
        """Test update_marks fetches quotes for all symbols."""
        base_context.symbols = ("BTC-PERP", "ETH-PERP")
        coordinator.update_context(base_context)

        # Mock quote responses
        btc_quote = Mock(last=Decimal("50000"), ts=None)
        eth_quote = Mock(last=Decimal("3000"), ts=None)

        base_context.broker.get_quote = AsyncMock(side_effect=[btc_quote, eth_quote])

        await coordinator.update_marks()

        assert base_context.broker.get_quote.call_count == 2
        calls = base_context.broker.get_quote.call_args_list
        assert calls[0][0][0] == "BTC-PERP"
        assert calls[1][0][0] == "ETH-PERP"

    def test_process_quote_update_updates_mark_window(self, coordinator, base_context):
        """Test _process_quote_update updates mark window correctly."""
        symbol = "BTC-PERP"
        quote = Mock(last=Decimal("50000"), ts=None)

        coordinator._process_quote_update(symbol, quote)

        runtime_state = base_context.runtime_state
        assert symbol in runtime_state.mark_windows
        assert runtime_state.mark_windows[symbol][-1] == Decimal("50000")

    def test_update_mark_window_respects_max_size(self, coordinator, base_context):
        """Test update_mark_window respects maximum window size."""
        symbol = "BTC-PERP"
        runtime_state = base_context.runtime_state

        # Add marks up to limit
        max_size = 35  # Based on config short_ma + long_ma + buffer
        for i in range(max_size + 5):
            coordinator.update_mark_window(symbol, Decimal(str(50000 + i)))

        window = runtime_state.mark_windows[symbol]
        assert len(window) <= max_size


class TestTradingCycleExecution:
    """Test _execute_trading_cycle method."""

    @pytest.mark.asyncio
    async def test_execute_trading_cycle_processes_all_symbols(self, coordinator, base_context):
        """Test _execute_trading_cycle processes all symbols."""
        base_context.symbols = ("BTC-PERP", "ETH-PERP")
        coordinator.update_context(base_context)

        # Mock processor that expects context
        processor = Mock()
        processor.process_symbol = AsyncMock()
        coordinator.set_symbol_processor(processor)

        trading_state = {
            "balances": [],
            "positions": [],
            "position_map": {},
            "account_equity": None,
        }

        await coordinator._execute_trading_cycle(trading_state)

        assert processor.process_symbol.call_count == 2
        calls = processor.process_symbol.call_args_list
        assert calls[0][0][0] == "BTC-PERP"
        assert calls[1][0][0] == "ETH-PERP"

    @pytest.mark.asyncio
    async def test_execute_trading_cycle_passes_context_when_needed(
        self, coordinator, base_context
    ):
        """Test _execute_trading_cycle passes context parameters when required."""
        base_context.symbols = ("BTC-PERP",)
        coordinator.update_context(base_context)

        # Mock processor requiring context
        processor = Mock()
        processor.process_symbol = AsyncMock()
        coordinator.set_symbol_processor(processor)

        # Force context detection to require context
        coordinator._process_symbol_expects_context = Mock(return_value=True)

        trading_state = {
            "balances": [ScenarioBuilder.create_balance()],
            "positions": [],
            "position_map": {"BTC-PERP": ScenarioBuilder.create_position()},
            "account_equity": Decimal("10000"),
        }

        await coordinator._execute_trading_cycle(trading_state)

        processor.process_symbol.assert_called_once()
        call_args = processor.process_symbol.call_args
        assert call_args[0][0] == "BTC-PERP"
        assert call_args[0][1] == trading_state["balances"]
        assert call_args[0][2] == trading_state["position_map"]


class TestConfigurationDriftHandling:
    """Test configuration drift detection and handling."""

    @pytest.mark.asyncio
    async def test_validate_config_handles_valid_config(self, coordinator, base_context):
        """Test _validate_configuration_and_handle_drift returns True for valid config."""
        guardian = Mock()
        validation_result = Mock()
        validation_result.is_valid = True
        guardian.pre_cycle_check = Mock(return_value=validation_result)

        base_context.configuration_guardian = guardian
        coordinator.update_context(base_context)

        trading_state = {
            "balances": [],
            "positions": [],
            "account_equity": None,
        }

        result = await coordinator._validate_configuration_and_handle_drift(trading_state)

        assert result is True
        guardian.pre_cycle_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_config_handles_critical_drift(self, coordinator, base_context):
        """Test _validate_configuration_and_handle_drift handles critical drift."""
        guardian = Mock()
        validation_result = Mock()
        validation_result.is_valid = False
        validation_result.errors = ["emergency_shutdown_required", "critical_error"]
        guardian.pre_cycle_check = Mock(return_value=validation_result)

        base_context.configuration_guardian = guardian
        base_context.set_running_flag = Mock()
        base_context.shutdown_hook = AsyncMock()

        coordinator.update_context(base_context)

        trading_state = {
            "balances": [],
            "positions": [],
            "account_equity": None,
        }

        result = await coordinator._validate_configuration_and_handle_drift(trading_state)

        assert result is False
        base_context.set_running_flag.assert_called_once_with(False)
        base_context.shutdown_hook.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_config_enables_reduce_only_for_high_severity(
        self, coordinator, base_context
    ):
        """Test _validate_configuration_and_handle_drift enables reduce-only for high severity."""
        guardian = Mock()
        validation_result = Mock()
        validation_result.is_valid = False
        validation_result.errors = ["high_severity_violation"]
        guardian.pre_cycle_check = Mock(return_value=validation_result)

        base_context.configuration_guardian = guardian
        base_context.set_reduce_only_mode = Mock()

        coordinator.update_context(base_context)

        trading_state = {
            "balances": [],
            "positions": [],
            "account_equity": None,
        }

        result = await coordinator._validate_configuration_and_handle_drift(trading_state)

        assert result is False
        base_context.set_reduce_only_mode.assert_called_once_with(
            True, "Configuration drift detected"
        )


class TestProcessSymbol:
    """Test process_symbol method."""

    @pytest.mark.asyncio
    async def test_process_symbol_delegates_to_processor(self, coordinator, base_context):
        """Test process_symbol delegates to symbol processor."""
        processor = Mock()
        processor.process_symbol = AsyncMock(return_value="processed")
        coordinator.set_symbol_processor(processor)

        result = await coordinator.process_symbol("BTC-PERP")

        processor.process_symbol.assert_called_once_with("BTC-PERP")
        assert result == "processed"

    @pytest.mark.asyncio
    async def test_process_symbol_passes_context_when_required(self, coordinator, base_context):
        """Test process_symbol passes context parameters when processor requires them."""
        processor = Mock()
        processor.process_symbol = AsyncMock()
        coordinator.set_symbol_processor(processor)

        # Force context requirement
        coordinator._process_symbol_expects_context = Mock(return_value=True)

        balances = [ScenarioBuilder.create_balance()]
        position_map = {"BTC-PERP": ScenarioBuilder.create_position()}

        await coordinator.process_symbol("BTC-PERP", balances, position_map)

        processor.process_symbol.assert_called_once_with("BTC-PERP", balances, position_map)


class TestExecutionDelegation:
    """Test execution delegation methods."""

    def test_execute_decision_delegates_to_execution_coordinator(self, coordinator, base_context):
        """Test execute_decision delegates to execution coordinator."""
        execution_coordinator = Mock()
        execution_coordinator.execute_decision = AsyncMock()

        base_context.execution_coordinator = execution_coordinator
        coordinator.update_context(base_context)

        decision = ScenarioBuilder.create_decision()
        product = ScenarioBuilder.create_product()

        # Call synchronously since we're testing delegation
        import asyncio

        async def test():
            await coordinator.execute_decision(
                "BTC-PERP", decision, Decimal("50000"), product, None
            )
            execution_coordinator.execute_decision.assert_called_once_with(
                "BTC-PERP", decision, Decimal("50000"), product, None
            )

        asyncio.run(test())

    def test_ensure_order_lock_delegates_to_execution_coordinator(self, coordinator, base_context):
        """Test ensure_order_lock delegates to execution coordinator."""
        execution_coordinator = Mock()
        execution_coordinator.ensure_order_lock = Mock(return_value=Mock())

        base_context.execution_coordinator = execution_coordinator
        coordinator.update_context(base_context)

        result = coordinator.ensure_order_lock()

        assert result == execution_coordinator.ensure_order_lock.return_value

    def test_place_order_delegates_to_execution_coordinator(self, coordinator, base_context):
        """Test place_order delegates to execution coordinator."""
        execution_coordinator = Mock()
        execution_coordinator.place_order = AsyncMock(return_value=Mock())

        base_context.execution_coordinator = execution_coordinator
        base_context.runtime_state.exec_engine = Mock()
        coordinator.update_context(base_context)

        # Call synchronously since we're testing delegation
        import asyncio

        async def test():
            await coordinator.place_order(symbol="BTC-PERP")
            execution_coordinator.place_order.assert_called_once()

        asyncio.run(test())


class TestHealthCheck:
    """Test health check functionality."""

    def test_health_check_returns_status(self, coordinator, base_context):
        """Test health_check returns proper status."""
        base_context.symbols = ("BTC-PERP", "ETH-PERP")
        base_context.runtime_state.last_decisions = {"BTC-PERP": Mock()}
        coordinator.update_context(base_context)

        status = coordinator.health_check()

        assert status.component == "strategy"
        assert status.healthy is True
        assert status.details["symbols_tracked"] == 2
        assert status.details["last_decisions"] == 1


class TestStaticUtilities:
    """Test static utility methods."""

    def test_calculate_spread_bps_with_valid_inputs(self):
        """Test calculate_spread_bps with valid bid/ask."""
        bid = Decimal("49900")
        ask = Decimal("50100")

        result = StrategyCoordinator.calculate_spread_bps(bid, ask)

        # Spread = (50100 - 49900) / ((49900 + 50100) / 2) * 10000
        # = 200 / 50000 * 10000 = 40 bps
        assert result == Decimal("40")

    def test_calculate_spread_bps_with_zero_mid(self):
        """Test calculate_spread_bps handles zero mid price."""
        bid = Decimal("0")
        ask = Decimal("0")

        result = StrategyCoordinator.calculate_spread_bps(bid, ask)

        assert result == Decimal("0")

    def test_calculate_spread_bps_with_exception(self):
        """Test calculate_spread_bps handles exceptions gracefully."""
        bid = None
        ask = None

        result = StrategyCoordinator.calculate_spread_bps(bid, ask)  # type: ignore

        assert result == Decimal("0")
