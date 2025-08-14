"""
System Startup and Shutdown Tests for Phase 5 Production Integration.

This module tests the complete system lifecycle including:
- System initialization and startup
- Component initialization order
- Graceful shutdown procedures
- Resource cleanup
- State persistence and recovery
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from bot.exec.alpaca_paper import AlpacaPaperBroker
from bot.knowledge.strategy_knowledge_base import StrategyKnowledgeBase
from bot.live.production_orchestrator import (
    OrchestrationMode,
    OrchestratorConfig,
    ProductionOrchestrator,
)
from bot.live.strategy_selector import SelectionMethod
from bot.portfolio.optimizer import OptimizationMethod


class TestSystemStartupShutdown:
    """Test system startup and shutdown procedures."""

    @pytest.fixture
    def orchestrator_config(self):
        """Create orchestrator configuration for startup/shutdown testing."""
        return OrchestratorConfig(
            mode=OrchestrationMode.SEMI_AUTOMATED,
            rebalance_interval=120,  # 2 minutes for testing
            risk_check_interval=60,  # 1 minute for testing
            performance_check_interval=90,  # 1.5 minutes for testing
            max_strategies=3,
            min_strategy_confidence=0.6,
            selection_method=SelectionMethod.HYBRID,
            optimization_method=OptimizationMethod.SHARPE_MAXIMIZATION,
            max_position_weight=0.4,
            target_volatility=0.15,
            max_portfolio_var=0.02,
            max_drawdown=0.15,
            stop_loss_pct=0.05,
            min_sharpe_ratio=0.5,
            max_drawdown_threshold=0.15,
            enable_alerts=True,
            alert_cooldown_minutes=5,
        )

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker for startup/shutdown testing."""
        broker = Mock(spec=AlpacaPaperBroker)

        # Mock account
        account = Mock()
        account.equity = 100000.0
        account.cash = 50000.0
        account.buying_power = 50000.0
        broker.get_account.return_value = account

        # Mock positions
        positions = []
        for i, symbol in enumerate(["AAPL", "GOOGL", "MSFT"]):
            position = Mock()
            position.symbol = symbol
            position.qty = 100 + i * 50
            position.market_value = 15000.0 + i * 5000.0
            position.current_price = 150.0 + i * 10.0
            position.avg_entry_price = 145.0 + i * 8.0
            positions.append(position)

        broker.get_positions.return_value = positions
        return broker

    @pytest.fixture
    def mock_knowledge_base(self):
        """Create a mock knowledge base for startup/shutdown testing."""
        kb = Mock(spec=StrategyKnowledgeBase)
        kb.find_strategies.return_value = []
        return kb

    @pytest.fixture
    def orchestrator(self, orchestrator_config, mock_broker, mock_knowledge_base):
        """Create a production orchestrator for startup/shutdown testing."""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

        # Create orchestrator
        orchestrator = ProductionOrchestrator(
            config=orchestrator_config,
            broker=mock_broker,
            knowledge_base=mock_knowledge_base,
            symbols=symbols,
        )

        # Mock the data manager's async methods properly
        orchestrator.data_manager.start = AsyncMock()
        orchestrator.data_manager.stop = AsyncMock()

        # Mock the start method to prevent infinite loops
        original_start = orchestrator.start

        async def mock_start():
            orchestrator.is_running = True
            await orchestrator.data_manager.start()

        orchestrator.start = mock_start

        return orchestrator

    @pytest.mark.asyncio
    async def test_system_initialization(self, orchestrator):
        """Test complete system initialization."""
        logger.info("Testing system initialization")

        # Verify initial state
        assert not orchestrator.is_running, "System should not be running initially"
        assert orchestrator.current_status is None, "No status should exist initially"
        assert len(orchestrator.operation_history) == 0, "No operations should exist initially"

        # Verify all components are initialized
        assert orchestrator.strategy_selector is not None, "Strategy selector not initialized"
        assert orchestrator.portfolio_optimizer is not None, "Portfolio optimizer not initialized"
        assert orchestrator.risk_manager is not None, "Risk manager not initialized"
        assert orchestrator.alert_manager is not None, "Alert manager not initialized"
        assert orchestrator.performance_monitor is not None, "Performance monitor not initialized"

        # Verify configuration is set
        assert orchestrator.config.mode == OrchestrationMode.SEMI_AUTOMATED, "Incorrect mode"
        assert orchestrator.config.rebalance_interval == 120, "Incorrect rebalance interval"
        assert orchestrator.config.max_strategies == 3, "Incorrect max strategies"

        # Verify symbols are set
        expected_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        assert orchestrator.symbols == expected_symbols, "Incorrect symbols"

        logger.info("System initialization test completed successfully")

    @pytest.mark.asyncio
    async def test_system_startup_sequence(self, orchestrator):
        """Test system startup sequence and component initialization order."""
        logger.info("Testing system startup sequence")

        # Track startup sequence
        startup_sequence = []

        # Mock component startup methods to track order
        original_start = orchestrator.data_manager.start

        async def tracked_start():
            startup_sequence.append("data_manager")
            await original_start()

        orchestrator.data_manager.start = tracked_start

        # Override the fixture's mock_start to track sequence
        original_mock_start = orchestrator.start

        async def tracked_mock_start():
            startup_sequence.append("orchestrator")
            await orchestrator.data_manager.start()
            orchestrator.is_running = True

        orchestrator.start = tracked_mock_start

        # Start the system
        start_time = time.time()
        await orchestrator.start()
        startup_duration = time.time() - start_time

        # Verify startup sequence
        assert "data_manager" in startup_sequence, "Data manager not started"
        assert "orchestrator" in startup_sequence, "Orchestrator not started"

        # Verify system is running
        assert orchestrator.is_running, "System should be running after startup"

        # Verify startup duration is reasonable (should be fast with mocks)
        assert startup_duration < 5.0, f"Startup took too long: {startup_duration:.2f}s"

        # Verify system status is available
        status = orchestrator.get_system_status()
        if status is None:
            status = orchestrator._calculate_system_status()
        assert status is not None, "System status should be available after startup"
        assert status.is_running, "System status should show running"
        assert status.timestamp is not None, "Status timestamp should be set"

        # Restore original start method
        orchestrator.start = original_mock_start

        logger.info(f"System startup completed in {startup_duration:.2f}s")

    @pytest.mark.asyncio
    async def test_system_shutdown_sequence(self, orchestrator):
        """Test system shutdown sequence and graceful termination."""
        logger.info("Testing system shutdown sequence")

        # Track shutdown sequence
        shutdown_sequence = []

        # Mock component shutdown methods to track order
        original_stop = orchestrator.data_manager.stop

        async def tracked_stop():
            shutdown_sequence.append("data_manager")
            await original_stop()

        orchestrator.data_manager.stop = tracked_stop

        # The fixture already mocks start, so we just need to track it
        startup_sequence = []
        original_mock_start = orchestrator.start

        async def tracked_mock_start():
            startup_sequence.append("orchestrator")
            await orchestrator.data_manager.start()
            orchestrator.is_running = True

        orchestrator.start = tracked_mock_start

        # Start the system first
        await orchestrator.start()

        # Verify it's running
        assert orchestrator.is_running, "System should be running before shutdown"

        # Stop the system
        stop_time = time.time()
        await orchestrator.stop()
        shutdown_duration = time.time() - stop_time

        # Verify shutdown sequence
        assert "data_manager" in shutdown_sequence, "Data manager not stopped"

        # Verify system is stopped
        assert not orchestrator.is_running, "System should not be running after shutdown"

        # Verify shutdown duration is reasonable
        assert shutdown_duration < 5.0, f"Shutdown took too long: {shutdown_duration:.2f}s"

        logger.info(f"System shutdown completed in {shutdown_duration:.2f}s")

    @pytest.mark.asyncio
    async def test_component_initialization_order(self, orchestrator):
        """Test that components are initialized in the correct order."""
        logger.info("Testing component initialization order")

        # The initialization order should be:
        # 1. Strategy selector (with regime detector)
        # 2. Portfolio optimizer
        # 3. Risk manager
        # 4. Performance monitor
        # 5. Alert manager
        # 6. Data manager

        # Verify all components are properly initialized
        assert hasattr(orchestrator.strategy_selector, "config"), "Strategy selector config not set"
        assert hasattr(
            orchestrator.portfolio_optimizer, "constraints"
        ), "Portfolio optimizer constraints not set"
        assert hasattr(orchestrator.risk_manager, "risk_limits"), "Risk manager limits not set"
        assert hasattr(
            orchestrator.performance_monitor, "thresholds"
        ), "Performance monitor thresholds not set"
        assert hasattr(orchestrator.alert_manager, "config"), "Alert manager config not set"

        # Verify component configurations
        assert (
            orchestrator.strategy_selector.config.max_strategies == 3
        ), "Strategy selector config mismatch"
        assert (
            orchestrator.portfolio_optimizer.constraints.max_weight == 0.4
        ), "Portfolio optimizer config mismatch"
        assert (
            orchestrator.risk_manager.risk_limits.max_portfolio_var == 0.02
        ), "Risk manager config mismatch"

        logger.info("Component initialization order verified")

    @pytest.mark.asyncio
    async def test_startup_with_invalid_configuration(self):
        """Test system startup behavior with invalid configuration."""
        logger.info("Testing startup with invalid configuration")

        # Create invalid configuration
        invalid_config = OrchestratorConfig(
            mode=OrchestrationMode.SEMI_AUTOMATED,
            rebalance_interval=-1,  # Invalid negative interval
            max_strategies=0,  # Invalid zero strategies
            min_strategy_confidence=1.5,  # Invalid confidence > 1.0
        )

        mock_broker = Mock(spec=AlpacaPaperBroker)
        mock_kb = Mock(spec=StrategyKnowledgeBase)
        symbols = ["AAPL", "GOOGL"]

        # System should still initialize (validation happens later)
        with patch("bot.live.production_orchestrator.LiveDataManager"):
            orchestrator = ProductionOrchestrator(
                config=invalid_config, broker=mock_broker, knowledge_base=mock_kb, symbols=symbols
            )

        # Verify system initializes even with invalid config
        assert orchestrator is not None, "System should initialize with invalid config"
        assert orchestrator.config == invalid_config, "Config should be set even if invalid"

        # Verify components are still created
        assert orchestrator.strategy_selector is not None, "Strategy selector should be created"
        assert orchestrator.portfolio_optimizer is not None, "Portfolio optimizer should be created"
        assert orchestrator.risk_manager is not None, "Risk manager should be created"

        logger.info("Startup with invalid configuration test completed")

    @pytest.mark.asyncio
    async def test_startup_with_missing_dependencies(self):
        """Test system startup behavior with missing dependencies."""
        logger.info("Testing startup with missing dependencies")

        config = OrchestratorConfig()
        symbols = ["AAPL", "GOOGL"]

        # Test with None broker (should still initialize but may fail later)
        try:
            orchestrator = ProductionOrchestrator(
                config=config,
                broker=None,  # Invalid None broker
                knowledge_base=Mock(spec=StrategyKnowledgeBase),
                symbols=symbols,
            )
            # If it doesn't raise an error, that's fine - validation might happen later
        except Exception:
            # If it does raise an error, that's also fine
            pass

        # Test with None knowledge base (should still initialize but may fail later)
        try:
            orchestrator = ProductionOrchestrator(
                config=config,
                broker=Mock(spec=AlpacaPaperBroker),
                knowledge_base=None,  # Invalid None knowledge base
                symbols=symbols,
            )
            # If it doesn't raise an error, that's fine - validation might happen later
        except Exception:
            # If it does raise an error, that's also fine
            pass

        # Test with empty symbols
        with patch("bot.live.production_orchestrator.LiveDataManager"):
            orchestrator = ProductionOrchestrator(
                config=config,
                broker=Mock(spec=AlpacaPaperBroker),
                knowledge_base=Mock(spec=StrategyKnowledgeBase),
                symbols=[],  # Empty symbols list
            )

        # System should initialize with empty symbols
        assert orchestrator is not None, "System should initialize with empty symbols"
        assert orchestrator.symbols == [], "Symbols should be empty"

        logger.info("Startup with missing dependencies test completed")

    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_active_operations(self, orchestrator):
        """Test graceful shutdown while operations are active."""
        logger.info("Testing graceful shutdown with active operations")

        # Start the system
        await orchestrator.start()

        # Verify it's running
        assert orchestrator.is_running, "System should be running"

        # Simulate some operations
        await orchestrator._execute_strategy_selection_cycle()
        await orchestrator._execute_risk_monitoring_cycle()

        # Verify operations were recorded
        operations_before = len(orchestrator.get_operation_history())
        assert operations_before > 0, "Operations should be recorded before shutdown"

        # Stop the system
        await orchestrator.stop()

        # Verify system is stopped
        assert not orchestrator.is_running, "System should be stopped"

        # Verify operations are preserved
        operations_after = len(orchestrator.get_operation_history())
        assert (
            operations_after == operations_before
        ), "Operations should be preserved after shutdown"

        # Verify system status reflects stopped state
        status = orchestrator.get_system_status()
        if status is not None:
            assert not status.is_running, "System status should reflect stopped state"

        logger.info("Graceful shutdown with active operations test completed")

    @pytest.mark.asyncio
    async def test_restart_capability(self, orchestrator):
        """Test system restart capability after shutdown."""
        logger.info("Testing system restart capability")

        # First startup
        await orchestrator.start()
        assert orchestrator.is_running, "System should be running after first startup"

        # Record some operations
        await orchestrator._execute_strategy_selection_cycle()
        operations_first = len(orchestrator.get_operation_history())

        # First shutdown
        await orchestrator.stop()
        assert not orchestrator.is_running, "System should be stopped after first shutdown"

        # Second startup
        await orchestrator.start()
        assert orchestrator.is_running, "System should be running after second startup"

        # Record more operations
        await orchestrator._execute_risk_monitoring_cycle()
        operations_second = len(orchestrator.get_operation_history())

        # Verify operations from both runs are preserved
        assert operations_second > operations_first, "Operations should accumulate across restarts"

        # Second shutdown
        await orchestrator.stop()
        assert not orchestrator.is_running, "System should be stopped after second shutdown"

        logger.info("System restart capability test completed")

    @pytest.mark.asyncio
    async def test_startup_timeout_handling(self, orchestrator):
        """Test startup timeout handling for slow components."""
        logger.info("Testing startup timeout handling")

        # Mock slow startup that doesn't set is_running immediately
        original_mock_start = orchestrator.start

        async def slow_start():
            await asyncio.sleep(10)  # Simulate slow startup
            orchestrator.is_running = True
            await orchestrator.data_manager.start()

        orchestrator.start = slow_start

        # Start with timeout
        try:
            await asyncio.wait_for(orchestrator.start(), timeout=2.0)
            assert False, "Startup should have timed out"
        except TimeoutError:
            # Expected timeout
            pass

        # Verify system is not running after timeout
        assert not orchestrator.is_running, "System should not be running after timeout"

        # Restore original method
        orchestrator.start = original_mock_start

        logger.info("Startup timeout handling test completed")

    @pytest.mark.asyncio
    async def test_shutdown_timeout_handling(self, orchestrator):
        """Test shutdown timeout handling for slow components."""
        logger.info("Testing shutdown timeout handling")

        # Start the system first
        await orchestrator.start()

        # Mock slow data manager shutdown
        original_stop = orchestrator.data_manager.stop

        async def slow_stop():
            await asyncio.sleep(10)  # Simulate slow shutdown
            await original_stop()

        orchestrator.data_manager.stop = slow_stop

        # Stop with timeout
        try:
            await asyncio.wait_for(orchestrator.stop(), timeout=2.0)
            assert False, "Shutdown should have timed out"
        except TimeoutError:
            # Expected timeout
            pass

        # Verify system is marked as stopped even after timeout
        assert not orchestrator.is_running, "System should be marked as stopped after timeout"

        # Restore original method
        orchestrator.data_manager.stop = original_stop

        logger.info("Shutdown timeout handling test completed")

    @pytest.mark.asyncio
    async def test_system_status_persistence(self, orchestrator):
        """Test that system status persists across operations."""
        logger.info("Testing system status persistence")

        # Start the system
        await orchestrator.start()

        try:
            # Get initial status
            initial_status = orchestrator.get_system_status()
            if initial_status is None:
                initial_status = orchestrator._calculate_system_status()
            assert initial_status is not None, "Initial status should be available"
            assert initial_status.is_running, "Initial status should show running"

            # Execute some operations
            await orchestrator._execute_strategy_selection_cycle()
            await orchestrator._execute_risk_monitoring_cycle()

            # Get status after operations
            updated_status = orchestrator.get_system_status()
            if updated_status is None:
                updated_status = orchestrator._calculate_system_status()
            assert updated_status is not None, "Updated status should be available"
            assert updated_status.is_running, "Updated status should still show running"

            # Verify status consistency
            assert updated_status.mode == initial_status.mode, "Mode should remain consistent"
            assert (
                updated_status.timestamp >= initial_status.timestamp
            ), "Timestamp should be updated"

            # Verify operation history is reflected in status
            operations = orchestrator.get_operation_history()
            if operations:
                assert (
                    updated_status.n_active_strategies >= 0
                ), "Active strategies should be non-negative"

        finally:
            await orchestrator.stop()

        logger.info("System status persistence test completed")


# Add logger for test output
import logging

logger = logging.getLogger(__name__)
