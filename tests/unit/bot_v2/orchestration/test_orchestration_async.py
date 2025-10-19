"""Async tests for orchestration components."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot_v2.features.brokerages.core.interfaces import Balance, Position
from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision
from bot_v2.orchestration.execution.guards import GuardManager, RuntimeGuardState
from bot_v2.orchestration.execution.state_collection import StateCollector
from bot_v2.orchestration.strategy_orchestrator import StrategyOrchestrator, SymbolProcessingContext


@pytest.fixture
def mock_broker():
    return Mock()


@pytest.fixture
def mock_risk_manager():
    return Mock()


@pytest.fixture
def mock_bot():
    bot = Mock()
    bot.config = Mock()
    bot.config.profile = Mock()
    bot.config.profile.SPOT = "spot"
    bot.config.symbols = ["BTC-USD", "ETH-USD"]
    bot.config.short_ma = 10
    bot.config.long_ma = 30
    bot.config.target_leverage = 5
    bot.config.trailing_stop_pct = Decimal("0.05")
    bot.config.enable_shorts = True
    bot.config.perps_position_fraction = Decimal("0.1")
    bot.config.derivatives_enabled = True
    bot.runtime_state = Mock()
    bot.runtime_state.symbol_strategies = {}
    bot.runtime_state.strategy = None
    bot.last_decisions = {}
    bot.mark_windows = {"BTC-USD": [Decimal("50000"), Decimal("50100")], "ETH-USD": []}
    bot.risk_manager = mock_risk_manager
    return bot


@pytest.fixture
def mock_spot_profile_service():
    service = Mock()
    service.load.return_value = {
        "BTC-USD": {"short_window": 5, "long_window": 15, "position_fraction": 0.05}
    }
    service.get.return_value = {}
    return service


@pytest.fixture
def orchestrator(mock_bot, mock_spot_profile_service):
    return StrategyOrchestrator(mock_bot, mock_spot_profile_service)


@pytest.fixture
def state_collector(mock_broker):
    return StateCollector(mock_broker)


@pytest.fixture
def guard_manager(mock_broker, mock_risk_manager):
    def mock_equity_calc(balances):
        return Decimal("10000"), balances, Decimal("10000")

    def mock_cancel_orders():
        return 0

    def mock_invalidate_cache():
        pass

    return GuardManager(
        mock_broker,
        mock_risk_manager,
        mock_equity_calc,
        mock_cancel_orders,
        mock_invalidate_cache,
    )


class TestStrategyOrchestratorAsync:
    """Test async paths in StrategyOrchestrator."""

    @pytest.mark.asyncio
    async def test_process_symbol_missing_marks_returns_early(self, orchestrator, mock_bot):
        """Test that process_symbol returns early when marks are missing."""
        mock_bot.mark_windows = {"BTC-USD": []}  # Empty marks

        await orchestrator.process_symbol("BTC-USD")

        # Should not call execute_decision due to missing marks
        mock_bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_symbol_kill_switch_engaged(self, orchestrator, mock_bot):
        """Test that process_symbol returns early when kill switch is engaged."""
        mock_bot.risk_manager.config.kill_switch_enabled = True

        await orchestrator.process_symbol("BTC-USD")

        # Should not proceed with decision execution
        mock_bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_symbol_missing_product_metadata(self, orchestrator, mock_bot):
        """Test process_symbol when product metadata is missing."""
        mock_bot.get_product.side_effect = Exception("Product not found")

        await orchestrator.process_symbol("BTC-USD")

        # Should log warning but not crash
        mock_bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_symbol_full_flow(self, orchestrator, mock_bot):
        """Test full async flow of process_symbol."""
        # Setup mocks
        mock_balance = Mock(spec=Balance)
        mock_balance.asset = "USD"
        mock_balance.available = Decimal("10000")
        mock_balance.total = Decimal("10000")

        mock_position = Mock(spec=Position)
        mock_position.symbol = "BTC-USD"
        mock_position.quantity = Decimal("1.0")
        mock_position.side = "long"

        mock_bot.broker.list_balances = AsyncMock(return_value=[mock_balance])
        mock_bot.broker.list_positions = AsyncMock(return_value=[mock_position])
        mock_bot.get_product.return_value = Mock()
        mock_bot.risk_manager.config.kill_switch_enabled = False
        mock_bot.config.profile.SPOT = "not_spot"  # Use perps profile

        # Mock strategy decision
        mock_strategy = Mock()
        mock_strategy.decide.return_value = Decision(action=Action.BUY, reason="test")
        orchestrator.get_strategy = Mock(return_value=mock_strategy)

        # Mock risk gates
        mock_bot.risk_manager.check_volatility_circuit_breaker.return_value.triggered = False
        mock_bot.risk_manager.check_mark_staleness.return_value = False

        await orchestrator.process_symbol("BTC-USD")

        # Verify execution was attempted
        mock_bot.execute_decision.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_balances_async_fallback(self, orchestrator, mock_bot):
        """Test async fallback for balance fetching."""
        balances = [Mock(spec=Balance)]

        # Test with provided balances
        result = await orchestrator._ensure_balances(balances)
        assert result == balances

        # Test with None (should call broker)
        mock_bot.broker.list_balances = AsyncMock(return_value=balances)
        result = await orchestrator._ensure_balances(None)
        assert result == balances

    @pytest.mark.asyncio
    async def test_ensure_positions_async_fallback(self, orchestrator, mock_bot):
        """Test async fallback for position fetching."""
        positions = {"BTC-USD": Mock(spec=Position)}

        # Test with provided positions
        result = await orchestrator._ensure_positions(positions)
        assert result == positions

        # Test with None (should call broker)
        mock_bot.broker.list_positions = AsyncMock(return_value=list(positions.values()))
        result = await orchestrator._ensure_positions(None)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_apply_spot_filters_insufficient_data(self, orchestrator, mock_bot):
        """Test spot filters when insufficient candle data."""
        context = SymbolProcessingContext(
            symbol="BTC-USD",
            balances=[],
            equity=Decimal("10000"),
            positions={},
            position_state=None,
            position_quantity=Decimal("0"),
            marks=[Decimal("50000")],
            product=None,
        )

        # Mock empty candle data
        mock_bot.broker.get_candles = AsyncMock(return_value=[])

        decision = Decision(action=Action.BUY, reason="test")
        result = await orchestrator._apply_spot_filters(context, decision)

        assert result.action == Action.HOLD
        assert "indicator_data_unavailable" in result.reason

    @pytest.mark.asyncio
    async def test_apply_spot_filters_volume_filter_block(self, orchestrator, mock_bot):
        """Test spot volume filter blocking entry."""
        from datetime import datetime

        # Mock candle data
        mock_candle = Mock()
        mock_candle.close = Decimal("50000")
        mock_candle.volume = Decimal("100")  # Low volume
        mock_candle.ts = datetime.utcnow()

        context = SymbolProcessingContext(
            symbol="BTC-USD",
            balances=[],
            equity=Decimal("10000"),
            positions={},
            position_state=None,
            position_quantity=Decimal("0"),
            marks=[Decimal("50000")],
            product=None,
        )

        mock_bot.broker.get_candles = AsyncMock(return_value=[mock_candle] * 30)

        # Configure volume filter to block
        mock_spot_profile_service = orchestrator._spot_profiles
        mock_spot_profile_service.get.return_value = {
            "volume_filter": {"window": 20, "multiplier": 2.0}
        }

        decision = Decision(action=Action.BUY, reason="test")
        result = await orchestrator._apply_spot_filters(context, decision)

        assert result.action == Action.HOLD
        assert "volume_filter_blocked" in result.reason

    @pytest.mark.asyncio
    async def test_apply_spot_filters_momentum_filter_block(self, orchestrator, mock_bot):
        """Test spot momentum filter blocking entry."""
        from datetime import datetime

        # Mock candle data with oversold RSI
        mock_candles = []
        for i in range(30):
            candle = Mock()
            candle.close = Decimal("50000") + Decimal(str(i))  # Trending up
            candle.volume = Decimal("1000")
            candle.ts = datetime.utcnow()
            mock_candles.append(candle)

        context = SymbolProcessingContext(
            symbol="BTC-USD",
            balances=[],
            equity=Decimal("10000"),
            positions={},
            position_state=None,
            position_quantity=Decimal("0"),
            marks=[Decimal("50000")],
            product=None,
        )

        mock_bot.broker.get_candles = AsyncMock(return_value=mock_candles)

        # Configure momentum filter to block
        mock_spot_profile_service = orchestrator._spot_profiles
        mock_spot_profile_service.get.return_value = {
            "momentum_filter": {"window": 14, "overbought": 70, "oversold": 30}
        }

        decision = Decision(action=Action.BUY, reason="test")
        result = await orchestrator._apply_spot_filters(context, decision)

        assert result.action == Action.HOLD
        assert "momentum_filter_blocked" in result.reason

    @pytest.mark.asyncio
    async def test_apply_spot_filters_trend_filter_block(self, orchestrator, mock_bot):
        """Test spot trend filter blocking entry."""
        from datetime import datetime

        # Mock candle data with insufficient trend
        mock_candles = []
        for i in range(30):
            candle = Mock()
            candle.close = Decimal("50000")  # Flat trend
            candle.volume = Decimal("1000")
            candle.ts = datetime.utcnow()
            mock_candles.append(candle)

        context = SymbolProcessingContext(
            symbol="BTC-USD",
            balances=[],
            equity=Decimal("10000"),
            positions={},
            position_state=None,
            position_quantity=Decimal("0"),
            marks=[Decimal("50000")],
            product=None,
        )

        mock_bot.broker.get_candles = AsyncMock(return_value=mock_candles)

        # Configure trend filter to block
        mock_spot_profile_service = orchestrator._spot_profiles
        mock_spot_profile_service.get.return_value = {
            "trend_filter": {"window": 20, "min_slope": 0.001}
        }

        decision = Decision(action=Action.BUY, reason="test")
        result = await orchestrator._apply_spot_filters(context, decision)

        assert result.action == Action.HOLD
        assert "trend_filter_blocked" in result.reason

    @pytest.mark.asyncio
    async def test_apply_spot_filters_volatility_filter_block(self, orchestrator, mock_bot):
        """Test spot volatility filter blocking entry."""
        from datetime import datetime

        # Mock candle data with high volatility
        mock_candles = []
        for i in range(30):
            candle = Mock()
            # Create volatile closes
            volatility = Decimal("0.1") if i % 2 == 0 else Decimal("-0.1")
            candle.close = Decimal("50000") * (Decimal("1") + volatility)
            candle.high = candle.close * Decimal("1.01")
            candle.low = candle.close * Decimal("0.99")
            candle.volume = Decimal("1000")
            candle.ts = datetime.utcnow()
            mock_candles.append(candle)

        context = SymbolProcessingContext(
            symbol="BTC-USD",
            balances=[],
            equity=Decimal("10000"),
            positions={},
            position_state=None,
            position_quantity=Decimal("0"),
            marks=[Decimal("50000")],
            product=None,
        )

        mock_bot.broker.get_candles = AsyncMock(return_value=mock_candles)

        # Configure volatility filter to block
        mock_spot_profile_service = orchestrator._spot_profiles
        mock_spot_profile_service.get.return_value = {
            "volatility_filter": {"window": 20, "min_vol": 0.001, "max_vol": 0.01}
        }

        decision = Decision(action=Action.BUY, reason="test")
        result = await orchestrator._apply_spot_filters(context, decision)

        assert result.action == Action.HOLD
        assert "volatility_filter_blocked" in result.reason


class TestStateCollectorAsync:
    """Test async paths in StateCollector."""

    def test_collect_account_state_sync_fallback(self, state_collector, mock_broker):
        """Test that StateCollector uses sync broker methods."""
        mock_balance = Mock(spec=Balance)
        mock_balance.asset = "USD"
        mock_balance.available = Decimal("10000")

        mock_position = Mock(spec=Position)
        mock_position.symbol = "BTC-USD"

        mock_broker.list_balances.return_value = [mock_balance]
        mock_broker.list_positions.return_value = [mock_position]

        balances, equity, collateral_balances, total_balance, positions = (
            state_collector.collect_account_state()
        )

        assert len(balances) == 1
        assert equity == Decimal("10000")
        assert len(positions) == 1

    def test_resolve_effective_price_fallback_chain(self, state_collector, mock_broker):
        """Test price resolution fallback chain."""
        product = Mock()
        product.bid_price = None
        product.ask_price = None
        product.price = None
        product.quote_increment = Decimal("0.01")

        # Test quote increment fallback
        price = state_collector.resolve_effective_price("BTC-USD", "buy", None, product)
        assert price == Decimal("10.0")  # 0.01 * 100


class TestGuardManagerAsync:
    """Test async guard execution patterns."""

    def test_run_runtime_guards_force_full(self, guard_manager, mock_broker, mock_risk_manager):
        """Test forced full guard run."""
        mock_balance = Mock(spec=Balance)
        mock_balance.asset = "USD"
        mock_balance.available = Decimal("10000")

        mock_broker.list_balances.return_value = [mock_balance]
        mock_broker.list_positions.return_value = []

        # Mock risk manager methods
        mock_risk_manager.track_daily_pnl.return_value = False
        mock_risk_manager.check_liquidation_buffer.return_value = None
        mock_risk_manager.check_mark_staleness.return_value = False
        mock_risk_manager.append_risk_metrics.return_value = None
        mock_risk_manager.check_correlation_risk.return_value = None
        mock_risk_manager.check_volatility_circuit_breaker.return_value.triggered = False

        state = guard_manager.run_runtime_guards(force_full=True)

        assert isinstance(state, RuntimeGuardState)
        assert state.equity == Decimal("10000")

    def test_run_runtime_guards_incremental_caching(
        self, guard_manager, mock_broker, mock_risk_manager
    ):
        """Test incremental guard runs with caching."""
        mock_balance = Mock(spec=Balance)
        mock_balance.asset = "USD"
        mock_balance.available = Decimal("10000")

        mock_broker.list_balances.return_value = [mock_balance]
        mock_broker.list_positions.return_value = []

        # Mock risk manager methods
        mock_risk_manager.track_daily_pnl.return_value = False
        mock_risk_manager.check_liquidation_buffer.return_value = None
        mock_risk_manager.check_mark_staleness.return_value = False
        mock_risk_manager.append_risk_metrics.return_value = None
        mock_risk_manager.check_correlation_risk.return_value = None
        mock_risk_manager.check_volatility_circuit_breaker.return_value.triggered = False

        # First run (full)
        state1 = guard_manager.run_runtime_guards(force_full=True)

        # Second run (should be incremental)
        state2 = guard_manager.run_runtime_guards(force_full=False)

        # Should return cached state
        assert state1 is state2

    def test_guard_daily_loss_triggered_cancel_orders(
        self, guard_manager, mock_broker, mock_risk_manager
    ):
        """Test daily loss guard triggering order cancellation."""
        state = RuntimeGuardState(
            timestamp=0.0,
            balances=[],
            equity=Decimal("10000"),
            positions=[],
            positions_pnl={},
            positions_dict={},
        )

        mock_risk_manager.track_daily_pnl.return_value = True  # Loss triggered

        with patch.object(guard_manager, "_cancel_all_orders") as mock_cancel:
            guard_manager.guard_daily_loss(state)

        mock_cancel.assert_called_once()

    def test_guard_volatility_circuit_breaker_with_candles(
        self, guard_manager, mock_broker, mock_risk_manager
    ):
        """Test volatility guard with candle data fetching."""
        state = RuntimeGuardState(
            timestamp=0.0,
            balances=[],
            equity=Decimal("10000"),
            positions=[],
            positions_pnl={},
            positions_dict={"BTC-USD": {"quantity": Decimal("1.0")}},
        )

        # Mock candle data
        mock_candle = Mock()
        mock_candle.close = Decimal("50000")
        mock_broker.get_candles.return_value = [mock_candle] * 25

        mock_risk_manager.config.volatility_window_periods = 20
        mock_risk_manager.check_volatility_circuit_breaker.return_value.triggered = True

        guard_manager.guard_volatility(state)

        mock_broker.get_candles.assert_called()
        assert len(state.guard_events) == 1

    def test_guard_mark_staleness_with_missing_marks(
        self, guard_manager, mock_broker, mock_risk_manager
    ):
        """Test mark staleness guard when marks are missing."""
        state = RuntimeGuardState(
            timestamp=0.0,
            balances=[],
            equity=Decimal("10000"),
            positions=[],
            positions_pnl={},
            positions_dict={},
        )

        # Mock broker with mark cache
        mock_broker._mark_cache.get_mark.return_value = None
        mock_risk_manager.last_mark_update = {"BTC-USD": 0.0}

        guard_manager.guard_mark_staleness(state)

        mock_risk_manager.check_mark_staleness.assert_called_with("BTC-USD")

    def test_guard_correlation_risk_computation_error(self, guard_manager, mock_risk_manager):
        """Test correlation guard with computation error."""
        state = RuntimeGuardState(
            timestamp=0.0,
            balances=[],
            equity=Decimal("10000"),
            positions=[],
            positions_pnl={},
            positions_dict={"BTC-USD": {"quantity": Decimal("1.0")}},
        )

        mock_risk_manager.check_correlation_risk.side_effect = Exception("Computation failed")

        with pytest.raises(Exception):  # Should be wrapped in RiskGuardComputationError
            guard_manager.guard_correlation(state)
