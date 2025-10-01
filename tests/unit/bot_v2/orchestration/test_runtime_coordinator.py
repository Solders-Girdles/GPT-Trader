"""Tests for runtime coordinator"""

import pytest
from unittest.mock import Mock, patch
from bot_v2.orchestration.runtime_coordinator import (
    RuntimeCoordinator,
    BrokerBootstrapError,
)
from bot_v2.orchestration.configuration import Profile


@pytest.fixture
def mock_bot():
    """Mock PerpsBot instance"""
    bot = Mock()
    bot.config = Mock()
    bot.config.profile = Profile.DEV
    bot.config.mock_broker = True
    bot.config.dry_run = False
    bot.config.symbols = ["BTC-USD"]
    bot.config.short_ma = 10
    bot.config.long_ma = 30
    bot.config.target_leverage = 2
    bot.config.trailing_stop_pct = 0.02
    bot.config.enable_shorts = True
    bot.config.reduce_only_mode = False
    bot.config.perps_position_fraction = None
    bot.config.max_leverage = None
    bot.config.derivatives_enabled = False
    bot.broker = None
    bot.risk_manager = None
    bot.registry = Mock()
    bot.registry.broker = None
    bot.registry.risk_manager = None
    bot.registry.with_updates = Mock(return_value=bot.registry)
    bot._product_map = {}
    bot.strategy_orchestrator = Mock()
    bot.strategy_orchestrator.init_strategy = Mock()
    bot.execution_coordinator = Mock()
    bot.execution_coordinator.init_execution = Mock()
    bot.config_controller = Mock()
    bot.config_controller.sync_with_risk_manager = Mock()
    bot.config_controller.reduce_only_mode = False
    bot.config_controller.set_reduce_only_mode = Mock(return_value=True)
    bot.config_controller.is_reduce_only_mode = Mock(return_value=False)
    bot.config_controller.apply_risk_update = Mock(return_value=True)
    bot.event_store = Mock()
    bot.event_store.append_metric = Mock()
    bot.event_store.append_error = Mock()
    bot.orders_store = Mock()
    bot.bot_id = "test_bot"
    bot._last_positions = {}
    return bot


@pytest.fixture
def runtime_coordinator(mock_bot):
    """Create RuntimeCoordinator instance"""
    return RuntimeCoordinator(mock_bot)


class TestRuntimeCoordinator:
    """Test suite for RuntimeCoordinator"""

    def test_initialization(self, runtime_coordinator, mock_bot):
        """Test coordinator initialization"""
        assert runtime_coordinator._bot == mock_bot

    def test_bootstrap_success(self, runtime_coordinator, mock_bot):
        """Test successful bootstrap"""
        runtime_coordinator.bootstrap()

        # Should initialize broker, risk, strategy, and execution
        assert mock_bot.broker is not None
        assert mock_bot.risk_manager is not None
        mock_bot.strategy_orchestrator.init_strategy.assert_called_once()
        mock_bot.execution_coordinator.init_execution.assert_called_once()

    def test_init_broker_from_registry(self, runtime_coordinator, mock_bot):
        """Test broker initialization from registry"""
        mock_broker = Mock()
        mock_bot.registry.broker = mock_broker

        runtime_coordinator._init_broker()

        assert mock_bot.broker == mock_broker

    def test_init_broker_deterministic_in_dev(self, runtime_coordinator, mock_bot):
        """Test deterministic broker creation in DEV mode"""
        mock_bot.config.profile = Profile.DEV
        mock_bot.registry.broker = None

        runtime_coordinator._init_broker()

        assert mock_bot.broker is not None
        mock_bot.registry.with_updates.assert_called()

    def test_init_broker_mock_when_configured(self, runtime_coordinator, mock_bot):
        """Test mock broker when configured"""
        mock_bot.config.mock_broker = True
        mock_bot.config.profile = Profile.PROD
        mock_bot.registry.broker = None

        runtime_coordinator._init_broker()

        assert mock_bot.broker is not None

    @patch('bot_v2.orchestration.runtime_coordinator.create_brokerage')
    def test_init_broker_real_production(
        self, mock_create_brokerage, runtime_coordinator, mock_bot
    ):
        """Test real broker initialization in production"""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False
        mock_bot.config.perps_paper_trading = False
        mock_bot.config.perps_force_mock = False
        mock_bot.registry.broker = None

        mock_broker = Mock()
        mock_broker.connect = Mock(return_value=True)
        mock_broker.list_products = Mock(return_value=[])
        mock_create_brokerage.return_value = mock_broker

        with patch.dict('os.environ', {
            'BROKER': 'coinbase',
            'COINBASE_API_KEY': 'test_key',
            'COINBASE_API_SECRET': 'test_secret',
        }):
            runtime_coordinator._init_broker()

        assert mock_bot.broker == mock_broker
        mock_broker.connect.assert_called_once()

    @patch('bot_v2.orchestration.runtime_coordinator.create_brokerage')
    def test_init_broker_connection_failure(
        self, mock_create_brokerage, runtime_coordinator, mock_bot
    ):
        """Test broker initialization with connection failure"""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False
        mock_bot.config.perps_paper_trading = False
        mock_bot.config.perps_force_mock = False
        mock_bot.registry.broker = None

        mock_broker = Mock()
        mock_broker.connect = Mock(return_value=False)
        mock_create_brokerage.return_value = mock_broker

        with patch.dict('os.environ', {
            'BROKER': 'coinbase',
            'COINBASE_API_KEY': 'test',
            'COINBASE_API_SECRET': 'test',
        }):
            with pytest.raises(BrokerBootstrapError):
                runtime_coordinator._init_broker()

    @patch.dict('os.environ', {'BROKER': ''})
    def test_validate_broker_environment_missing_broker(
        self, runtime_coordinator, mock_bot
    ):
        """Test broker validation with missing BROKER env"""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False
        mock_bot.config.perps_paper_trading = False
        mock_bot.config.perps_force_mock = False

        with pytest.raises(RuntimeError, match="BROKER must be set"):
            runtime_coordinator._validate_broker_environment()

    @patch.dict('os.environ', {'BROKER': 'coinbase', 'COINBASE_SANDBOX': '1'})
    def test_validate_broker_environment_sandbox_not_allowed(
        self, runtime_coordinator, mock_bot
    ):
        """Test broker validation rejects sandbox in production"""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False
        mock_bot.config.perps_paper_trading = False
        mock_bot.config.perps_force_mock = False

        with pytest.raises(RuntimeError, match="COINBASE_SANDBOX=1 is not supported"):
            runtime_coordinator._validate_broker_environment()

    def test_validate_broker_environment_skip_in_dev(
        self, runtime_coordinator, mock_bot
    ):
        """Test broker validation skipped in DEV mode"""
        mock_bot.config.profile = Profile.DEV

        # Should not raise
        runtime_coordinator._validate_broker_environment()

    @patch.dict('os.environ', {
        'BROKER': 'coinbase',
        'COINBASE_API_MODE': 'advanced',
        'COINBASE_PROD_CDP_API_KEY': 'test_key',
        'COINBASE_PROD_CDP_PRIVATE_KEY': 'test_private',
    })
    def test_validate_broker_environment_derivatives_success(
        self, runtime_coordinator, mock_bot
    ):
        """Test broker validation for derivatives"""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False
        mock_bot.config.derivatives_enabled = True
        mock_bot.config.symbols = ["BTC-PERP"]

        # Should not raise
        runtime_coordinator._validate_broker_environment()

    @patch('bot_v2.orchestration.runtime_coordinator.RiskConfig')
    def test_init_risk_manager_from_env(
        self, mock_risk_config_class, runtime_coordinator, mock_bot
    ):
        """Test risk manager initialization from environment"""
        mock_config = Mock()
        mock_risk_config_class.from_env.return_value = mock_config

        runtime_coordinator._init_risk_manager()

        assert mock_bot.risk_manager is not None
        mock_bot.config_controller.sync_with_risk_manager.assert_called()

    def test_init_risk_manager_from_registry(
        self, runtime_coordinator, mock_bot
    ):
        """Test risk manager from registry"""
        mock_risk_manager = Mock()
        mock_bot.registry.risk_manager = mock_risk_manager

        runtime_coordinator._init_risk_manager()

        assert mock_bot.risk_manager == mock_risk_manager

    def test_set_reduce_only_mode_enabled(
        self, runtime_coordinator, mock_bot
    ):
        """Test enabling reduce-only mode"""
        runtime_coordinator.set_reduce_only_mode(True, reason="test")

        mock_bot.config_controller.set_reduce_only_mode.assert_called_once_with(
            True, reason="test", risk_manager=mock_bot.risk_manager
        )

    def test_set_reduce_only_mode_disabled(
        self, runtime_coordinator, mock_bot
    ):
        """Test disabling reduce-only mode"""
        runtime_coordinator.set_reduce_only_mode(False, reason="test")

        mock_bot.config_controller.set_reduce_only_mode.assert_called_once()

    def test_is_reduce_only_mode(self, runtime_coordinator, mock_bot):
        """Test checking reduce-only mode status"""
        mock_bot.config_controller.is_reduce_only_mode.return_value = True

        result = runtime_coordinator.is_reduce_only_mode()

        assert result is True

    def test_on_risk_state_change(self, runtime_coordinator, mock_bot):
        """Test risk state change handling"""
        from bot_v2.features.live_trade.risk import RiskRuntimeState

        state = RiskRuntimeState(
            reduce_only_mode=True,
            last_reduce_only_reason="max_drawdown",
        )

        runtime_coordinator.on_risk_state_change(state)

        mock_bot.config_controller.apply_risk_update.assert_called_once_with(True)

    def test_emit_reduce_only_metric(self, runtime_coordinator, mock_bot):
        """Test reduce-only metric emission"""
        runtime_coordinator._emit_reduce_only_metric(True, "test_reason")

        mock_bot.event_store.append_metric.assert_called_once()
        call_kwargs = mock_bot.event_store.append_metric.call_args[1]
        assert call_kwargs["metrics"]["enabled"] is True
        assert call_kwargs["metrics"]["reason"] == "test_reason"

    @pytest.mark.asyncio
    async def test_reconcile_state_on_startup_dry_run(
        self, runtime_coordinator, mock_bot
    ):
        """Test startup reconciliation skipped in dry-run"""
        mock_bot.config.dry_run = True

        await runtime_coordinator.reconcile_state_on_startup()

        # Should skip reconciliation

    @pytest.mark.asyncio
    @patch('bot_v2.orchestration.runtime_coordinator.OrderReconciler')
    async def test_reconcile_state_on_startup_success(
        self, mock_reconciler_class, runtime_coordinator, mock_bot
    ):
        """Test successful startup reconciliation"""
        from unittest.mock import AsyncMock

        mock_bot.config.dry_run = False
        mock_bot.config.perps_skip_startup_reconcile = False

        mock_reconciler = Mock()
        mock_reconciler.fetch_local_open_orders = Mock(return_value={})
        mock_reconciler.fetch_exchange_open_orders = AsyncMock(return_value={})
        mock_reconciler.record_snapshot = AsyncMock()
        mock_reconciler.diff_orders = Mock(return_value=Mock(
            missing_on_exchange=[],
            missing_locally=[],
        ))
        mock_reconciler.reconcile_missing_on_exchange = AsyncMock()
        mock_reconciler.reconcile_missing_locally = Mock()
        mock_reconciler.snapshot_positions = AsyncMock(return_value={})
        mock_reconciler_class.return_value = mock_reconciler

        await runtime_coordinator.reconcile_state_on_startup()

        mock_reconciler.fetch_local_open_orders.assert_called_once()

    @pytest.mark.asyncio
    @patch('bot_v2.orchestration.runtime_coordinator.OrderReconciler')
    async def test_reconcile_state_on_startup_failure(
        self, mock_reconciler_class, runtime_coordinator, mock_bot
    ):
        """Test startup reconciliation failure handling"""
        mock_bot.config.dry_run = False
        mock_bot.config.perps_skip_startup_reconcile = False
        mock_reconciler_class.side_effect = Exception("Reconciliation failed")

        await runtime_coordinator.reconcile_state_on_startup()

        # Should enable reduce-only mode on failure
        mock_bot.config_controller.set_reduce_only_mode.assert_called_with(
            True, reason="startup_reconcile_failed", risk_manager=mock_bot.risk_manager
        )