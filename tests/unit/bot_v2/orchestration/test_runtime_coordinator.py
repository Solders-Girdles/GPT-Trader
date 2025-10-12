"""
Tests for RuntimeCoordinator.

Tests broker/risk bootstrapping, runtime safety toggles,
and startup state reconciliation.
"""

from decimal import Decimal
from unittest.mock import Mock, MagicMock, AsyncMock, patch

import pytest

from bot_v2.orchestration.runtime_coordinator import (
    RuntimeCoordinator,
    BrokerBootstrapError,
)
from bot_v2.orchestration.configuration import Profile
from bot_v2.features.live_trade.risk import RiskRuntimeState
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState


@pytest.fixture
def mock_bot():
    """Create mock PerpsBot instance."""
    bot = Mock()
    bot.bot_id = "test_bot"
    bot.config = Mock()
    bot.config.profile = Profile.PROD
    bot.config.mock_broker = False
    bot.config.dry_run = False
    bot.config.symbols = []
    bot.config.reduce_only_mode = False
    bot.config.max_leverage = None
    bot.config.perps_paper_trading = False
    bot.config.perps_force_mock = False
    bot.config.derivatives_enabled = False
    bot.config_controller = Mock()
    bot.config_controller.reduce_only_mode = False
    bot.config_controller.set_reduce_only_mode = Mock(return_value=True)
    bot.config_controller.is_reduce_only_mode = Mock(return_value=False)
    bot.config_controller.apply_risk_update = Mock(return_value=True)
    bot.config_controller.sync_with_risk_manager = Mock()
    bot.registry = Mock()
    bot.registry.broker = None
    bot.registry.risk_manager = None
    bot.registry.with_updates = Mock(return_value=bot.registry)
    bot.event_store = Mock()
    bot.orders_store = Mock()
    state = PerpsBotRuntimeState([])
    bot.runtime_state = state
    bot._product_map = state.product_map
    bot._last_positions = state.last_positions
    bot.strategy_orchestrator = Mock()
    bot.execution_coordinator = Mock()
    return bot


@pytest.fixture
def coordinator(mock_bot):
    """Create RuntimeCoordinator instance."""
    return RuntimeCoordinator(bot=mock_bot)


class TestRuntimeCoordinatorInitialization:
    """Test RuntimeCoordinator initialization."""

    def test_initialization_with_bot(self, mock_bot):
        """Test coordinator initializes with bot reference."""
        coordinator = RuntimeCoordinator(bot=mock_bot)

        assert coordinator._bot == mock_bot


class TestBootstrap:
    """Test bootstrap method."""

    def test_bootstrap_calls_initialization_methods(self, coordinator, mock_bot):
        """Test bootstrap orchestrates initialization."""
        with patch.object(coordinator, "_init_broker"):
            with patch.object(coordinator, "_init_risk_manager"):
                coordinator.bootstrap()

                coordinator._init_broker.assert_called_once()
                coordinator._init_risk_manager.assert_called_once()
                mock_bot.strategy_orchestrator.init_strategy.assert_called_once()
                mock_bot.execution_coordinator.init_execution.assert_called_once()


class TestInitBroker:
    """Test _init_broker method."""

    def test_uses_broker_from_registry_when_available(self, coordinator, mock_bot):
        """Test uses broker from registry when present."""
        registry_broker = Mock()
        mock_bot.registry.broker = registry_broker

        coordinator._init_broker()

        assert mock_bot.broker == registry_broker

    def test_initializes_deterministic_broker_when_mock_enabled(self, coordinator, mock_bot):
        """Test initializes deterministic broker when mock flag set."""
        mock_bot.config.mock_broker = True

        coordinator._init_broker()

        assert mock_bot.broker is not None
        # Should be DeterministicBroker
        assert type(mock_bot.broker).__name__ == "DeterministicBroker"

    def test_initializes_deterministic_broker_for_dev_profile(self, coordinator, mock_bot):
        """Test initializes deterministic broker for DEV profile."""
        mock_bot.config.profile = Profile.DEV
        mock_bot.config.mock_broker = False

        coordinator._init_broker()

        assert mock_bot.broker is not None
        assert type(mock_bot.broker).__name__ == "DeterministicBroker"

    def test_initializes_real_broker_when_not_mock(self, coordinator, mock_bot):
        """Test initializes real broker when not in mock mode."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False
        mock_bot.config.perps_paper_trading = False
        mock_bot.config.perps_force_mock = False
        mock_product = Mock()
        mock_product.symbol = "BTC-PERP"

        with patch.object(coordinator, "_validate_broker_environment"):
            with patch("bot_v2.orchestration.runtime_coordinator.create_brokerage") as mock_create:
                real_broker = Mock()
                real_broker.connect = Mock(return_value=True)
                real_broker.list_products = Mock(return_value=[mock_product])
                mock_create.return_value = (
                    real_broker,
                    mock_bot.event_store,
                    Mock(),
                    Mock(),
                )

                coordinator._init_broker()

                assert mock_bot.broker == real_broker
                assert "BTC-PERP" in mock_bot.runtime_state.product_map

    def test_raises_bootstrap_error_when_connection_fails(self, coordinator, mock_bot):
        """Test raises BrokerBootstrapError when connection fails."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False
        mock_bot.config.perps_paper_trading = False
        mock_bot.config.perps_force_mock = False

        with patch.object(coordinator, "_validate_broker_environment"):
            with patch("bot_v2.orchestration.runtime_coordinator.create_brokerage") as mock_create:
                real_broker = Mock()
                real_broker.connect = Mock(return_value=False)
                mock_create.return_value = (
                    real_broker,
                    mock_bot.event_store,
                    Mock(),
                    Mock(),
                )

                with pytest.raises(BrokerBootstrapError):
                    coordinator._init_broker()


class TestValidateBrokerEnvironment:
    """Test _validate_broker_environment method."""

    def test_skips_validation_for_dev_profile(self, coordinator, mock_bot):
        """Test skips validation for DEV profile."""
        mock_bot.config.profile = Profile.DEV

        # Should not raise
        coordinator._validate_broker_environment()

    def test_skips_validation_when_mock_broker_enabled(self, coordinator, mock_bot):
        """Test skips validation when mock_broker is True."""
        mock_bot.config.mock_broker = True

        # Should not raise
        coordinator._validate_broker_environment()

    def test_raises_when_broker_env_not_coinbase(self, coordinator, mock_bot):
        """Test raises when BROKER env is not coinbase."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False
        mock_bot.config.perps_paper_trading = False
        mock_bot.config.perps_force_mock = False

        with patch.dict("os.environ", {"BROKER": "binance"}, clear=True):
            with pytest.raises(RuntimeError, match="BROKER must be set"):
                coordinator._validate_broker_environment()

    def test_raises_when_sandbox_enabled_for_live(self, coordinator, mock_bot):
        """Test raises when COINBASE_SANDBOX=1 for live trading."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False
        mock_bot.config.perps_paper_trading = False
        mock_bot.config.perps_force_mock = False

        with patch.dict("os.environ", {"BROKER": "coinbase", "COINBASE_SANDBOX": "1"}, clear=True):
            with pytest.raises(RuntimeError, match="COINBASE_SANDBOX"):
                coordinator._validate_broker_environment()

    def test_raises_when_perp_without_derivatives(self, coordinator, mock_bot):
        """Test raises when perp symbol but derivatives not enabled."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False
        mock_bot.config.perps_paper_trading = False
        mock_bot.config.perps_force_mock = False
        mock_bot.config.derivatives_enabled = False
        mock_bot.config.symbols = ["BTC-PERP"]

        with patch.dict("os.environ", {"BROKER": "coinbase"}, clear=True):
            with pytest.raises(RuntimeError, match="perpetual but.*not enabled"):
                coordinator._validate_broker_environment()

    def test_raises_when_missing_cdp_credentials(self, coordinator, mock_bot):
        """Test raises when CDP credentials missing for derivatives."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False
        mock_bot.config.perps_paper_trading = False
        mock_bot.config.perps_force_mock = False
        mock_bot.config.derivatives_enabled = True

        with patch.dict(
            "os.environ",
            {
                "BROKER": "coinbase",
                "COINBASE_API_MODE": "advanced",
            },
            clear=True,
        ):
            with pytest.raises(RuntimeError, match="Missing CDP JWT"):
                coordinator._validate_broker_environment()


class TestInitRiskManager:
    """Test _init_risk_manager method."""

    def test_uses_risk_manager_from_registry(self, coordinator, mock_bot):
        """Test uses risk manager from registry when present."""
        registry_risk_manager = Mock()
        mock_bot.registry.risk_manager = registry_risk_manager

        coordinator._init_risk_manager()

        assert mock_bot.risk_manager == registry_risk_manager
        registry_risk_manager.set_state_listener.assert_called_once()
        registry_risk_manager.set_reduce_only_mode.assert_called_once()

    def test_creates_risk_manager_from_env(self, coordinator, mock_bot):
        """Test creates risk manager from environment when registry empty."""
        with patch("bot_v2.orchestration.runtime_coordinator.RiskConfig") as mock_config:
            risk_instance = Mock()
            mock_config.from_env = Mock(return_value=risk_instance)

            with patch("bot_v2.orchestration.runtime_coordinator.LiveRiskManager") as mock_manager:
                manager_instance = Mock()
                mock_manager.return_value = manager_instance

                coordinator._init_risk_manager()

                mock_config.from_env.assert_called_once()
                assert mock_bot.risk_manager == manager_instance

    def test_applies_max_leverage_override(self, coordinator, mock_bot):
        """Test applies max_leverage override from config."""
        mock_bot.config.max_leverage = 10

        with patch("bot_v2.orchestration.runtime_coordinator.RiskConfig") as mock_config:
            risk_instance = Mock()
            mock_config.from_env = Mock(return_value=risk_instance)

            with patch("bot_v2.orchestration.runtime_coordinator.LiveRiskManager"):
                coordinator._init_risk_manager()

                assert risk_instance.max_leverage == 10

    def test_sets_risk_info_provider_from_broker(self, coordinator, mock_bot):
        """Test sets risk info provider from broker when available."""
        mock_bot.broker = Mock()
        mock_bot.broker.get_position_risk = Mock()

        with patch("bot_v2.orchestration.runtime_coordinator.RiskConfig"):
            with patch("bot_v2.orchestration.runtime_coordinator.LiveRiskManager") as mock_manager:
                manager_instance = Mock()
                mock_manager.return_value = manager_instance

                coordinator._init_risk_manager()

                manager_instance.set_risk_info_provider.assert_called_once()


class TestSetReduceOnlyMode:
    """Test set_reduce_only_mode method."""

    def test_sets_reduce_only_mode_enabled(self, coordinator, mock_bot):
        """Test sets reduce-only mode to enabled."""
        coordinator.set_reduce_only_mode(True, reason="test_safety")

        mock_bot.config_controller.set_reduce_only_mode.assert_called_once_with(
            True, reason="test_safety", risk_manager=mock_bot.risk_manager
        )

    def test_emits_metric_when_mode_changed(self, coordinator, mock_bot):
        """Test emits metric when reduce-only mode changes."""
        coordinator.set_reduce_only_mode(True, reason="test")

        mock_bot.event_store.append_metric.assert_called_once()
        call_kwargs = mock_bot.event_store.append_metric.call_args.kwargs
        assert call_kwargs["metrics"]["event_type"] == "reduce_only_mode_changed"
        assert call_kwargs["metrics"]["enabled"] is True
        assert call_kwargs["metrics"]["reason"] == "test"

    def test_returns_early_when_controller_returns_false(self, coordinator, mock_bot):
        """Test returns early when controller returns False."""
        mock_bot.config_controller.set_reduce_only_mode = Mock(return_value=False)

        coordinator.set_reduce_only_mode(True, reason="test")

        # Should not emit metric
        mock_bot.event_store.append_metric.assert_not_called()


class TestIsReduceOnlyMode:
    """Test is_reduce_only_mode method."""

    def test_returns_true_when_mode_active(self, coordinator, mock_bot):
        """Test returns True when reduce-only mode is active."""
        mock_bot.config_controller.is_reduce_only_mode = Mock(return_value=True)

        result = coordinator.is_reduce_only_mode()

        assert result is True

    def test_returns_false_when_mode_inactive(self, coordinator, mock_bot):
        """Test returns False when reduce-only mode is inactive."""
        mock_bot.config_controller.is_reduce_only_mode = Mock(return_value=False)

        result = coordinator.is_reduce_only_mode()

        assert result is False


class TestOnRiskStateChange:
    """Test on_risk_state_change callback."""

    def test_handles_reduce_only_mode_change(self, coordinator, mock_bot):
        """Test handles reduce-only mode state change from risk manager."""
        state = RiskRuntimeState(
            reduce_only_mode=True,
            last_reduce_only_reason="volatility_spike",
        )

        coordinator.on_risk_state_change(state)

        mock_bot.config_controller.apply_risk_update.assert_called_once_with(True)
        mock_bot.event_store.append_metric.assert_called_once()

    def test_returns_early_when_controller_returns_false(self, coordinator, mock_bot):
        """Test returns early when controller returns False."""
        mock_bot.config_controller.apply_risk_update = Mock(return_value=False)

        state = RiskRuntimeState(
            reduce_only_mode=True,
            last_reduce_only_reason="test",
        )

        coordinator.on_risk_state_change(state)

        # Should not emit metric
        mock_bot.event_store.append_metric.assert_not_called()


class TestReconcileStateOnStartup:
    """Test reconcile_state_on_startup async method."""

    @pytest.mark.asyncio
    async def test_skips_reconciliation_in_dry_run(self, coordinator, mock_bot):
        """Test skips reconciliation when dry_run is True."""
        mock_bot.config.dry_run = True

        await coordinator.reconcile_state_on_startup()

        # Should not create reconciler
        mock_bot.orders_store.get_open_orders.assert_not_called()

    @pytest.mark.asyncio
    async def test_performs_order_reconciliation(self, coordinator, mock_bot):
        """Test performs order reconciliation on startup."""
        mock_bot.config.dry_run = False
        mock_bot.config.perps_skip_startup_reconcile = False

        with patch(
            "bot_v2.orchestration.runtime_coordinator.OrderReconciler"
        ) as mock_reconciler_class:
            reconciler = Mock()
            reconciler.fetch_local_open_orders = Mock(return_value={})
            reconciler.fetch_exchange_open_orders = AsyncMock(return_value={})
            reconciler.record_snapshot = AsyncMock()
            reconciler.diff_orders = Mock(
                return_value=Mock(missing_on_exchange=[], missing_locally=[])
            )
            reconciler.reconcile_missing_on_exchange = AsyncMock()
            reconciler.reconcile_missing_locally = Mock()
            reconciler.snapshot_positions = AsyncMock(return_value=None)
            mock_reconciler_class.return_value = reconciler

            await coordinator.reconcile_state_on_startup()

            reconciler.fetch_local_open_orders.assert_called_once()
            reconciler.fetch_exchange_open_orders.assert_called_once()
            reconciler.diff_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_enables_reduce_only_on_reconciliation_error(self, coordinator, mock_bot):
        """Test enables reduce-only mode on reconciliation error."""
        mock_bot.config.dry_run = False
        mock_bot.config.perps_skip_startup_reconcile = False

        with patch(
            "bot_v2.orchestration.runtime_coordinator.OrderReconciler"
        ) as mock_reconciler_class:
            mock_reconciler_class.side_effect = Exception("reconciliation failed")

            with patch.object(coordinator, "set_reduce_only_mode") as mock_set_reduce:
                await coordinator.reconcile_state_on_startup()

                mock_set_reduce.assert_called_once_with(True, reason="startup_reconcile_failed")

    @pytest.mark.asyncio
    async def test_snapshots_initial_positions(self, coordinator, mock_bot):
        """Test snapshots initial positions on successful reconciliation."""
        mock_bot.config.dry_run = False
        mock_bot.config.perps_skip_startup_reconcile = False
        initial_positions = {"BTC-PERP": {"quantity": "0.5", "side": "long"}}

        with patch(
            "bot_v2.orchestration.runtime_coordinator.OrderReconciler"
        ) as mock_reconciler_class:
            reconciler = Mock()
            reconciler.fetch_local_open_orders = Mock(return_value={})
            reconciler.fetch_exchange_open_orders = AsyncMock(return_value={})
            reconciler.record_snapshot = AsyncMock()
            reconciler.diff_orders = Mock(
                return_value=Mock(missing_on_exchange=[], missing_locally=[])
            )
            reconciler.reconcile_missing_on_exchange = AsyncMock()
            reconciler.reconcile_missing_locally = Mock()
            reconciler.snapshot_positions = AsyncMock(return_value=initial_positions)
            mock_reconciler_class.return_value = reconciler

            await coordinator.reconcile_state_on_startup()

            assert mock_bot.runtime_state.last_positions == initial_positions
