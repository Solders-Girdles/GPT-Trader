"""
Tests for ExecutionCoordinator initialization and engine selection.
"""

from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine


class TestExecutionCoordinatorInitialization:
    """Test ExecutionCoordinator initialization and engine selection."""

    def test_initialize_missing_dependencies(self, execution_coordinator, execution_context):
        """Test initialize handles missing broker/risk_manager."""
        execution_context = execution_context.with_updates(broker=None, risk_manager=None)

        result = execution_coordinator.initialize(execution_context)

        assert result == execution_context

    def test_initialize_advanced_engine_selection(self, execution_coordinator, execution_context):
        """Test initialize selects AdvancedExecutionEngine when dynamic sizing enabled."""
        execution_context.risk_manager.config.enable_dynamic_position_sizing = True

        result = execution_coordinator.initialize(execution_context)

        assert isinstance(result.runtime_state.exec_engine, AdvancedExecutionEngine)

    def test_initialize_live_engine_selection(self, execution_coordinator, execution_context):
        """Test initialize selects LiveExecutionEngine by default."""
        result = execution_coordinator.initialize(execution_context)

        # Should be LiveExecutionEngine (not Advanced)
        assert result.runtime_state.exec_engine is not None
        assert not isinstance(result.runtime_state.exec_engine, AdvancedExecutionEngine)

    def test_initialize_impact_estimator_building_failure(
        self, execution_coordinator, execution_context
    ):
        """Test initialize handles impact estimator building failures."""
        execution_context.risk_manager.config.enable_market_impact_guard = True

        with patch("bot_v2.features.live_trade.liquidity_service.LiquidityService") as mock_ls:
            mock_ls.side_effect = Exception("LiquidityService init failed")

            result = execution_coordinator.initialize(execution_context)

            # Should continue with engine initialization despite impact estimator failure
            assert result.runtime_state.exec_engine is not None

    def test_initialize_runtime_settings_loading(self, execution_coordinator, execution_context):
        """Test initialize loads runtime settings."""
        from unittest.mock import patch

        with patch(
            "bot_v2.orchestration.coordinators.execution.load_runtime_settings"
        ) as mock_load:
            mock_settings = Mock()
            mock_settings.raw_env = {"PERPS_COLLATERAL_ASSETS": "USDC,BTC"}
            mock_load.return_value = mock_settings

            updated_context = execution_coordinator.initialize(execution_context)

            mock_load.assert_called_once()
            assert updated_context.runtime_state.exec_engine is not None

    def test_initialize_runtime_settings_none_fallback(
        self, execution_coordinator, execution_context
    ):
        """Test initialize handles None runtime settings."""
        from unittest.mock import patch

        with patch(
            "bot_v2.orchestration.coordinators.execution.load_runtime_settings"
        ) as mock_load:
            mock_load.return_value = None

            updated_context = execution_coordinator.initialize(execution_context)

            # Should still initialize successfully
            assert updated_context.runtime_state.exec_engine is not None

    def test_get_order_reconciler_creates_instance(self, execution_coordinator, execution_context):
        """Test _get_order_reconciler creates OrderReconciler instance."""
        # Add missing stores and event store
        execution_context = execution_context.with_updates(orders_store=Mock(), event_store=Mock())
        execution_coordinator.update_context(execution_context)

        reconciler = execution_coordinator._get_order_reconciler()

        assert reconciler is not None
        # Should cache the instance
        assert execution_coordinator._order_reconciler is reconciler

    def test_get_order_reconciler_missing_dependencies(
        self, execution_coordinator, execution_context
    ):
        """Test _get_order_reconciler handles missing dependencies."""
        execution_context = execution_context.with_updates(broker=None)
        execution_coordinator.update_context(execution_context)

        with pytest.raises(RuntimeError, match="Cannot create OrderReconciler"):
            execution_coordinator._get_order_reconciler()

    def test_reset_order_reconciler(self, execution_coordinator, execution_context):
        """Test reset_order_reconciler clears cached reconciler."""
        execution_coordinator._order_reconciler = Mock()
        execution_coordinator.reset_order_reconciler()

        assert execution_coordinator._order_reconciler is None

    def test_build_impact_estimator_success(self, execution_coordinator, execution_context):
        """Test _build_impact_estimator creates estimator function."""
        from bot_v2.features.brokerages.core.interfaces import OrderSide

        with patch(
            "bot_v2.features.live_trade.liquidity_service.LiquidityService"
        ) as mock_ls_class:
            mock_ls = Mock()
            mock_ls_class.return_value = mock_ls
            mock_ls.analyze_order_book = Mock()
            mock_ls.estimate_market_impact = Mock(return_value=Decimal("0.05"))

            estimator = execution_coordinator._build_impact_estimator(execution_context)

            assert callable(estimator)

            # Test estimator execution
            mock_req = Mock()
            mock_req.symbol = "BTC-PERP"
            mock_req.quantity = Decimal("1.0")
            mock_req.side = OrderSide.BUY

            execution_context.broker.get_quote = Mock(
                return_value=Mock(bid=Decimal("50000"), ask=Decimal("50010"), last=Decimal("50005"))
            )
            execution_context.broker.order_books = {}

            result = estimator(mock_req)

            assert result == Decimal("0.05")
            mock_ls.analyze_order_book.assert_called_once()
            mock_ls.estimate_market_impact.assert_called_once()

    def test_build_impact_estimator_broker_quote_failure(
        self, execution_coordinator, execution_context
    ):
        """Test _build_impact_estimator handles broker quote failures."""
        from bot_v2.features.brokerages.core.interfaces import OrderSide

        with patch(
            "bot_v2.features.live_trade.liquidity_service.LiquidityService"
        ) as mock_ls_class:
            mock_ls = Mock()
            mock_ls_class.return_value = mock_ls
            mock_ls.analyze_order_book = Mock()
            mock_ls.estimate_market_impact = Mock(return_value=Decimal("0.05"))

            estimator = execution_coordinator._build_impact_estimator(execution_context)

            mock_req = Mock()
            mock_req.symbol = "BTC-PERP"
            mock_req.quantity = Decimal("1.0")
            mock_req.side = OrderSide.BUY

            execution_context.broker.get_quote = Mock(side_effect=Exception("Quote failed"))
            execution_context.broker.order_books = {}

            result = estimator(mock_req)

            # Should still return estimate using fallback mid price
            assert result == Decimal("0.05")

    def test_build_impact_estimator_no_seeded_orderbooks(
        self, execution_coordinator, execution_context
    ):
        """Test _build_impact_estimator handles missing seeded orderbooks."""
        from bot_v2.features.brokerages.core.interfaces import OrderSide

        with patch(
            "bot_v2.features.live_trade.liquidity_service.LiquidityService"
        ) as mock_ls_class:
            mock_ls = Mock()
            mock_ls_class.return_value = mock_ls
            mock_ls.analyze_order_book = Mock()
            mock_ls.estimate_market_impact = Mock(return_value=Decimal("0.05"))

            estimator = execution_coordinator._build_impact_estimator(execution_context)

            mock_req = Mock()
            mock_req.symbol = "BTC-PERP"
            mock_req.quantity = Decimal("1.0")
            mock_req.side = OrderSide.BUY

            # No seeded orderbooks on broker
            execution_context.broker.order_books = None
            execution_context.broker.get_quote = Mock(
                return_value=Mock(bid=Decimal("50000"), ask=Decimal("50010"), last=Decimal("50005"))
            )

            result = estimator(mock_req)

            assert result == Decimal("0.05")
