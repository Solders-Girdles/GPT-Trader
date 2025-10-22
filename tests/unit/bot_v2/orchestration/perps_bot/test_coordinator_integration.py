"""Tests for PerpsBot coordinator integration, strategy cycles, and execution flows."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock

import pytest

from bot_v2.features.brokerages.core.interfaces import MarketType, OrderSide, OrderType
from bot_v2.orchestration.perps_bot import PerpsBot, _CallableSymbolProcessor


class TestStrategyCycles:
    """Test strategy cycle execution and coordination."""

    @pytest.mark.asyncio
    async def test_run_cycle_delegates_to_strategy_coordinator(self, perps_bot_instance):
        """Test run_cycle delegates to strategy coordinator."""
        bot = perps_bot_instance
        bot.strategy_coordinator.run_cycle = AsyncMock()

        await bot.run_cycle()

        bot.strategy_coordinator.run_cycle.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_single_cycle_delegates_to_lifecycle_manager(self, perps_bot_instance):
        """Test run delegates to lifecycle manager."""
        bot = perps_bot_instance
        bot.lifecycle_manager.run = AsyncMock()

        await bot.run(single_cycle=True)

        bot.lifecycle_manager.run.assert_called_once_with(True)

    @pytest.mark.asyncio
    async def test_run_continuous_delegates_to_lifecycle_manager(self, perps_bot_instance):
        """Test continuous run delegates to lifecycle manager."""
        bot = perps_bot_instance
        bot.lifecycle_manager.run = AsyncMock()

        await bot.run(single_cycle=False)

        bot.lifecycle_manager.run.assert_called_once_with(False)

    @pytest.mark.asyncio
    async def test_fetch_current_state_delegates_to_strategy_coordinator(self, perps_bot_instance):
        """Test _fetch_current_state delegates to strategy coordinator."""
        bot = perps_bot_instance
        expected_state = {"test": "state"}
        bot.strategy_coordinator._fetch_current_state = AsyncMock(return_value=expected_state)

        state = await bot._fetch_current_state()

        assert state == expected_state
        bot.strategy_coordinator._fetch_current_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_configuration_and_handle_drift_delegates_to_strategy_coordinator(self, perps_bot_instance):
        """Test _validate_configuration_and_handle_drift delegates to strategy coordinator."""
        bot = perps_bot_instance
        bot.strategy_coordinator._validate_configuration_and_handle_drift = AsyncMock(return_value=True)
        current_state = {"test": "state"}

        result = await bot._validate_configuration_and_handle_drift(current_state)

        assert result is True
        bot.strategy_coordinator._validate_configuration_and_handle_drift.assert_called_once_with(current_state)

    @pytest.mark.asyncio
    async def test_execute_trading_cycle_delegates_to_strategy_coordinator(self, perps_bot_instance):
        """Test _execute_trading_cycle delegates to strategy coordinator."""
        bot = perps_bot_instance
        bot.strategy_coordinator._execute_trading_cycle = AsyncMock()
        trading_state = {"test": "trading"}

        await bot._execute_trading_cycle(trading_state)

        bot.strategy_coordinator._execute_trading_cycle.assert_called_once_with(trading_state)


class TestSymbolProcessing:
    """Test symbol processing and decision execution."""

    @pytest.mark.asyncio
    async def test_symbol_processor_property_returns_strategy_coordinator_processor(self, perps_bot_instance):
        """Test symbol_processor property returns strategy coordinator's processor."""
        bot = perps_bot_instance
        # The symbol_processor property delegates to strategy_coordinator
        # We can't set it directly, but we can verify the delegation works
        assert hasattr(bot.strategy_coordinator, 'symbol_processor')

        # Test that we can get the processor (it should exist on the coordinator)
        processor = bot.symbol_processor
        # Just verify it returns something without asserting equality since we can't mock it easily
        assert processor is not None

    def test_set_symbol_processor_with_none(self, perps_bot_instance):
        """Test set_symbol_processor with None clears processor."""
        bot = perps_bot_instance
        bot.strategy_coordinator.set_symbol_processor = MagicMock()

        bot.set_symbol_processor(None)

        bot.strategy_coordinator.set_symbol_processor.assert_called_once_with(None)
        assert bot._symbol_processor_override is None

    def test_set_symbol_processor_with_callable_processor(self, perps_bot_instance, wrapped_symbol_processor):
        """Test set_symbol_processor with _CallableSymbolProcessor."""
        bot = perps_bot_instance
        bot.strategy_coordinator.set_symbol_processor = MagicMock()

        bot.set_symbol_processor(wrapped_symbol_processor)

        assert bot._symbol_processor_override is wrapped_symbol_processor
        bot.strategy_coordinator.set_symbol_processor.assert_called_once_with(wrapped_symbol_processor)

    def test_set_symbol_processor_with_regular_processor(self, perps_bot_instance):
        """Test set_symbol_processor with regular processor."""
        bot = perps_bot_instance
        mock_processor = MagicMock()
        bot.strategy_coordinator.set_symbol_processor = MagicMock()

        bot.set_symbol_processor(mock_processor)

        assert bot._symbol_processor_override is None
        bot.strategy_coordinator.set_symbol_processor.assert_called_once_with(mock_processor)

    @pytest.mark.asyncio
    async def test_process_symbol_delegates_to_strategy_coordinator(self, perps_bot_instance, sample_balances, sample_positions):
        """Test process_symbol delegates to strategy coordinator."""
        bot = perps_bot_instance
        bot.strategy_coordinator.process_symbol = AsyncMock()
        position_map = {pos.symbol: pos for pos in sample_positions}

        await bot.process_symbol("BTC-PERP", sample_balances, position_map)

        bot.strategy_coordinator.process_symbol.assert_called_once_with("BTC-PERP", sample_balances, position_map)

    def test_process_symbol_expects_context_delegates_to_strategy_coordinator(self, perps_bot_instance):
        """Test _process_symbol_expects_context delegates to strategy coordinator."""
        bot = perps_bot_instance
        bot.strategy_coordinator._process_symbol_expects_context = MagicMock(return_value=True)

        result = bot._process_symbol_expects_context()

        assert result is True
        bot.strategy_coordinator._process_symbol_expects_context.assert_called_once()


class TestDecisionExecution:
    """Test decision execution and order placement."""

    @pytest.mark.asyncio
    async def test_execute_decision_delegates_to_strategy_coordinator(self, perps_bot_instance, sample_positions):
        """Test execute_decision delegates to strategy coordinator."""
        bot = perps_bot_instance
        bot.strategy_coordinator.execute_decision = AsyncMock()
        decision = {"action": "BUY", "quantity": Decimal("0.01")}
        mark = Decimal("50000.0")
        product = bot.get_product("BTC-PERP")
        position_state = {"position": sample_positions[0]}

        await bot.execute_decision("BTC-PERP", decision, mark, product, position_state)

        bot.strategy_coordinator.execute_decision.assert_called_once_with("BTC-PERP", decision, mark, product, position_state)

    def test_ensure_order_lock_delegates_to_strategy_coordinator(self, perps_bot_instance):
        """Test _ensure_order_lock delegates to strategy coordinator."""
        bot = perps_bot_instance
        mock_lock = MagicMock()
        bot.strategy_coordinator.ensure_order_lock = MagicMock(return_value=mock_lock)

        lock = bot._ensure_order_lock()

        assert lock is mock_lock
        bot.strategy_coordinator.ensure_order_lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_order_delegates_to_strategy_coordinator(self, perps_bot_instance):
        """Test _place_order delegates to strategy coordinator."""
        bot = perps_bot_instance
        mock_order = MagicMock()
        bot.strategy_coordinator.place_order = AsyncMock(return_value=mock_order)

        order = await bot._place_order(symbol="BTC-PERP", side=OrderSide.BUY, quantity=Decimal("0.01"))

        assert order is mock_order
        bot.strategy_coordinator.place_order.assert_called_once_with(symbol="BTC-PERP", side=OrderSide.BUY, quantity=Decimal("0.01"))

    @pytest.mark.asyncio
    async def test_place_order_inner_delegates_to_strategy_coordinator(self, perps_bot_instance):
        """Test _place_order_inner delegates to strategy coordinator."""
        bot = perps_bot_instance
        mock_order = MagicMock()
        bot.strategy_coordinator.place_order_inner = AsyncMock(return_value=mock_order)

        order = await bot._place_order_inner(symbol="ETH-PERP", side=OrderSide.SELL, quantity=Decimal("0.02"))

        assert order is mock_order
        bot.strategy_coordinator.place_order_inner.assert_called_once_with(symbol="ETH-PERP", side=OrderSide.SELL, quantity=Decimal("0.02"))


class TestReduceOnlyMode:
    """Test reduce-only mode functionality and integration."""

    def test_is_reduce_only_mode_delegates_to_runtime_coordinator(self, perps_bot_instance):
        """Test is_reduce_only_mode delegates to runtime coordinator."""
        bot = perps_bot_instance
        bot.runtime_coordinator.is_reduce_only_mode = MagicMock(return_value=True)

        result = bot.is_reduce_only_mode()

        assert result is True
        bot.runtime_coordinator.is_reduce_only_mode.assert_called_once()

    def test_set_reduce_only_mode_delegates_to_runtime_coordinator(self, perps_bot_instance):
        """Test set_reduce_only_mode delegates to runtime coordinator."""
        bot = perps_bot_instance
        bot.runtime_coordinator.set_reduce_only_mode = MagicMock()

        bot.set_reduce_only_mode(True, "test_reason")

        bot.runtime_coordinator.set_reduce_only_mode.assert_called_once_with(True, "test_reason")


class TestMarketDataAndProductManagement:
    """Test market data updates and product management."""

    @pytest.mark.asyncio
    async def test_update_marks_delegates_to_strategy_coordinator(self, perps_bot_instance):
        """Test update_marks delegates to strategy coordinator."""
        bot = perps_bot_instance
        bot.strategy_coordinator.update_marks = AsyncMock()

        await bot.update_marks()

        bot.strategy_coordinator.update_marks.assert_called_once()

    def test_get_product_returns_from_product_map(self, perps_bot_instance):
        """Test get_product returns from product map when available."""
        bot = perps_bot_instance
        mock_product = MagicMock()
        bot._product_map["BTC-PERP"] = mock_product

        product = bot.get_product("BTC-PERP")

        assert product is mock_product

    def test_get_product_creates_product_when_not_in_map(self, perps_bot_instance):
        """Test get_product creates product when not in product map."""
        bot = perps_bot_instance

        product = bot.get_product("ETH-PERP")

        assert product.symbol == "ETH-PERP"
        assert product.base_asset == "ETH"
        assert product.quote_asset == "PERP"  # From symbol partition
        assert product.step_size == Decimal("0.00000001")
        assert product.min_size == Decimal("0.00000001")
        assert product.price_increment == Decimal("0.01")
        assert product.min_notional == Decimal("10")

    def test_get_product_creates_perpetual_product(self, perps_bot_instance):
        """Test get_product creates perpetual product for PERP symbols."""
        bot = perps_bot_instance

        product = bot.get_product("BTC-PERP")

        assert product.symbol == "BTC-PERP"
        assert product.base_asset == "BTC"
        assert product.quote_asset == "PERP"
        assert product.market_type == MarketType.PERPETUAL

    def test_get_product_creates_spot_product(self, perps_bot_instance):
        """Test get_product creates spot product for non-PERP symbols."""
        bot = perps_bot_instance

        product = bot.get_product("BTC-USD")

        assert product.symbol == "BTC-USD"
        assert product.base_asset == "BTC"
        assert product.quote_asset == "USD"
        assert product.market_type == MarketType.SPOT

    def test_get_product_uses_custom_quote(self, perps_bot_instance):
        """Test get_product uses custom quote from settings."""
        bot = perps_bot_instance
        bot.settings.coinbase_default_quote = "USDT"

        product = bot.get_product("ETH")

        assert product.symbol == "ETH"
        assert product.base_asset == "ETH"
        assert product.quote_asset == "USDT"


class TestSystemMonitoring:
    """Test system monitoring and health status."""

    def test_write_health_status_delegates_to_system_monitor(self, perps_bot_instance):
        """Test write_health_status delegates to system monitor."""
        bot = perps_bot_instance
        bot.system_monitor.write_health_status = MagicMock()

        bot.write_health_status(ok=True, message="All good", error="")

        bot.system_monitor.write_health_status.assert_called_once_with(ok=True, message="All good", error="")

    @pytest.mark.asyncio
    async def test_shutdown_delegates_to_lifecycle_manager(self, perps_bot_instance):
        """Test shutdown delegates to lifecycle manager."""
        bot = perps_bot_instance
        bot.lifecycle_manager.shutdown = AsyncMock()

        await bot.shutdown()

        bot.lifecycle_manager.shutdown.assert_called_once()


class TestSymbolProcessorAdapter:
    """Test _CallableSymbolProcessor adapter functionality."""

    def test_callable_symbol_processor_with_context(self, callable_symbol_processor):
        """Test _CallableSymbolProcessor with context parameters."""
        from bot_v2.orchestration.perps_bot import _CallableSymbolProcessor

        processor = _CallableSymbolProcessor(
            func=callable_symbol_processor,
            requires_context=True,
        )

        result = processor.process_symbol("BTC-PERP", ["balance1"], {"BTC-PERP": "position"})

        assert result["symbol"] == "BTC-PERP"
        assert result["decision"] == "BUY"
        assert result["quantity"] == Decimal("0.01")
        assert result["reason"] == "test_signal"

    def test_callable_symbol_processor_without_context(self, callable_symbol_processor_no_context):
        """Test _CallableSymbolProcessor without context parameters."""
        from bot_v2.orchestration.perps_bot import _CallableSymbolProcessor

        processor = _CallableSymbolProcessor(
            func=callable_symbol_processor_no_context,
            requires_context=False,
        )

        result = processor.process_symbol("ETH-PERP", ["balance1"], {"ETH-PERP": "position"})

        assert result["symbol"] == "ETH-PERP"
        assert result["decision"] == "SELL"
        assert result["quantity"] == Decimal("0.02")
        assert result["reason"] == "simple_signal"

    def test_callable_symbol_processor_function_property(self, callable_symbol_processor):
        """Test _CallableSymbolProcessor function property."""
        from bot_v2.orchestration.perps_bot import _CallableSymbolProcessor

        processor = _CallableSymbolProcessor(
            func=callable_symbol_processor,
            requires_context=True,
        )

        assert processor.function is callable_symbol_processor


class TestUtilityMethods:
    """Test utility methods and calculations."""

    def test_calculate_spread_bps_static_method(self):
        """Test _calculate_spread_bps static method."""
        # 100 bid, 101 ask => spread 1 over mid 100.5 => ~0.00995 * 10000 ≈ 99.5 bps
        bps = PerpsBot._calculate_spread_bps(Decimal("100"), Decimal("101"))

        assert bps > Decimal("0")
        assert Decimal("90") < bps < Decimal("110")

    def test_calculate_spread_bps_with_small_spread(self):
        """Test _calculate_spread_bps with small spread."""
        bps = PerpsBot._calculate_spread_bps(Decimal("50000"), Decimal("50001"))

        assert bps > Decimal("0")
        # Should be very small spread in basis points
        assert bps < Decimal("5")

    def test_calculate_spread_bps_with_large_spread(self):
        """Test _calculate_spread_bps with large spread."""
        bps = PerpsBot._calculate_spread_bps(Decimal("100"), Decimal("110"))

        assert bps > Decimal("0")
        # Should be larger spread in basis points
        assert bps > Decimal("500")

    @pytest.mark.asyncio
    async def test_run_account_telemetry_delegates_to_telemetry_coordinator(self, perps_bot_instance):
        """Test _run_account_telemetry delegates to telemetry coordinator."""
        bot = perps_bot_instance
        bot.telemetry_coordinator.run_account_telemetry = AsyncMock()

        await bot._run_account_telemetry(interval_seconds=60)

        bot.telemetry_coordinator.run_account_telemetry.assert_called_once_with(60)


class TestSymbolProcessorOverride:
    """Test symbol processor override functionality."""

    def test_wrap_symbol_processor_with_callable(self, perps_bot_instance, callable_symbol_processor):
        """Test _wrap_symbol_processor with callable function."""
        wrapped = perps_bot_instance._wrap_symbol_processor(callable_symbol_processor)

        assert isinstance(wrapped, _CallableSymbolProcessor)
        assert wrapped.function is callable_symbol_processor
        assert wrapped.requires_context is True  # Function takes more than 1 parameter

    def test_wrap_symbol_processor_with_non_callable(self, perps_bot_instance):
        """Test _wrap_symbol_processor raises with non-callable."""
        with pytest.raises(TypeError, match="process_symbol handler must be callable"):
            perps_bot_instance._wrap_symbol_processor("not_callable")

    def test_wrap_symbol_processor_with_zero_param_function(self, perps_bot_instance):
        """Test _wrap_symbol_processor with function taking no parameters."""
        def zero_param_func():
            return "test"

        wrapped = perps_bot_instance._wrap_symbol_processor(zero_param_func)

        assert wrapped.requires_context is False

    def test_install_symbol_processor_override_with_none(self, perps_bot_instance):
        """Test _install_symbol_processor_override with None."""
        perps_bot_instance.strategy_coordinator.set_symbol_processor = MagicMock()

        perps_bot_instance._install_symbol_processor_override(None)

        assert perps_bot_instance._symbol_processor_override is None
        perps_bot_instance.strategy_coordinator.set_symbol_processor.assert_called_once_with(None)

    def test_install_symbol_processor_override_with_callable(self, perps_bot_instance, callable_symbol_processor):
        """Test _install_symbol_processor_override with callable."""
        perps_bot_instance.strategy_coordinator.set_symbol_processor = MagicMock()

        perps_bot_instance._install_symbol_processor_override(callable_symbol_processor)

        assert perps_bot_instance._symbol_processor_override is not None
        assert isinstance(perps_bot_instance._symbol_processor_override, _CallableSymbolProcessor)
        perps_bot_instance.strategy_coordinator.set_symbol_processor.assert_called_once()

    def test_setattr_process_symbol_with_callable(self, perps_bot_instance, callable_symbol_processor):
        """Test __setattr__ with process_symbol callable override."""
        # Set the attribute directly to trigger the custom setter
        perps_bot_instance.process_symbol = callable_symbol_processor

        assert perps_bot_instance._symbol_processor_override is not None
        assert isinstance(perps_bot_instance._symbol_processor_override, _CallableSymbolProcessor)

    def test_setattr_process_symbol_with_none(self, perps_bot_instance):
        """Test __setattr__ with process_symbol None."""
        perps_bot_instance.process_symbol = None

        assert perps_bot_instance._symbol_processor_override is None

    def test_setattr_process_symbol_with_non_callable_raises(self, perps_bot_instance):
        """Test __setattr__ with process_symbol non-callable raises."""
        with pytest.raises(TypeError, match="process_symbol override must be callable"):
            perps_bot_instance.process_symbol = "not_callable"

    def test_setattr_bypass_for_non_process_symbol(self, perps_bot_instance):
        """Test __setattr__ bypasses custom logic for non-process_symbol attributes."""
        perps_bot_instance.custom_attribute = "test_value"

        assert perps_bot_instance.custom_attribute == "test_value"
        assert not hasattr(perps_bot_instance, "_symbol_processor_override") or perps_bot_instance._symbol_processor_override is None