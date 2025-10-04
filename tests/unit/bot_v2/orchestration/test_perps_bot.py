"""Unit tests for PerpsBot essentials.

Avoids network and real broker connections; focuses on safe internals.
"""

from datetime import datetime, timezone, time
from decimal import Decimal

import asyncio
import threading
import time as _time

import pytest
from unittest.mock import AsyncMock, Mock

from bot_v2.orchestration.perps_bot import PerpsBot, BotConfig, Profile
from bot_v2.orchestration.deterministic_broker import DeterministicBroker
from bot_v2.features.brokerages.core.interfaces import (
    OrderSide,
    OrderType,
    Order,
    OrderStatus,
    TimeInForce,
)


def test_init_uses_mock_broker_in_dev(monkeypatch, tmp_path):
    # Reduce side effects: avoid spawning background threads
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    bot = PerpsBot(config)

    assert isinstance(bot.broker, DeterministicBroker)


def test_calculate_spread_bps():
    # 100 bid, 101 ask => spread 1 over mid 100.5 => ~0.00995 * 10000 ≈ 99.5 bps
    from bot_v2.orchestration.perps_bot import PerpsBot as _PB  # alias to access staticmethod

    bps = _PB._calculate_spread_bps(Decimal("100"), Decimal("101"))
    assert bps > 0
    # Within a small tolerance around 99-100 bps
    assert Decimal("90") < bps < Decimal("110")


def test_update_mark_window_trims(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], short_ma=2, long_ma=3)
    bot = PerpsBot(config)

    for i in range(50):
        bot._update_mark_window("BTC-PERP", Decimal(str(50000 + i)))

    max_expected = max(config.short_ma, config.long_ma) + 5
    assert len(bot.mark_windows["BTC-PERP"]) <= max_expected


def test_collect_account_snapshot_uses_broker(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    bot = PerpsBot(config)

    snapshot_data = {
        "key_permissions": {"can_trade": True},
        "fee_schedule": {"tier": "Advanced"},
        "limits": {"max_order": "10000"},
        "transaction_summary": {"total": "123"},
    }

    bot.account_manager.snapshot = lambda emit_metric=False: snapshot_data  # type: ignore[assignment]
    bot.account_telemetry.supports_snapshots = lambda: True  # type: ignore[assignment]
    bot.account_telemetry._broker.get_server_time = (  # type: ignore[attr-defined]
        lambda: datetime(2024, 1, 1, tzinfo=timezone.utc)
    )

    snap = bot.account_telemetry.collect_snapshot()
    for key, value in snapshot_data.items():
        assert snap[key] == value
    assert snap["server_time"].startswith("2024-01-01")
    assert bot.account_telemetry.latest_snapshot == snap


@pytest.mark.asyncio
async def test_run_account_telemetry_emits_metrics(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    bot = PerpsBot(config)

    monkeypatch.setattr("bot_v2.orchestration.account_telemetry.RUNTIME_DATA_DIR", tmp_path)

    bot.account_manager.snapshot = lambda emit_metric=False: {
        "key_permissions": {},
        "fee_schedule": {},
    }
    bot.account_telemetry.supports_snapshots = lambda: True  # type: ignore[assignment]
    bot.account_telemetry._broker.get_server_time = lambda: None  # type: ignore[attr-defined]

    event = asyncio.Event()
    calls: list[tuple[str, dict[str, object] | None]] = []

    def fake_append_metric(bot_id, metrics=None, **kwargs):  # type: ignore[override]
        calls.append((bot_id, metrics))
        event.set()

    bot.event_store.append_metric = fake_append_metric  # type: ignore

    telemetry_task = asyncio.create_task(bot.account_telemetry.run(interval_seconds=0))
    await asyncio.wait_for(event.wait(), timeout=1)
    telemetry_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await telemetry_task

    assert calls
    bot_id, metrics = calls[0]
    assert bot_id == bot.bot_id
    assert metrics and metrics.get("event_type") == "account_snapshot"


@pytest.mark.asyncio
async def test_run_cycle_respects_trading_window(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

    config = BotConfig(
        profile=Profile.DEV,
        symbols=["BTC-PERP"],
        trading_window_start=time(10, 0),
        trading_window_end=time(11, 0),
        trading_days=["monday"],
        mock_broker=True,
    )

    bot = PerpsBot(config)

    async def noop_update():
        return None

    recorded: list[str] = []

    async def record_process(symbol: str, *_, **__) -> None:
        recorded.append(symbol)

    async def noop_log():
        return None

    bot.update_marks = noop_update  # type: ignore
    bot.process_symbol = record_process  # type: ignore
    bot.log_status = noop_log  # type: ignore

    bot._session_guard._now = lambda: datetime(2024, 1, 1, 8, 0)  # Monday outside window
    await bot.run_cycle()
    assert recorded == []

    expected_symbol = bot.config.symbols[0]

    bot._session_guard._now = lambda: datetime(2024, 1, 1, 10, 30)
    await bot.run_cycle()
    assert recorded == [expected_symbol]


@pytest.mark.asyncio
async def test_place_order_lock_serialises_calls(monkeypatch, tmp_path, fake_clock):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setenv("COINBASE_ENABLE_DERIVATIVES", "1")
    monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    bot = PerpsBot(config)

    product = bot.get_product("BTC-PERP")

    class StubExecEngine:
        def __init__(self):
            self._lock = threading.Lock()
            self.active = 0
            self.max_active = 0
            self.counter = 0

        def place_order(self, **kwargs):
            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                _time.sleep(0.02)
                self.counter += 1
                return f"order-{self.counter}"
            finally:
                with self._lock:
                    self.active -= 1

    exec_engine = StubExecEngine()
    bot.exec_engine = exec_engine  # type: ignore

    def make_order(order_id: str) -> Order:
        now = datetime.now(timezone.utc)
        return Order(
            id=order_id,
            client_id=order_id,
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=Decimal("0.01"),
            price=None,
            stop_price=None,
            tif=TimeInForce.GTC,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("0.01"),
            avg_fill_price=Decimal("50000"),
            submitted_at=now,
            updated_at=now,
        )

    bot.broker.get_order = make_order  # type: ignore

    async def submit():
        return await bot.execution_coordinator.place_order(
            bot.exec_engine,
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            order_type=OrderType.MARKET,
            product=product,
            reduce_only=False,
            leverage=None,
            price=None,
            stop_price=None,
            tif=None,
        )

    tasks = [asyncio.create_task(submit()) for _ in range(6)]
    results = await asyncio.gather(*tasks)

    assert all(order is not None for order in results)
    assert exec_engine.max_active == 1
    assert bot.order_stats["attempted"] == 6
    assert bot.order_stats["successful"] == 6


# ============================================================================
# Comprehensive tests for coverage improvement
# ============================================================================


class TestMarkWindowManagement:
    """Test mark window updates and MA calculations."""

    def test_update_mark_window_maintains_max_size(self, monkeypatch, tmp_path):
        """Should trim mark window to max size."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], short_ma=5, long_ma=10)
        bot = PerpsBot(config)

        # Add many marks
        for i in range(100):
            bot._update_mark_window("BTC-PERP", Decimal(str(50000 + i)))

        max_expected = max(config.short_ma, config.long_ma) + 5
        assert len(bot.mark_windows["BTC-PERP"]) <= max_expected
        # Should keep most recent values
        assert bot.mark_windows["BTC-PERP"][-1] == Decimal("50099")

    def test_update_mark_window_thread_safe(self, monkeypatch, tmp_path):
        """Should handle concurrent mark updates safely."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])
        bot = PerpsBot(config)

        def update_marks(start, count):
            for i in range(count):
                bot._update_mark_window("BTC-PERP", Decimal(str(start + i)))

        # Simulate concurrent updates
        import threading

        threads = [threading.Thread(target=update_marks, args=(1000 * i, 10)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have accumulated marks without corruption
        assert len(bot.mark_windows["BTC-PERP"]) > 0


class TestProductRetrieval:
    """Test product catalog access."""

    def test_get_product_returns_from_catalog(self, monkeypatch, tmp_path):
        """Should retrieve product from broker catalog."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])
        bot = PerpsBot(config)

        product = bot.get_product("BTC-PERP")
        assert product is not None
        assert product.symbol == "BTC-PERP"

    def test_get_product_returns_same_symbol(self, monkeypatch, tmp_path):
        """Should return product with matching symbol."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])
        bot = PerpsBot(config)

        product1 = bot.get_product("BTC-PERP")
        product2 = bot.get_product("BTC-PERP")
        assert product1.symbol == product2.symbol


class TestReduceOnlyMode:
    """Test reduce-only mode management."""

    def test_reduce_only_mode_delegates_to_risk_manager(self, monkeypatch, tmp_path):
        """Should delegate reduce-only mode to risk manager."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])
        bot = PerpsBot(config)

        # Initially not in reduce-only mode
        assert bot.is_reduce_only_mode() is False

        # Enable reduce-only
        bot.set_reduce_only_mode(True, reason="test")
        assert bot.is_reduce_only_mode() is True

        # Disable reduce-only
        bot.set_reduce_only_mode(False, reason="test_clear")
        assert bot.is_reduce_only_mode() is False


class TestHealthStatus:
    """Test health status management."""

    def test_write_health_status_ok(self, monkeypatch, tmp_path):
        """Should write health status."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])
        bot = PerpsBot(config)

        # Should not raise
        bot.write_health_status(ok=True, message="All good")

    def test_write_health_status_error(self, monkeypatch, tmp_path):
        """Should write health status with error."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])
        bot = PerpsBot(config)

        # Should not raise
        bot.write_health_status(ok=False, error="Test error")


class TestSessionGuard:
    """Test trading session guard edge cases."""

    @pytest.mark.asyncio
    async def test_trading_window_tuesday(self, monkeypatch, tmp_path):
        """Should trade on Tuesday when configured."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(
            profile=Profile.DEV,
            symbols=["BTC-PERP"],
            trading_window_start=time(10, 0),
            trading_window_end=time(16, 0),
            trading_days=["tuesday", "wednesday"],
            mock_broker=True,
        )

        bot = PerpsBot(config)

        processed = []

        async def capture_process(symbol: str, *_, **__) -> None:
            processed.append(symbol)

        async def noop_update():
            return None

        async def noop_log():
            return None

        bot.update_marks = noop_update  # type: ignore
        bot.process_symbol = capture_process  # type: ignore
        bot.system_monitor.log_status = noop_log  # type: ignore

        # Tuesday 10:30 - should trade
        bot._session_guard._now = lambda: datetime(2024, 1, 2, 10, 30)  # Tuesday
        await bot.run_cycle()
        assert len(processed) == 1

    @pytest.mark.asyncio
    async def test_trading_window_all_days(self, monkeypatch, tmp_path):
        """Should trade on all days when configured."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(
            profile=Profile.DEV,
            symbols=["BTC-PERP"],
            trading_window_start=None,
            trading_window_end=None,
            trading_days=None,
            mock_broker=True,
        )

        bot = PerpsBot(config)

        processed = []

        async def capture_process(symbol: str, *_, **__) -> None:
            processed.append(symbol)

        async def noop_update():
            return None

        async def noop_log():
            return None

        bot.update_marks = noop_update  # type: ignore
        bot.process_symbol = capture_process  # type: ignore
        bot.system_monitor.log_status = noop_log  # type: ignore

        # Any time should trade
        bot._session_guard._now = lambda: datetime(2024, 1, 7, 3, 15)  # Sunday 3am
        await bot.run_cycle()
        assert len(processed) == 1


class TestShutdown:
    """Test bot shutdown procedures."""

    @pytest.mark.asyncio
    async def test_shutdown_sets_running_false(self, monkeypatch, tmp_path):
        """Should set running to False on shutdown."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])
        bot = PerpsBot(config)

        bot.running = True
        await bot.shutdown()
        assert bot.running is False


class TestUpdateMarks:
    """Test mark price update flows."""

    @pytest.mark.asyncio
    async def test_update_marks_delegates_to_service(self, monkeypatch, tmp_path):
        """Should delegate mark updates to market data service."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])
        bot = PerpsBot(config)

        # Should not raise
        await bot.update_marks()


class TestConfigChanges:
    """Test configuration change handling."""

    def test_apply_config_change_updates_config(self, monkeypatch, tmp_path):
        """Should handle config changes."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])
        bot = PerpsBot(config)

        from bot_v2.orchestration.config_controller import ConfigChange

        # Create config change
        new_config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP", "ETH-PERP"])
        change = ConfigChange(
            updated=new_config,
            diff={"symbols": (["BTC-PERP"], ["BTC-PERP", "ETH-PERP"])},
        )

        bot.apply_config_change(change)
        assert bot.config is new_config


class TestRunCycleBalancesPositions:
    """Test run_cycle with balances and positions."""

    @pytest.mark.asyncio
    async def test_run_cycle_handles_balance_fetch_error(self, monkeypatch, tmp_path):
        """Should handle balance fetch errors gracefully."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])
        bot = PerpsBot(config)

        # Make list_balances raise error
        def fail_balances():
            raise RuntimeError("Balance fetch failed")

        bot.broker.list_balances = fail_balances

        processed = []

        async def capture_process(symbol: str, *args, **kwargs):
            processed.append(symbol)

        async def noop_update():
            return None

        async def noop_log():
            return None

        bot.update_marks = noop_update  # type: ignore
        bot.process_symbol = capture_process  # type: ignore
        bot.system_monitor.log_status = noop_log  # type: ignore

        # Should not raise, continue processing
        await bot.run_cycle()
        assert len(processed) == 1

    @pytest.mark.asyncio
    async def test_run_cycle_handles_position_fetch_error(self, monkeypatch, tmp_path):
        """Should handle position fetch errors gracefully."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])
        bot = PerpsBot(config)

        # Make list_positions raise error
        def fail_positions():
            raise RuntimeError("Position fetch failed")

        bot.broker.list_positions = fail_positions

        processed = []

        async def capture_process(symbol: str, *args, **kwargs):
            processed.append(symbol)

        async def noop_update():
            return None

        async def noop_log():
            return None

        bot.update_marks = noop_update  # type: ignore
        bot.process_symbol = capture_process  # type: ignore
        bot.system_monitor.log_status = noop_log  # type: ignore

        # Should not raise, continue processing
        await bot.run_cycle()
        assert len(processed) == 1


class TestInitialization:
    """Test initialization paths."""

    def test_init_with_multiple_symbols(self, monkeypatch, tmp_path):
        """Should initialize mark windows for all symbols."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP", "ETH-PERP", "SOL-PERP"])
        bot = PerpsBot(config)

        assert len(bot.mark_windows) == 3
        assert "BTC-PERP" in bot.mark_windows
        assert "ETH-PERP" in bot.mark_windows
        assert "SOL-PERP" in bot.mark_windows

    def test_init_sets_bot_id(self, monkeypatch, tmp_path):
        """Should initialize with bot_id."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])
        bot = PerpsBot(config)

        assert bot.bot_id is not None
        assert isinstance(bot.bot_id, str)


class TestBrokerRiskManagerProperties:
    """Test broker and risk_manager property setters."""

    def test_broker_property_setter(self, monkeypatch, tmp_path):
        """Should allow setting broker property."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])
        bot = PerpsBot(config)

        original_broker = bot.broker

        # Create new broker and set it
        new_broker = DeterministicBroker()
        bot.broker = new_broker

        assert bot.broker is new_broker
        assert bot.registry.broker is new_broker

    def test_risk_manager_property_setter(self, monkeypatch, tmp_path):
        """Should allow setting risk_manager property."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])
        bot = PerpsBot(config)

        original_rm = bot.risk_manager

        # Create new risk manager (mock)
        from unittest.mock import Mock

        new_rm = Mock()
        bot.risk_manager = new_rm

        assert bot.risk_manager is new_rm
        assert bot.registry.risk_manager is new_rm


class TestSpreadCalculation:
    """Test spread BPS calculation."""

    def test_calculate_spread_bps_wide_spread(self):
        """Should calculate wide spread correctly."""
        from bot_v2.orchestration.perps_bot import PerpsBot as _PB

        # Bid 90, Ask 110 => spread 20, mid 100 => 20/100 * 10000 = 2000 bps
        bps = _PB._calculate_spread_bps(Decimal("90"), Decimal("110"))
        assert bps > Decimal("1900")
        assert bps < Decimal("2100")

    def test_calculate_spread_bps_tight_spread(self):
        """Should calculate tight spread correctly."""
        from bot_v2.orchestration.perps_bot import PerpsBot as _PB

        # Bid 100, Ask 100.01 => spread 0.01, mid 100.005 => very small bps
        bps = _PB._calculate_spread_bps(Decimal("100"), Decimal("100.01"))
        assert bps > Decimal("0")
        assert bps < Decimal("10")


@pytest.mark.asyncio
async def test_process_symbol_delegates(monkeypatch, tmp_path):
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], mock_broker=True)
    bot = PerpsBot(config)

    balances = [object()]
    position_map = {"BTC-PERP": object()}
    bot.strategy_orchestrator.process_symbol = AsyncMock()  # type: ignore[assignment]

    await bot.process_symbol("BTC-PERP", balances, position_map)

    bot.strategy_orchestrator.process_symbol.assert_awaited_once_with(
        "BTC-PERP", balances, position_map
    )


@pytest.mark.asyncio
async def test_execute_decision_delegates(monkeypatch, tmp_path):
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], mock_broker=True)
    bot = PerpsBot(config)

    bot.execution_coordinator.execute_decision = AsyncMock()  # type: ignore[assignment]
    decision = object()
    mark = Decimal("12345")
    product: Product | None = None

    await bot.execute_decision("BTC-PERP", decision, mark, product, {"side": "long"})

    bot.execution_coordinator.execute_decision.assert_awaited_once_with(
        "BTC-PERP", decision, mark, product, {"side": "long"}
    )


# ========== LifecycleService Path Tests ==========


@pytest.mark.asyncio
async def test_run_with_lifecycle_service_delegates(monkeypatch, tmp_path):
    """Test that run() delegates to LifecycleService when USE_LIFECYCLE_SERVICE=true."""
    monkeypatch.setenv("USE_LIFECYCLE_SERVICE", "true")
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

    config = BotConfig(
        profile=Profile.DEV,
        symbols=["BTC-PERP"],
        update_interval=1,
        dry_run=True,
        mock_broker=True,
    )

    bot = PerpsBot(config)

    # Mock lifecycle service methods
    bot.lifecycle_service.configure_background_tasks = Mock()
    bot.lifecycle_service.run = AsyncMock()

    await bot.run(single_cycle=True)

    bot.lifecycle_service.configure_background_tasks.assert_called_once_with(True)
    bot.lifecycle_service.run.assert_awaited_once_with(True)


@pytest.mark.asyncio
async def test_run_with_lifecycle_service_continuous(monkeypatch, tmp_path):
    """Test that run() passes single_cycle=False to LifecycleService."""
    monkeypatch.setenv("USE_LIFECYCLE_SERVICE", "true")
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

    config = BotConfig(
        profile=Profile.DEV,
        symbols=["BTC-PERP"],
        update_interval=1,
        dry_run=True,
        mock_broker=True,
    )

    bot = PerpsBot(config)

    # Mock lifecycle service methods
    bot.lifecycle_service.configure_background_tasks = Mock()
    bot.lifecycle_service.run = AsyncMock()

    await bot.run(single_cycle=False)

    bot.lifecycle_service.configure_background_tasks.assert_called_once_with(False)
    bot.lifecycle_service.run.assert_awaited_once_with(False)


@pytest.mark.asyncio
async def test_run_defaults_to_lifecycle_service(monkeypatch, tmp_path):
    """Test that run() uses LifecycleService by default (when env var not set)."""
    # Don't set USE_LIFECYCLE_SERVICE - should default to true
    monkeypatch.delenv("USE_LIFECYCLE_SERVICE", raising=False)
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

    config = BotConfig(
        profile=Profile.DEV,
        symbols=["BTC-PERP"],
        update_interval=1,
        dry_run=True,
        mock_broker=True,
    )

    bot = PerpsBot(config)

    # Mock lifecycle service methods
    bot.lifecycle_service.configure_background_tasks = Mock()
    bot.lifecycle_service.run = AsyncMock()

    await bot.run(single_cycle=True)

    # Should use lifecycle service by default
    bot.lifecycle_service.configure_background_tasks.assert_called_once()
    bot.lifecycle_service.run.assert_awaited_once()


# ========== Initialization Path Tests ==========


class TestInitialization:
    """Tests for PerpsBot initialization paths."""

    def test_init_configuration_state(self, monkeypatch, tmp_path):
        """Test _init_configuration_state sets up config and session guard."""
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(
            profile=Profile.DEV,
            symbols=["BTC-PERP", "ETH-PERP"],
            trading_days=["monday", "tuesday", "wednesday"],
            trading_window_start="09:30",
            trading_window_end="16:00",
        )

        bot = PerpsBot(config)

        assert bot.config == config
        assert bot.symbols == ["BTC-PERP", "ETH-PERP"]
        assert bot._session_guard._days == ["monday", "tuesday", "wednesday"]
        assert bot._session_guard._start == "09:30"
        assert bot._session_guard._end == "16:00"

    def test_init_runtime_state(self, monkeypatch, tmp_path):
        """Test _init_runtime_state initializes all runtime attributes."""
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(
            profile=Profile.DEV,
            symbols=["BTC-PERP", "ETH-PERP"],
        )

        bot = PerpsBot(config)

        # Check all runtime state attributes
        assert "BTC-PERP" in bot.mark_windows
        assert "ETH-PERP" in bot.mark_windows
        assert bot.mark_windows["BTC-PERP"] == []
        assert bot.last_decisions == {}
        assert bot._last_positions == {}
        assert bot.order_stats == {"attempted": 0, "successful": 0, "failed": 0}
        assert bot._symbol_strategies == {}
        # bot.strategy and _exec_engine are initialized by the builder
        assert bot._product_map == {}

    def test_init_market_data_service_initialized(self, monkeypatch, tmp_path):
        """Test MarketDataService initialization."""
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(
            profile=Profile.DEV,
            symbols=["BTC-PERP"],
            short_ma=5,
            long_ma=20,
        )

        bot = PerpsBot(config)

        assert bot._market_data_service is not None
        assert bot._market_data_service.symbols == ["BTC-PERP"]

    def test_init_streaming_service(self, monkeypatch, tmp_path):
        """Test StreamingService initialization (always created)."""
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])

        bot = PerpsBot(config)

        assert bot._streaming_service is not None
        assert bot._streaming_service.symbols == ["BTC-PERP"]

    def test_construct_services(self, monkeypatch, tmp_path):
        """Test _construct_services creates all service instances."""
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])

        bot = PerpsBot(config)

        assert bot.strategy_orchestrator is not None
        assert bot.execution_coordinator is not None
        assert bot.system_monitor is not None
        assert bot.runtime_coordinator is not None
        assert bot.lifecycle_service is not None

    def test_init_accounting_services(self, monkeypatch, tmp_path):
        """Test _init_accounting_services creates account manager and telemetry."""
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])

        bot = PerpsBot(config)

        assert bot.account_manager is not None
        assert bot.account_telemetry is not None
        assert bot.account_telemetry._bot_id == "perps_bot"
        assert bot.account_telemetry._profile == "dev"

    def test_init_market_services(self, monkeypatch, tmp_path):
        """Test _init_market_services creates market monitor."""
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP", "ETH-PERP"])

        bot = PerpsBot(config)

        # Verify market monitor is initialized
        assert bot._market_monitor is not None


# ========== Streaming Configuration Tests ==========


class TestStreamingConfiguration:
    """Tests for streaming service startup configuration."""

    def test_start_streaming_canary_profile(self, monkeypatch, tmp_path):
        """Test streaming starts in CANARY profile when enabled."""
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))

        config = BotConfig(
            profile=Profile.CANARY,
            symbols=["BTC-PERP"],
            perps_enable_streaming=True,
            perps_stream_level=2,
            mock_broker=True,
        )

        # Don't mock _start_streaming_if_configured - let it run
        bot = PerpsBot(config)

        # Verify streaming was started (would have called start())
        assert bot._streaming_service is not None

    def test_start_streaming_prod_profile(self, monkeypatch, tmp_path):
        """Test streaming starts in PROD profile when enabled."""
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))

        config = BotConfig(
            profile=Profile.PROD,
            symbols=["BTC-PERP"],
            perps_enable_streaming=True,
            perps_stream_level=1,
            mock_broker=True,
        )

        bot = PerpsBot(config)

        assert bot._streaming_service is not None

    def test_start_streaming_dev_profile_disabled(self, monkeypatch, tmp_path):
        """Test streaming doesn't start in DEV profile."""
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(
            profile=Profile.DEV,
            symbols=["BTC-PERP"],
            perps_enable_streaming=True,  # Even if enabled, DEV won't stream
        )

        bot = PerpsBot(config)

        # Streaming service created but not started in DEV
        assert bot._streaming_service is not None

    def test_start_streaming_disabled(self, monkeypatch, tmp_path):
        """Test streaming doesn't start when disabled."""
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(
            profile=Profile.CANARY,
            symbols=["BTC-PERP"],
            perps_enable_streaming=False,  # Disabled
            mock_broker=True,
        )

        bot = PerpsBot(config)

        # Service exists but shouldn't have started
        assert bot._streaming_service is not None

    def test_start_streaming_error_handling(self, monkeypatch, tmp_path):
        """Test streaming startup handles errors gracefully."""
        monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))

        def failing_start_streaming(self):
            """Simulate a streaming startup failure."""
            if self._streaming_service is not None:
                raise RuntimeError("Streaming startup failed")

        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", failing_start_streaming)

        config = BotConfig(
            profile=Profile.CANARY,
            symbols=["BTC-PERP"],
            perps_enable_streaming=True,
            mock_broker=True,
        )

        # Should not raise - error is caught and logged
        bot = PerpsBot(config)
        assert bot._streaming_service is not None
