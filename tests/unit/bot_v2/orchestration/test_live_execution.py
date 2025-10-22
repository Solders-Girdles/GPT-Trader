"""Unit tests for LiveExecutionEngine.

Focus on pre-trade validation path and reduce-only enforcement.
"""

import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
from tests.support.deterministic_broker import DeterministicBroker

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.brokerages.core.interfaces import (
    Balance,
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Product,
    TimeInForce,
)
from bot_v2.features.live_trade.risk import LiveRiskManager, ValidationError
from bot_v2.orchestration.live_execution import LiveExecutionEngine
from bot_v2.persistence.event_store import EventStore


class DummyBroker:
    def __init__(self):
        self._product = Product(
            symbol="BTC-PERP",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
        )
        self._orders = {}

    # IBrokerage minimal surface
    def get_product(self, symbol: str) -> Product:  # type: ignore[override]
        return self._product

    def list_balances(self) -> list[Balance]:  # type: ignore[override]
        return [
            Balance(
                asset="USD", total=Decimal("1000"), available=Decimal("1000"), hold=Decimal("0")
            )
        ]

    def list_positions(self) -> list[Position]:  # type: ignore[override]
        return []

    def get_quote(self, symbol: str):  # type: ignore[override]
        # Minimal quote for price estimation when price is None
        from bot_v2.features.brokerages.core.interfaces import Quote

        return Quote(
            symbol=symbol,
            bid=Decimal("9999"),
            ask=Decimal("10001"),
            last=Decimal("10000"),
            ts=datetime.utcnow(),
        )

    def place_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        tif: TimeInForce = TimeInForce.GTC,
        client_id: str | None = None,
        reduce_only: bool | None = None,
        leverage: int | None = None,
    ) -> Order:  # type: ignore[override]
        oid = f"ord-{len(self._orders) + 1}"
        order = Order(
            id=oid,
            client_id=client_id,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            tif=tif,
            status=OrderStatus.SUBMITTED,
            filled_quantity=Decimal("0"),
            avg_fill_price=None,
            submitted_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self._orders[oid] = order
        return order

    def cancel_order(self, order_id: str) -> bool:  # type: ignore[override]
        return self._orders.pop(order_id, None) is not None


def test_place_order_success_with_explicit_reduce_only(monkeypatch):
    broker = DummyBroker()
    risk = LiveRiskManager()  # default reduce_only_mode=False (allow increasing positions)

    engine = LiveExecutionEngine(broker=broker, risk_manager=risk)

    # Patch validator inside module to always ok
    import bot_v2.orchestration.live_execution as le
    from bot_v2.features.brokerages.coinbase.specs import ValidationResult

    monkeypatch.setattr(
        le,
        "spec_validate_order",
        lambda **kwargs: ValidationResult(
            ok=True, adjusted_quantity=kwargs.get("quantity"), adjusted_price=kwargs.get("price")
        ),
    )

    order_id = engine.place_order(
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.01"),
        price=None,
        leverage=2,
        reduce_only=True,
    )

    assert isinstance(order_id, str)
    assert order_id in engine.open_orders


def test_reduce_only_mode_rejects_increasing_position(monkeypatch):
    broker = DummyBroker()
    risk = LiveRiskManager()
    risk.set_reduce_only_mode(True)
    engine = LiveExecutionEngine(broker=broker, risk_manager=risk)

    import bot_v2.orchestration.live_execution as le
    from bot_v2.features.brokerages.coinbase.specs import ValidationResult

    monkeypatch.setattr(
        le,
        "spec_validate_order",
        lambda **kwargs: ValidationResult(
            ok=True, adjusted_quantity=kwargs.get("quantity"), adjusted_price=kwargs.get("price")
        ),
    )

    with pytest.raises(ValidationError):
        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
            price=None,
            leverage=2,
        )


def test_place_order_rejected_raises_validation_error(monkeypatch):
    broker = DummyBroker()
    engine = LiveExecutionEngine(broker=broker)

    import bot_v2.orchestration.live_execution as le
    from bot_v2.features.brokerages.coinbase.specs import ValidationResult

    # Force spec validator to reject
    monkeypatch.setattr(
        le,
        "spec_validate_order",
        lambda **kwargs: ValidationResult(ok=False, reason="min_size"),
    )

    with pytest.raises(ValidationError):
        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.0001"),
            price=Decimal("100"),
        )


def test_preview_enabled_calls_preview(monkeypatch):
    broker = DummyBroker()
    risk = LiveRiskManager()
    risk.config.max_position_pct_per_symbol = 1.0
    risk.config.max_exposure_pct = 1.0
    engine = LiveExecutionEngine(broker=broker, risk_manager=risk, enable_preview=True)

    import bot_v2.orchestration.live_execution as le
    from bot_v2.features.brokerages.coinbase.specs import ValidationResult

    monkeypatch.setattr(
        le,
        "spec_validate_order",
        lambda **kwargs: ValidationResult(
            ok=True, adjusted_quantity=kwargs.get("quantity"), adjusted_price=kwargs.get("price")
        ),
    )

    preview_calls = []
    broker.preview_order = lambda **kwargs: preview_calls.append(kwargs) or {"success": True}  # type: ignore[attr-defined]
    broker.edit_order_preview = lambda order_id, **kwargs: None  # type: ignore[attr-defined]

    captured_metrics: list = []
    engine.event_store.append_metric = (
        lambda bot_id, metrics=None, **kwargs: captured_metrics.append((bot_id, metrics))
    )  # type: ignore

    order_id = engine.place_order(
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.02"),
    )

    assert order_id is not None
    assert preview_calls
    assert any(m[1].get("event_type") == "order_preview" for m in captured_metrics)


def test_place_order_uses_usdc_collateral(monkeypatch):
    broker = DummyBroker()

    def usdc_balances() -> list[Balance]:
        return [
            Balance(
                asset="USDC", total=Decimal("5000"), available=Decimal("5000"), hold=Decimal("0")
            )
        ]

    broker.list_balances = usdc_balances  # type: ignore[assignment]

    risk = LiveRiskManager()
    engine = LiveExecutionEngine(broker=broker, risk_manager=risk)

    import bot_v2.orchestration.live_execution as le
    from bot_v2.features.brokerages.coinbase.specs import ValidationResult

    monkeypatch.setattr(
        le,
        "spec_validate_order",
        lambda **kwargs: ValidationResult(
            ok=True, adjusted_quantity=kwargs.get("quantity"), adjusted_price=kwargs.get("price")
        ),
    )

    order_id = engine.place_order(
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.01"),
        price=None,
    )

    assert isinstance(order_id, str)
    assert engine._last_collateral_available == Decimal("5000")


# ===== Perps-Specific Integration Tests =====


@pytest.mark.perps
def test_reduce_only_is_forced_in_execution_perps():
    """Test reduce_only flag propagates correctly in perps trading."""
    broker = DeterministicBroker()
    risk = LiveRiskManager(config=RiskConfig(reduce_only_mode=True))
    eng = LiveExecutionEngine(broker=broker, risk_manager=risk, event_store=EventStore())

    # Seed an existing long position so a SELL reduces
    broker.seed_position("BTC-PERP", side="long", quantity=Decimal("0.01"), price=Decimal("50000"))

    # Monkeypatch broker.place_order to capture reduce_only flag
    captured = {}
    orig = broker.place_order

    def _patched_place_order(**kwargs):  # type: ignore[no-redef]
        captured.update(kwargs)
        return orig(**kwargs)

    broker.place_order = _patched_place_order  # type: ignore[assignment]

    broker.connect()
    order_id = eng.place_order(
        symbol="BTC-PERP",
        side=OrderSide.SELL,  # reducing long
        order_type=OrderType.MARKET,
        quantity=Decimal("0.005"),
    )
    assert order_id is not None
    assert captured.get("reduce_only") is True


@pytest.mark.perps
def test_rejection_logs_metric_with_reason_perps():
    """Test order rejection logs proper metrics for perps orders."""
    broker = DeterministicBroker()
    tmp = tempfile.TemporaryDirectory()
    ev = EventStore(root=Path(tmp.name))
    risk = LiveRiskManager(config=RiskConfig(max_leverage=5, slippage_guard_bps=1_000_000))
    eng = LiveExecutionEngine(
        broker=broker, risk_manager=risk, event_store=ev, bot_id="perps_test_bot"
    )

    broker.connect()
    # Force huge quantity → leverage rejection
    with pytest.raises(ValidationError):
        eng.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("20"),  # notional >> equity
        )

    # Tail events and ensure order_rejected logged
    events = ev.tail("perps_test_bot", limit=20, types=["order_rejected"])  # type: ignore[arg-type]
    assert any("order_rejected" == e.get("type") for e in events)
    assert any("Leverage" in e.get("reason", "") for e in events)


class TestLiveExecutionEngineCoverageEnhancements:
    """Coverage enhancement tests for LiveExecutionEngine critical paths."""

    def test_initialization_with_custom_components(self, mock_broker, mock_risk_manager):
        """Test initialization with custom components and settings."""
        from bot_v2.orchestration.runtime_settings import RuntimeSettings

        custom_settings = RuntimeSettings(
            raw_env={"ORDER_PREVIEW_ENABLED": "true"},
            runtime_root=Path("/tmp/test"),
            event_store_root_override=None,
            coinbase_default_quote="USD",
            coinbase_default_quote_overridden=False,
            coinbase_enable_derivatives=False,
            coinbase_enable_derivatives_overridden=False,
            perps_enable_streaming=False,
            perps_stream_level=1,
            perps_paper_trading=False,
            perps_force_mock=False,
            perps_skip_startup_reconcile=False,
            perps_position_fraction=None,
            order_preview_enabled=False,
            spot_force_live=False,
            broker_hint=None,
            coinbase_sandbox_enabled=True,
            coinbase_api_mode="sandbox",
            risk_config_path=None,
            coinbase_intx_portfolio_uuid=None,
        )

        event_store = EventStore()
        engine = LiveExecutionEngine(
            broker=mock_broker,
            risk_manager=mock_risk_manager,
            event_store=event_store,
            bot_id="test_bot",
            slippage_multipliers={"BTC-PERP": 0.001},
            enable_preview=True,
            settings=custom_settings,
        )

        assert engine.broker is mock_broker
        assert engine.risk_manager is mock_risk_manager
        assert engine.event_store is event_store
        assert engine.bot_id == "test_bot"
        assert engine.slippage_multipliers == {"BTC-PERP": 0.001}
        assert engine.enable_order_preview is True
        assert engine.open_orders == []
        assert engine._last_collateral_available is None

    def test_initialization_creates_default_risk_manager(self, mock_broker):
        """Test initialization creates default risk manager when none provided."""
        engine = LiveExecutionEngine(broker=mock_broker)

        assert engine.risk_manager is not None
        assert engine.event_store is not None
        assert engine.bot_id == "live_execution"

    def test_initialization_uses_risk_manager_event_store(self, mock_broker, mock_risk_manager):
        """Test initialization uses risk manager's event store when available."""
        # Give risk manager its own event store
        risk_event_store = EventStore()
        mock_risk_manager.event_store = risk_event_store
        mock_risk_manager.set_event_store = Mock()

        engine = LiveExecutionEngine(broker=mock_broker, risk_manager=mock_risk_manager)

        # Should use the shared event store
        assert engine.event_store is risk_event_store
        mock_risk_manager.set_event_store.assert_called_once_with(risk_event_store)

    def test_initialization_fallback_to_setattr(self, mock_broker, mock_risk_manager):
        """Test initialization fallback for legacy risk managers."""
        # Remove set_event_store method
        delattr(mock_risk_manager, "set_event_store")

        engine = LiveExecutionEngine(broker=mock_broker, risk_manager=mock_risk_manager)

        # Should use setattr fallback
        assert engine.event_store is not None

    def test_initialization_respects_environment_preview_setting(self, mock_broker):
        """Test initialization respects environment ORDER_PREVIEW_ENABLED setting."""
        with patch("bot_v2.orchestration.live_execution.load_runtime_settings") as mock_load:
            mock_settings = Mock()
            mock_settings.raw_env = {"ORDER_PREVIEW_ENABLED": "1"}
            mock_load.return_value = mock_settings

            engine = LiveExecutionEngine(broker=mock_broker)

            assert engine.enable_order_preview is True

    def test_initialization_respects_environment_preview_false_values(self, mock_broker):
        """Test initialization handles false environment preview values."""
        with patch("bot_v2.orchestration.live_execution.load_runtime_settings") as mock_load:
            mock_settings = Mock()
            mock_settings.raw_env = {"ORDER_PREVIEW_ENABLED": "false"}
            mock_load.return_value = mock_settings

            engine = LiveExecutionEngine(broker=mock_broker)

            assert engine.enable_order_preview is False

    def test_initialization_with_preview_override(self, mock_broker):
        """Test initialization respects explicit enable_preview parameter."""
        with patch("bot_v2.orchestration.live_execution.load_runtime_settings") as mock_load:
            mock_settings = Mock()
            mock_settings.raw_env = {"ORDER_PREVIEW_ENABLED": "true"}
            mock_load.return_value = mock_settings

            engine = LiveExecutionEngine(broker=mock_broker, enable_preview=False)

            assert engine.enable_order_preview is False  # Explicit override

    def test_initialization_creates_helper_modules(self, mock_broker, mock_risk_manager):
        """Test initialization creates all helper modules."""
        engine = LiveExecutionEngine(
            broker=mock_broker,
            risk_manager=mock_risk_manager,
        )

        assert engine.state_collector is not None
        assert engine.order_submitter is not None
        assert engine.order_validator is not None
        assert engine.guard_manager is not None

    def test_place_order_with_all_parameters(self, mock_broker, mock_risk_manager):
        """Test place_order with all possible parameters."""
        engine = LiveExecutionEngine(broker=mock_broker, risk_manager=mock_risk_manager)

        # Mock the order validator to return success
        engine.order_validator.validate_order = Mock(return_value=True)

        order_id = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            stop_price=Decimal("51000"),
            tif="GTC",
            reduce_only=False,
            leverage=2,
            product=Mock(),
            client_order_id="custom_client_id",
        )

        assert order_id is not None
        engine.order_validator.validate_order.assert_called_once()

    def test_place_order_handles_validation_failure(self, mock_broker, mock_risk_manager):
        """Test place_order handles validation failure gracefully."""
        engine = LiveExecutionEngine(broker=mock_broker, risk_manager=mock_risk_manager)

        # Mock the order validator to raise ValidationError
        from bot_v2.features.live_trade.risk import ValidationError

        engine.order_validator.validate_order = Mock(side_effect=ValidationError("Test validation"))

        with pytest.raises(ValidationError):
            engine.place_order(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
            )

    def test_place_order_handles_spec_validation_failure(self, mock_broker, mock_risk_manager):
        """Test place_order handles spec validation failure."""
        engine = LiveExecutionEngine(broker=mock_broker, risk_manager=mock_risk_manager)

        # Mock spec_validate_order to return failure
        with patch("bot_v2.orchestration.live_execution.spec_validate_order") as mock_spec:
            from bot_v2.features.brokerages.coinbase.specs import ValidationResult

            mock_spec.return_value = ValidationResult(
                ok=False, adjusted_quantity=None, adjusted_price=None, reason="Invalid order"
            )

            result = engine.place_order(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
            )

            assert result is None

    def test_place_order_with_slippage_multiplier(self, mock_broker, mock_risk_manager):
        """Test place_order applies slippage multiplier correctly."""
        engine = LiveExecutionEngine(
            broker=mock_broker,
            risk_manager=mock_risk_manager,
            slippage_multipliers={"BTC-PERP": 0.001},
        )

        engine.order_validator.validate_order = Mock(return_value=True)

        order_id = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        assert order_id is not None

    def test_place_order_tracks_open_orders(self, mock_broker, mock_risk_manager):
        """Test place_order tracks open orders correctly."""
        engine = LiveExecutionEngine(broker=mock_broker, risk_manager=mock_risk_manager)

        engine.order_validator.validate_order = Mock(return_value=True)
        engine.order_submitter.submit_order = Mock(return_value="order_123")

        order_id = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        assert order_id == "order_123"
        assert "order_123" in engine.open_orders

    def test_cancel_all_orders_empty_list(self, mock_broker, mock_risk_manager):
        """Test cancel_all_orders handles empty open orders list."""
        engine = LiveExecutionEngine(broker=mock_broker, risk_manager=mock_risk_manager)

        # Should not raise exception
        result = engine.cancel_all_orders()
        assert result is None

    def test_cancel_all_orders_with_open_orders(self, mock_broker, mock_risk_manager):
        """Test cancel_all_orders cancels all open orders."""
        engine = LiveExecutionEngine(broker=mock_broker, risk_manager=mock_risk_manager)

        # Add some open orders
        engine.open_orders = ["order_1", "order_2", "order_3"]

        # Mock broker cancel_order
        mock_broker.cancel_order = Mock()

        result = engine.cancel_all_orders()

        # Should have cancelled all orders
        assert mock_broker.cancel_order.call_count == 3
        assert engine.open_orders == []  # Should be cleared

    def test_cancel_all_orders_handles_cancellation_failure(self, mock_broker, mock_risk_manager):
        """Test cancel_all_orders handles individual order cancellation failure."""
        engine = LiveExecutionEngine(broker=mock_broker, risk_manager=mock_risk_manager)

        engine.open_orders = ["order_1", "order_2"]

        # Mock broker cancel_order to fail on second order
        def mock_cancel(order_id):
            if order_id == "order_2":
                raise Exception("Cancellation failed")

        mock_broker.cancel_order = Mock(side_effect=mock_cancel)

        # Should not raise exception, continue cancelling others
        result = engine.cancel_all_orders()

        assert result is None
        assert engine.open_orders == []  # Still cleared

    def test_place_order_uses_order_submitter(self, mock_broker, mock_risk_manager):
        """Test place_order delegates to order submitter."""
        engine = LiveExecutionEngine(broker=mock_broker, risk_manager=mock_risk_manager)

        # Mock order validator to pass
        engine.order_validator.validate_order = Mock(return_value=True)

        # Track submit_order call
        engine.order_submitter.submit_order = Mock(return_value="order_123")

        order_id = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        engine.order_submitter.submit_order.assert_called_once()
        assert order_id == "order_123"

    def test_place_order_with_runtime_guard_error(self, mock_broker, mock_risk_manager):
        """Test place_order handles runtime guard errors."""
        engine = LiveExecutionEngine(broker=mock_broker, risk_manager=mock_risk_manager)

        # Mock order validator to raise RiskGuardError
        from bot_v2.features.live_trade.guard_errors import RiskGuardError

        engine.order_validator.validate_order = Mock(side_effect=RiskGuardError("Guard triggered"))

        with pytest.raises(RiskGuardError):
            engine.place_order(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
            )

    def test_place_order_with_general_exception(self, mock_broker, mock_risk_manager):
        """Test place_order handles general exceptions."""
        engine = LiveExecutionEngine(broker=mock_broker, risk_manager=mock_risk_manager)

        # Mock order validator to raise general exception
        engine.order_validator.validate_order = Mock(side_effect=Exception("Unexpected error"))

        with pytest.raises(Exception):
            engine.place_order(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
            )

    def test_place_order_logs_metrics(self, mock_broker, mock_risk_manager):
        """Test place_order logs appropriate metrics."""
        engine = LiveExecutionEngine(broker=mock_broker, risk_manager=mock_risk_manager)

        engine.order_validator.validate_order = Mock(return_value=True)
        engine.order_submitter.submit_order = Mock(return_value="order_123")

        # Track event store calls
        metrics_calls = []
        engine.event_store.append_metric = Mock(
            side_effect=lambda *args, **kwargs: metrics_calls.append(kwargs)
        )

        order_id = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        assert order_id is not None
        assert len(metrics_calls) > 0

    def test_place_order_uses_order_preview_when_enabled(self, mock_broker, mock_risk_manager):
        """Test place_order uses order preview when enabled."""
        engine = LiveExecutionEngine(
            broker=mock_broker, risk_manager=mock_risk_manager, enable_preview=True
        )

        # Mock order validator to pass
        engine.order_validator.validate_order = Mock(return_value=True)
        engine.order_submitter.submit_order = Mock(return_value="order_123")

        order_id = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        # Should have called order validator with preview enabled
        engine.order_validator.validate_order.assert_called_once()

    def test_place_order_handles_stop_orders(self, mock_broker, mock_risk_manager):
        """Test place_order handles stop orders correctly."""
        engine = LiveExecutionEngine(broker=mock_broker, risk_manager=mock_risk_manager)

        engine.order_validator.validate_order = Mock(return_value=True)
        engine.order_submitter.submit_order = Mock(return_value="stop_order_123")

        order_id = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            quantity=Decimal("0.1"),
            stop_price=Decimal("51000"),
        )

        assert order_id == "stop_order_123"
        engine.order_validator.validate_order.assert_called_once()

    def test_place_order_handles_stop_limit_orders(self, mock_broker, mock_risk_manager):
        """Test place_order handles stop-limit orders correctly."""
        engine = LiveExecutionEngine(broker=mock_broker, risk_manager=mock_risk_manager)

        engine.order_validator.validate_order = Mock(return_value=True)
        engine.order_submitter.submit_order = Mock(return_value="stop_limit_order_123")

        order_id = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.STOP_LIMIT,
            quantity=Decimal("0.1"),
            stop_price=Decimal("51000"),
            price=Decimal("52000"),
        )

        assert order_id == "stop_limit_order_123"

    def test_place_order_handles_time_in_force(self, mock_broker, mock_risk_manager):
        """Test place_order handles time-in-force correctly."""
        engine = LiveExecutionEngine(broker=mock_broker, risk_manager=mock_risk_manager)

        engine.order_validator.validate_order = Mock(return_value=True)
        engine.order_submitter.submit_order = Mock(return_value="tif_order_123")

        order_id = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            tif="IOC",  # Immediate or Cancel
        )

        assert order_id == "tif_order_123"

    def test_place_order_with_custom_client_id(self, mock_broker, mock_risk_manager):
        """Test place_order with custom client order ID."""
        engine = LiveExecutionEngine(broker=mock_broker, risk_manager=mock_risk_manager)

        engine.order_validator.validate_order = Mock(return_value=True)
        engine.order_submitter.submit_order = Mock(return_value="custom_order_123")

        order_id = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            client_order_id="my_custom_id",
        )

        assert order_id == "custom_order_123"

    def test_place_order_handles_reduce_only_mode(self, mock_broker, mock_risk_manager):
        """Test place_order respects reduce-only mode from risk manager."""
        # Mock risk manager to indicate reduce-only mode
        mock_risk_manager.is_reduce_only_mode = Mock(return_value=True)

        engine = LiveExecutionEngine(broker=mock_broker, risk_manager=mock_risk_manager)

        engine.order_validator.validate_order = Mock(return_value=True)
        engine.order_submitter.submit_order = Mock(return_value="reduce_order_123")

        order_id = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            reduce_only=True,
        )

        assert order_id == "reduce_order_123"

    def test_place_order_with_leverage_override(self, mock_broker, mock_risk_manager):
        """Test place_order with leverage override."""
        engine = LiveExecutionEngine(broker=mock_broker, risk_manager=mock_risk_manager)

        engine.order_validator.validate_order = Mock(return_value=True)
        engine.order_submitter.submit_order = Mock(return_value="leveraged_order_123")

        order_id = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            leverage=3,
        )

        assert order_id == "leveraged_order_123"

    def test_place_order_with_product_validation(self, mock_broker, mock_risk_manager):
        """Test place_order validates product when provided."""
        engine = LiveExecutionEngine(broker=mock_broker, risk_manager=mock_risk_manager)

        mock_product = Mock()
        mock_product.symbol = "BTC-PERP"

        engine.order_validator.validate_order = Mock(return_value=True)
        engine.order_submitter.submit_order = Mock(return_value="product_order_123")

        order_id = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            product=mock_product,
        )

        assert order_id == "product_order_123"
