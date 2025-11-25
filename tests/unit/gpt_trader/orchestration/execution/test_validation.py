"""Tests for orchestration/execution/validation.py."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.features.brokerages.core.interfaces import (
    MarketType,
    OrderSide,
    OrderType,
    Product,
    TimeInForce,
)
from gpt_trader.features.live_trade.risk import ValidationError
from gpt_trader.orchestration.execution.validation import OrderValidator

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_broker() -> MagicMock:
    """Create a mock broker."""
    broker = MagicMock()
    broker.get_market_snapshot = MagicMock(return_value=None)
    return broker


@pytest.fixture
def mock_risk_manager() -> MagicMock:
    """Create a mock risk manager."""
    rm = MagicMock()
    rm.config = MagicMock()
    rm.config.slippage_guard_bps = 100
    rm.check_mark_staleness = MagicMock(return_value=False)
    rm.is_reduce_only_mode = MagicMock(return_value=False)
    rm.pre_trade_validate = MagicMock()
    return rm


@pytest.fixture
def mock_product() -> Product:
    """Create a mock product."""
    return Product(
        symbol="BTC-PERP",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        min_size=Decimal("0.001"),
        step_size=Decimal("0.001"),
        min_notional=Decimal("10"),
        price_increment=Decimal("0.01"),
        leverage_max=20,
    )


@pytest.fixture
def validator(mock_broker: MagicMock, mock_risk_manager: MagicMock) -> OrderValidator:
    """Create an OrderValidator instance."""
    record_preview = MagicMock()
    record_rejection = MagicMock()
    return OrderValidator(
        broker=mock_broker,
        risk_manager=mock_risk_manager,
        enable_order_preview=True,
        record_preview_callback=record_preview,
        record_rejection_callback=record_rejection,
    )


# ============================================================
# Test: __init__
# ============================================================


class TestOrderValidatorInit:
    """Tests for OrderValidator initialization."""

    def test_init_stores_dependencies(
        self,
        mock_broker: MagicMock,
        mock_risk_manager: MagicMock,
    ) -> None:
        """Test that dependencies are stored correctly."""
        record_preview = MagicMock()
        record_rejection = MagicMock()

        validator = OrderValidator(
            broker=mock_broker,
            risk_manager=mock_risk_manager,
            enable_order_preview=True,
            record_preview_callback=record_preview,
            record_rejection_callback=record_rejection,
        )

        assert validator.broker is mock_broker
        assert validator.risk_manager is mock_risk_manager
        assert validator.enable_order_preview is True
        assert validator._record_preview is record_preview
        assert validator._record_rejection is record_rejection


# ============================================================
# Test: validate_exchange_rules
# ============================================================


class TestValidateExchangeRules:
    """Tests for validate_exchange_rules method."""

    @patch("gpt_trader.orchestration.execution.validation.spec_validate_order")
    def test_market_order_uses_none_price(
        self,
        mock_spec_validate: MagicMock,
        validator: OrderValidator,
        mock_product: Product,
    ) -> None:
        """Test that market orders pass None as price to validator."""
        mock_spec_validate.return_value = MagicMock(
            ok=True,
            reason=None,
            adjusted_quantity=None,
            adjusted_price=None,
        )

        validator.validate_exchange_rules(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=Decimal("50000"),  # Should be ignored
            effective_price=Decimal("50000"),
            product=mock_product,
        )

        mock_spec_validate.assert_called_once()
        call_kwargs = mock_spec_validate.call_args.kwargs
        assert call_kwargs["price"] is None

    @patch("gpt_trader.orchestration.execution.validation.spec_validate_order")
    def test_limit_order_uses_provided_price(
        self,
        mock_spec_validate: MagicMock,
        validator: OrderValidator,
        mock_product: Product,
    ) -> None:
        """Test that limit orders use the provided price."""
        mock_spec_validate.return_value = MagicMock(
            ok=True,
            reason=None,
            adjusted_quantity=None,
            adjusted_price=None,
        )

        validator.validate_exchange_rules(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=Decimal("49000"),
            effective_price=Decimal("50000"),
            product=mock_product,
        )

        mock_spec_validate.assert_called_once()
        call_kwargs = mock_spec_validate.call_args.kwargs
        assert call_kwargs["price"] == Decimal("49000")

    @patch("gpt_trader.orchestration.execution.validation.spec_validate_order")
    def test_limit_order_falls_back_to_effective_price(
        self,
        mock_spec_validate: MagicMock,
        validator: OrderValidator,
        mock_product: Product,
    ) -> None:
        """Test that limit orders fall back to effective_price when price is None."""
        mock_spec_validate.return_value = MagicMock(
            ok=True,
            reason=None,
            adjusted_quantity=None,
            adjusted_price=None,
        )

        validator.validate_exchange_rules(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            product=mock_product,
        )

        mock_spec_validate.assert_called_once()
        call_kwargs = mock_spec_validate.call_args.kwargs
        assert call_kwargs["price"] == Decimal("50000")

    @patch("gpt_trader.orchestration.execution.validation.spec_validate_order")
    def test_validation_failure_raises_error(
        self,
        mock_spec_validate: MagicMock,
        validator: OrderValidator,
        mock_product: Product,
    ) -> None:
        """Test that validation failure raises ValidationError."""
        mock_spec_validate.return_value = MagicMock(
            ok=False,
            reason="size_below_minimum",
            adjusted_quantity=None,
            adjusted_price=None,
        )

        with pytest.raises(ValidationError, match="size_below_minimum"):
            validator.validate_exchange_rules(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                order_quantity=Decimal("0.0001"),
                price=None,
                effective_price=Decimal("50000"),
                product=mock_product,
            )

        # Verify rejection was recorded
        validator._record_rejection.assert_called_once()

    @patch("gpt_trader.orchestration.execution.validation.spec_validate_order")
    def test_validation_failure_with_none_reason(
        self,
        mock_spec_validate: MagicMock,
        validator: OrderValidator,
        mock_product: Product,
    ) -> None:
        """Test that None reason becomes 'spec_violation'."""
        mock_spec_validate.return_value = MagicMock(
            ok=False,
            reason=None,
            adjusted_quantity=None,
            adjusted_price=None,
        )

        with pytest.raises(ValidationError, match="spec_violation"):
            validator.validate_exchange_rules(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                order_quantity=Decimal("0.0001"),
                price=None,
                effective_price=Decimal("50000"),
                product=mock_product,
            )

    @patch("gpt_trader.orchestration.execution.validation.spec_validate_order")
    @patch("gpt_trader.orchestration.execution.validation.quantize_price_side_aware")
    def test_limit_order_price_quantization(
        self,
        mock_quantize: MagicMock,
        mock_spec_validate: MagicMock,
        validator: OrderValidator,
        mock_product: Product,
    ) -> None:
        """Test that limit order prices are quantized."""
        mock_spec_validate.return_value = MagicMock(
            ok=True,
            reason=None,
            adjusted_quantity=None,
            adjusted_price=None,
        )
        mock_quantize.return_value = Decimal("49000.50")

        qty, price = validator.validate_exchange_rules(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=Decimal("49000.555"),
            effective_price=Decimal("50000"),
            product=mock_product,
        )

        mock_quantize.assert_called_once()
        assert price == Decimal("49000.50")

    @patch("gpt_trader.orchestration.execution.validation.spec_validate_order")
    def test_adjusted_values_from_validation(
        self,
        mock_spec_validate: MagicMock,
        validator: OrderValidator,
        mock_product: Product,
    ) -> None:
        """Test that adjusted values from spec validation are returned."""
        mock_spec_validate.return_value = MagicMock(
            ok=True,
            reason=None,
            adjusted_quantity=Decimal("1.001"),
            adjusted_price=Decimal("49000.00"),
        )

        qty, price = validator.validate_exchange_rules(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            product=mock_product,
        )

        assert qty == Decimal("1.001")
        assert price == Decimal("49000.00")


# ============================================================
# Test: ensure_mark_is_fresh
# ============================================================


class TestEnsureMarkIsFresh:
    """Tests for ensure_mark_is_fresh method."""

    def test_fresh_mark_passes(
        self,
        validator: OrderValidator,
        mock_risk_manager: MagicMock,
    ) -> None:
        """Test that fresh mark price passes validation."""
        mock_risk_manager.check_mark_staleness.return_value = False

        # Should not raise
        validator.ensure_mark_is_fresh("BTC-PERP")

        mock_risk_manager.check_mark_staleness.assert_called_once_with("BTC-PERP")

    def test_stale_mark_raises(
        self,
        validator: OrderValidator,
        mock_risk_manager: MagicMock,
    ) -> None:
        """Test that stale mark price raises ValidationError."""
        mock_risk_manager.check_mark_staleness.return_value = True

        with pytest.raises(ValidationError, match="stale"):
            validator.ensure_mark_is_fresh("BTC-PERP")

    def test_exception_in_staleness_check_is_suppressed(
        self,
        validator: OrderValidator,
        mock_risk_manager: MagicMock,
    ) -> None:
        """Test that non-ValidationError exceptions are suppressed."""
        mock_risk_manager.check_mark_staleness.side_effect = RuntimeError("API error")

        # Should not raise
        validator.ensure_mark_is_fresh("BTC-PERP")

    def test_validation_error_propagates(
        self,
        validator: OrderValidator,
        mock_risk_manager: MagicMock,
    ) -> None:
        """Test that ValidationError from check_mark_staleness propagates."""
        mock_risk_manager.check_mark_staleness.side_effect = ValidationError("Custom stale error")

        with pytest.raises(ValidationError, match="Custom stale error"):
            validator.ensure_mark_is_fresh("BTC-PERP")


# ============================================================
# Test: enforce_slippage_guard
# ============================================================


class TestEnforceSlippageGuard:
    """Tests for enforce_slippage_guard method."""

    def test_no_snapshot_method_passes(
        self,
        validator: OrderValidator,
        mock_broker: MagicMock,
    ) -> None:
        """Test that missing get_market_snapshot method is handled."""
        del mock_broker.get_market_snapshot

        # Should not raise
        validator.enforce_slippage_guard(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_quantity=Decimal("1.0"),
            effective_price=Decimal("50000"),
        )

    def test_no_snapshot_passes(
        self,
        validator: OrderValidator,
        mock_broker: MagicMock,
    ) -> None:
        """Test that None snapshot is handled gracefully."""
        mock_broker.get_market_snapshot.return_value = None

        # Should not raise
        validator.enforce_slippage_guard(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_quantity=Decimal("1.0"),
            effective_price=Decimal("50000"),
        )

    def test_slippage_within_guard_passes(
        self,
        validator: OrderValidator,
        mock_broker: MagicMock,
        mock_risk_manager: MagicMock,
    ) -> None:
        """Test that acceptable slippage passes."""
        mock_broker.get_market_snapshot.return_value = {
            "spread_bps": 5,
            "depth_l1": 1000000,  # High depth = low impact
        }
        mock_risk_manager.config.slippage_guard_bps = 100

        # Should not raise
        validator.enforce_slippage_guard(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_quantity=Decimal("0.1"),
            effective_price=Decimal("50000"),
        )

    def test_slippage_exceeds_guard_raises(
        self,
        validator: OrderValidator,
        mock_broker: MagicMock,
        mock_risk_manager: MagicMock,
    ) -> None:
        """Test that excessive slippage raises ValidationError."""
        mock_broker.get_market_snapshot.return_value = {
            "spread_bps": 50,
            "depth_l1": 1000,  # Low depth = high impact
        }
        mock_risk_manager.config.slippage_guard_bps = 10  # Very tight guard

        with pytest.raises(ValidationError, match="exceeds guard"):
            validator.enforce_slippage_guard(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_quantity=Decimal("10"),  # Large order
                effective_price=Decimal("50000"),
            )

    def test_zero_depth_uses_fallback(
        self,
        validator: OrderValidator,
        mock_broker: MagicMock,
        mock_risk_manager: MagicMock,
    ) -> None:
        """Test that zero depth uses fallback value of 1."""
        mock_broker.get_market_snapshot.return_value = {
            "spread_bps": 5,
            "depth_l1": 0,  # Zero depth
        }
        mock_risk_manager.config.slippage_guard_bps = 1000000  # Very loose guard

        # Should not raise (fallback to depth=1)
        validator.enforce_slippage_guard(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_quantity=Decimal("0.001"),
            effective_price=Decimal("50000"),
        )

    def test_exception_in_snapshot_is_suppressed(
        self,
        validator: OrderValidator,
        mock_broker: MagicMock,
    ) -> None:
        """Test that exceptions in get_market_snapshot are suppressed."""
        mock_broker.get_market_snapshot.side_effect = RuntimeError("API error")

        # Should not raise
        validator.enforce_slippage_guard(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_quantity=Decimal("1.0"),
            effective_price=Decimal("50000"),
        )


# ============================================================
# Test: run_pre_trade_validation
# ============================================================


class TestRunPreTradeValidation:
    """Tests for run_pre_trade_validation method."""

    def test_delegates_to_risk_manager(
        self,
        validator: OrderValidator,
        mock_risk_manager: MagicMock,
        mock_product: Product,
    ) -> None:
        """Test that validation is delegated to risk manager."""
        validator.run_pre_trade_validation(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_quantity=Decimal("1.0"),
            effective_price=Decimal("50000"),
            product=mock_product,
            equity=Decimal("100000"),
            current_positions={"ETH-PERP": {"quantity": Decimal("5")}},
        )

        mock_risk_manager.pre_trade_validate.assert_called_once_with(
            symbol="BTC-PERP",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            product=mock_product,
            equity=Decimal("100000"),
            current_positions={"ETH-PERP": {"quantity": Decimal("5")}},
        )

    def test_validation_error_propagates(
        self,
        validator: OrderValidator,
        mock_risk_manager: MagicMock,
        mock_product: Product,
    ) -> None:
        """Test that ValidationError from risk manager propagates."""
        mock_risk_manager.pre_trade_validate.side_effect = ValidationError(
            "Position limit exceeded"
        )

        with pytest.raises(ValidationError, match="Position limit exceeded"):
            validator.run_pre_trade_validation(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_quantity=Decimal("100"),
                effective_price=Decimal("50000"),
                product=mock_product,
                equity=Decimal("10000"),  # Low equity
                current_positions={},
            )


# ============================================================
# Test: maybe_preview_order
# ============================================================


class TestMaybePreviewOrder:
    """Tests for maybe_preview_order method."""

    def test_preview_disabled_skips(
        self,
        mock_broker: MagicMock,
        mock_risk_manager: MagicMock,
    ) -> None:
        """Test that disabled preview is skipped."""
        validator = OrderValidator(
            broker=mock_broker,
            risk_manager=mock_risk_manager,
            enable_order_preview=False,  # Disabled
            record_preview_callback=MagicMock(),
            record_rejection_callback=MagicMock(),
        )

        # Should return without calling anything
        validator.maybe_preview_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
        )

        # No preview should be recorded
        validator._record_preview.assert_not_called()

    def test_broker_without_preview_skips(
        self,
        mock_risk_manager: MagicMock,
    ) -> None:
        """Test that broker without preview_order method is skipped."""
        # Broker without preview_order method
        broker = MagicMock(spec=[])

        validator = OrderValidator(
            broker=broker,
            risk_manager=mock_risk_manager,
            enable_order_preview=True,
            record_preview_callback=MagicMock(),
            record_rejection_callback=MagicMock(),
        )

        # Should not raise
        validator.maybe_preview_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
        )

        validator._record_preview.assert_not_called()

    def test_preview_success_records_result(
        self,
        mock_risk_manager: MagicMock,
    ) -> None:
        """Test that successful preview is recorded."""
        # Create a broker that implements preview_order and edit_order_preview
        broker = MagicMock()
        broker.preview_order = MagicMock(return_value={"estimated_fee": "0.1"})
        broker.edit_order_preview = MagicMock()

        record_preview = MagicMock()
        validator = OrderValidator(
            broker=broker,
            risk_manager=mock_risk_manager,
            enable_order_preview=True,
            record_preview_callback=record_preview,
            record_rejection_callback=MagicMock(),
        )

        validator.maybe_preview_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
        )

        broker.preview_order.assert_called_once()
        record_preview.assert_called_once()
        call_args = record_preview.call_args[0]
        assert call_args[0] == "BTC-PERP"
        assert call_args[5] == {"estimated_fee": "0.1"}

    def test_preview_validation_error_propagates(
        self,
        mock_risk_manager: MagicMock,
    ) -> None:
        """Test that ValidationError from preview propagates."""
        broker = MagicMock()
        broker.preview_order = MagicMock(side_effect=ValidationError("Insufficient margin"))
        broker.edit_order_preview = MagicMock()

        validator = OrderValidator(
            broker=broker,
            risk_manager=mock_risk_manager,
            enable_order_preview=True,
            record_preview_callback=MagicMock(),
            record_rejection_callback=MagicMock(),
        )

        with pytest.raises(ValidationError, match="Insufficient margin"):
            validator.maybe_preview_order(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                order_quantity=Decimal("1.0"),
                effective_price=Decimal("50000"),
                stop_price=None,
                tif=TimeInForce.GTC,
                reduce_only=False,
                leverage=10,
            )

    def test_preview_generic_exception_is_suppressed(
        self,
        mock_risk_manager: MagicMock,
    ) -> None:
        """Test that generic exceptions from preview are suppressed."""
        broker = MagicMock()
        broker.preview_order = MagicMock(side_effect=RuntimeError("API error"))
        broker.edit_order_preview = MagicMock()

        validator = OrderValidator(
            broker=broker,
            risk_manager=mock_risk_manager,
            enable_order_preview=True,
            record_preview_callback=MagicMock(),
            record_rejection_callback=MagicMock(),
        )

        # Should not raise
        validator.maybe_preview_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
        )

    def test_preview_with_non_tif_value(
        self,
        mock_risk_manager: MagicMock,
    ) -> None:
        """Test that non-TimeInForce tif values are handled."""
        broker = MagicMock()
        broker.preview_order = MagicMock(return_value={})
        broker.edit_order_preview = MagicMock()

        validator = OrderValidator(
            broker=broker,
            risk_manager=mock_risk_manager,
            enable_order_preview=True,
            record_preview_callback=MagicMock(),
            record_rejection_callback=MagicMock(),
        )

        # Pass None as tif - should default to GTC
        validator.maybe_preview_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=10,
        )

        call_kwargs = broker.preview_order.call_args.kwargs
        assert call_kwargs["tif"] == TimeInForce.GTC


# ============================================================
# Test: finalize_reduce_only_flag
# ============================================================


class TestFinalizeReduceOnlyFlag:
    """Tests for finalize_reduce_only_flag method."""

    def test_not_in_reduce_only_mode_returns_user_flag(
        self,
        validator: OrderValidator,
        mock_risk_manager: MagicMock,
    ) -> None:
        """Test that user flag is returned when not in reduce-only mode."""
        mock_risk_manager.is_reduce_only_mode.return_value = False

        assert validator.finalize_reduce_only_flag(False, "BTC-PERP") is False
        assert validator.finalize_reduce_only_flag(True, "BTC-PERP") is True

    def test_reduce_only_mode_forces_true(
        self,
        validator: OrderValidator,
        mock_risk_manager: MagicMock,
    ) -> None:
        """Test that reduce-only mode forces flag to True."""
        mock_risk_manager.is_reduce_only_mode.return_value = True

        # Even False becomes True
        assert validator.finalize_reduce_only_flag(False, "BTC-PERP") is True
        # True stays True
        assert validator.finalize_reduce_only_flag(True, "BTC-PERP") is True


# ============================================================
# Test: Integration scenarios
# ============================================================


class TestValidationIntegration:
    """Integration tests for validation workflows."""

    @patch("gpt_trader.orchestration.execution.validation.spec_validate_order")
    def test_full_validation_flow(
        self,
        mock_spec_validate: MagicMock,
        mock_broker: MagicMock,
        mock_risk_manager: MagicMock,
        mock_product: Product,
    ) -> None:
        """Test a complete validation flow."""
        # Set up mocks
        mock_spec_validate.return_value = MagicMock(
            ok=True,
            reason=None,
            adjusted_quantity=Decimal("1.001"),
            adjusted_price=None,
        )
        mock_broker.get_market_snapshot.return_value = {
            "spread_bps": 5,
            "depth_l1": 100000000,  # Very high depth to minimize impact
        }
        mock_risk_manager.check_mark_staleness.return_value = False
        mock_risk_manager.is_reduce_only_mode.return_value = False
        mock_risk_manager.config.slippage_guard_bps = 500  # Generous guard

        validator = OrderValidator(
            broker=mock_broker,
            risk_manager=mock_risk_manager,
            enable_order_preview=False,
            record_preview_callback=MagicMock(),
            record_rejection_callback=MagicMock(),
        )

        # Step 1: Validate exchange rules
        qty, price = validator.validate_exchange_rules(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            product=mock_product,
        )
        assert qty == Decimal("1.001")

        # Step 2: Ensure mark is fresh
        validator.ensure_mark_is_fresh("BTC-PERP")

        # Step 3: Enforce slippage guard
        validator.enforce_slippage_guard(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_quantity=qty,
            effective_price=Decimal("50000"),
        )

        # Step 4: Pre-trade validation
        validator.run_pre_trade_validation(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_quantity=qty,
            effective_price=Decimal("50000"),
            product=mock_product,
            equity=Decimal("100000"),
            current_positions={},
        )

        # Step 5: Finalize reduce-only flag
        final_reduce_only = validator.finalize_reduce_only_flag(False, "BTC-PERP")
        assert final_reduce_only is False

        # All validation methods should have been called
        mock_spec_validate.assert_called_once()
        mock_risk_manager.check_mark_staleness.assert_called_once()
        mock_risk_manager.pre_trade_validate.assert_called_once()
