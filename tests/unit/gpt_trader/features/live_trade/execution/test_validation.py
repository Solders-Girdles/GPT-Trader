"""Tests for OrderValidator core behavior, mark freshness, and slippage guards."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import OrderSide, Product
from gpt_trader.features.live_trade.execution.validation import OrderValidator
from gpt_trader.features.live_trade.risk import ValidationError


class TestOrderValidatorInit:
    def test_init_stores_dependencies(
        self,
        mock_broker: MagicMock,
        mock_risk_manager: MagicMock,
        mock_failure_tracker: MagicMock,
    ) -> None:
        record_preview = MagicMock()
        record_rejection = MagicMock()

        validator = OrderValidator(
            broker=mock_broker,
            risk_manager=mock_risk_manager,
            enable_order_preview=True,
            record_preview_callback=record_preview,
            record_rejection_callback=record_rejection,
            failure_tracker=mock_failure_tracker,
        )

        assert validator.broker is mock_broker
        assert validator.risk_manager is mock_risk_manager
        assert validator.enable_order_preview is True
        assert validator._record_preview is record_preview
        assert validator._record_rejection is record_rejection


class TestRunPreTradeValidation:
    def test_delegates_to_risk_manager(
        self,
        validator: OrderValidator,
        mock_risk_manager: MagicMock,
        mock_product: Product,
    ) -> None:
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
                equity=Decimal("10000"),
                current_positions={},
            )


class TestFinalizeReduceOnlyFlag:
    def test_not_in_reduce_only_mode_returns_user_flag(
        self,
        validator: OrderValidator,
        mock_risk_manager: MagicMock,
    ) -> None:
        mock_risk_manager.is_reduce_only_mode.return_value = False

        assert validator.finalize_reduce_only_flag(False, "BTC-PERP") is False
        assert validator.finalize_reduce_only_flag(True, "BTC-PERP") is True

    def test_reduce_only_mode_forces_true(
        self,
        validator: OrderValidator,
        mock_risk_manager: MagicMock,
    ) -> None:
        mock_risk_manager.is_reduce_only_mode.return_value = True

        assert validator.finalize_reduce_only_flag(False, "BTC-PERP") is True
        assert validator.finalize_reduce_only_flag(True, "BTC-PERP") is True


class TestEnsureMarkIsFresh:
    def test_fresh_mark_passes(
        self, validator: OrderValidator, mock_risk_manager: MagicMock
    ) -> None:
        mock_risk_manager.check_mark_staleness.return_value = False

        validator.ensure_mark_is_fresh("BTC-PERP")
        mock_risk_manager.check_mark_staleness.assert_called_once_with("BTC-PERP")

    def test_stale_mark_raises(
        self, validator: OrderValidator, mock_risk_manager: MagicMock
    ) -> None:
        mock_risk_manager.check_mark_staleness.return_value = True

        with pytest.raises(ValidationError, match="stale"):
            validator.ensure_mark_is_fresh("BTC-PERP")

    def test_exception_in_staleness_check_is_suppressed(
        self, validator: OrderValidator, mock_risk_manager: MagicMock
    ) -> None:
        mock_risk_manager.check_mark_staleness.side_effect = RuntimeError("API error")
        validator.ensure_mark_is_fresh("BTC-PERP")
        mock_risk_manager.check_mark_staleness.assert_called_once_with("BTC-PERP")

    def test_validation_error_propagates(
        self, validator: OrderValidator, mock_risk_manager: MagicMock
    ) -> None:
        mock_risk_manager.check_mark_staleness.side_effect = ValidationError("Custom stale error")
        with pytest.raises(ValidationError, match="Custom stale error"):
            validator.ensure_mark_is_fresh("BTC-PERP")


class TestEnforceSlippageGuard:
    def test_no_snapshot_method_passes(
        self, validator: OrderValidator, mock_broker: MagicMock
    ) -> None:
        del mock_broker.get_market_snapshot
        validator.enforce_slippage_guard(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_quantity=Decimal("1.0"),
            effective_price=Decimal("50000"),
        )
        assert not hasattr(mock_broker, "get_market_snapshot")

    def test_no_snapshot_passes(self, validator: OrderValidator, mock_broker: MagicMock) -> None:
        mock_broker.get_market_snapshot.return_value = None
        validator.enforce_slippage_guard(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_quantity=Decimal("1.0"),
            effective_price=Decimal("50000"),
        )
        mock_broker.get_market_snapshot.assert_called_once_with("BTC-PERP")

    def test_slippage_within_guard_passes(
        self,
        validator: OrderValidator,
        mock_broker: MagicMock,
        mock_risk_manager: MagicMock,
    ) -> None:
        mock_broker.get_market_snapshot.return_value = {
            "spread_bps": 5,
            "depth_l1": 1000000,
        }
        mock_risk_manager.config.slippage_guard_bps = 100

        validator.enforce_slippage_guard(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_quantity=Decimal("0.1"),
            effective_price=Decimal("50000"),
        )
        mock_broker.get_market_snapshot.assert_called_once_with("BTC-PERP")

    def test_slippage_exceeds_guard_raises(
        self,
        validator: OrderValidator,
        mock_broker: MagicMock,
        mock_risk_manager: MagicMock,
    ) -> None:
        mock_broker.get_market_snapshot.return_value = {
            "spread_bps": 50,
            "depth_l1": 1000,
        }
        mock_risk_manager.config.slippage_guard_bps = 10

        with pytest.raises(ValidationError, match="exceeds guard"):
            validator.enforce_slippage_guard(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_quantity=Decimal("10"),
                effective_price=Decimal("50000"),
            )

    def test_zero_depth_uses_fallback(
        self,
        validator: OrderValidator,
        mock_broker: MagicMock,
        mock_risk_manager: MagicMock,
    ) -> None:
        mock_broker.get_market_snapshot.return_value = {
            "spread_bps": 5,
            "depth_l1": 0,
        }
        mock_risk_manager.config.slippage_guard_bps = 1000000

        validator.enforce_slippage_guard(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_quantity=Decimal("0.001"),
            effective_price=Decimal("50000"),
        )
        mock_broker.get_market_snapshot.assert_called_once_with("BTC-PERP")

    def test_exception_in_snapshot_is_suppressed(
        self,
        validator: OrderValidator,
        mock_broker: MagicMock,
    ) -> None:
        mock_broker.get_market_snapshot.side_effect = RuntimeError("API error")

        validator.enforce_slippage_guard(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_quantity=Decimal("1.0"),
            effective_price=Decimal("50000"),
        )
        mock_broker.get_market_snapshot.assert_called_once_with("BTC-PERP")
