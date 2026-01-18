"""Tests for core OrderValidator behavior."""

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
