"""Tests for full validation workflows."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.validation as validation_module
from gpt_trader.core import OrderSide, OrderType, Product
from gpt_trader.features.live_trade.execution.validation import OrderValidator


class TestValidationFlow:
    def test_full_validation_flow(
        self,
        mock_broker: MagicMock,
        mock_risk_manager: MagicMock,
        mock_product: Product,
        mock_failure_tracker: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_spec_validate = MagicMock(
            return_value=MagicMock(
                ok=True,
                reason=None,
                adjusted_quantity=Decimal("1.001"),
                adjusted_price=None,
            )
        )
        monkeypatch.setattr(validation_module, "spec_validate_order", mock_spec_validate)

        mock_broker.get_market_snapshot.return_value = {
            "spread_bps": 5,
            "depth_l1": 100000000,
        }
        mock_risk_manager.check_mark_staleness.return_value = False
        mock_risk_manager.is_reduce_only_mode.return_value = False
        mock_risk_manager.config.slippage_guard_bps = 500

        validator = OrderValidator(
            broker=mock_broker,
            risk_manager=mock_risk_manager,
            enable_order_preview=False,
            record_preview_callback=MagicMock(),
            record_rejection_callback=MagicMock(),
            failure_tracker=mock_failure_tracker,
        )

        qty, price = validator.validate_exchange_rules(  # naming: allow
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            product=mock_product,
        )
        assert qty == Decimal("1.001")  # naming: allow

        validator.ensure_mark_is_fresh("BTC-PERP")

        validator.enforce_slippage_guard(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_quantity=qty,  # naming: allow
            effective_price=Decimal("50000"),
        )

        validator.run_pre_trade_validation(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_quantity=qty,  # naming: allow
            effective_price=Decimal("50000"),
            product=mock_product,
            equity=Decimal("100000"),
            current_positions={},
        )

        final_reduce_only = validator.finalize_reduce_only_flag(False, "BTC-PERP")
        assert final_reduce_only is False

        mock_spec_validate.assert_called_once()
        mock_risk_manager.check_mark_staleness.assert_called_once()
        mock_risk_manager.pre_trade_validate.assert_called_once()
