"""Tests for CoinbaseRestServiceCore position metrics updates."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from tests.unit.gpt_trader.features.brokerages.coinbase.rest.rest_service_core_test_base import (
    RestServiceCoreTestBase,
)


class TestCoinbaseRestServiceCorePositionMetrics(RestServiceCoreTestBase):
    def test_update_position_metrics_no_position(self) -> None:
        self.service.update_position_metrics("BTC-USD")

        self.market_data.get_mark.assert_not_called()

    def test_update_position_metrics_no_mark_price(self) -> None:
        from gpt_trader.features.brokerages.coinbase.utilities import PositionState

        position = PositionState(
            symbol="BTC-USD", side="LONG", quantity=Decimal("0.1"), entry_price=Decimal("50000")
        )
        self.position_store.set("BTC-USD", position)
        self.market_data.get_mark.return_value = None

        self.service.update_position_metrics("BTC-USD")

        self.event_store.append_position.assert_not_called()

    def test_update_position_metrics_missing_position_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from gpt_trader.features.brokerages.coinbase.utilities import PositionState

        position = PositionState(
            symbol="BTC-USD", side="LONG", quantity=Decimal("0.1"), entry_price=Decimal("50000")
        )
        self.position_store.set("BTC-USD", position)

        get_mock = MagicMock(return_value=None)
        monkeypatch.setattr(self.position_store, "get", get_mock)
        self.service.update_position_metrics("BTC-USD")

        self.market_data.get_mark.assert_not_called()

    def test_update_position_metrics_success(self) -> None:
        from gpt_trader.features.brokerages.coinbase.utilities import PositionState

        position = PositionState(
            symbol="BTC-USD", side="LONG", quantity=Decimal("0.1"), entry_price=Decimal("50000")
        )
        self.position_store.set("BTC-USD", position)
        self.market_data.get_mark.return_value = Decimal("51000")
        self.product_catalog.get_funding.return_value = (
            Decimal("0.01"),
            datetime(2024, 1, 1, 12, 0, 0),
        )

        self.service.update_position_metrics("BTC-USD")

        self.event_store.append_metric.assert_called()
        self.event_store.append_position.assert_called()

        position_call = self.event_store.append_position.call_args
        assert position_call[1]["bot_id"] == "coinbase_perps"
        assert position_call[1]["position"]["symbol"] == "BTC-USD"
        assert position_call[1]["position"]["mark_price"] == "51000"

    def test_update_position_metrics_with_funding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gpt_trader.features.brokerages.coinbase.utilities import PositionState

        position = PositionState(
            symbol="BTC-USD", side="LONG", quantity=Decimal("0.1"), entry_price=Decimal("50000")
        )
        self.position_store.set("BTC-USD", position)
        self.market_data.get_mark.return_value = Decimal("51000")
        self.product_catalog.get_funding.return_value = (
            Decimal("0.01"),
            datetime(2024, 1, 1, 12, 0, 0),
        )

        accrue_mock = MagicMock(return_value=Decimal("5.0"))
        monkeypatch.setattr(self.service._funding_calculator, "accrue_if_due", accrue_mock)
        self.service.update_position_metrics("BTC-USD")

        assert self.event_store.append_metric.call_count >= 1
        funding_call = self.event_store.append_metric.call_args_list[0]
        assert funding_call.kwargs["metrics"]["event_type"] == "funding_accrual"
        assert funding_call.kwargs["metrics"]["funding_amount"] == "5.0"
        assert position.realized_pnl == Decimal("5.0")

    def test_update_position_metrics_skips_funding_when_none(self) -> None:
        from gpt_trader.features.brokerages.coinbase.utilities import PositionState

        position = PositionState(
            symbol="BTC-USD", side="LONG", quantity=Decimal("0.1"), entry_price=Decimal("50000")
        )
        self.position_store.set("BTC-USD", position)
        self.market_data.get_mark.return_value = Decimal("51000")
        self.product_catalog.get_funding.return_value = (None, None)

        self.service.update_position_metrics("BTC-USD")

        self.event_store.append_metric.assert_called_once()
        self.event_store.append_position.assert_called_once()

    def test_positions_property(self) -> None:
        from gpt_trader.features.brokerages.coinbase.utilities import PositionState

        position = PositionState(
            symbol="BTC-USD", side="LONG", quantity=Decimal("0.1"), entry_price=Decimal("50000")
        )
        self.position_store.set("BTC-USD", position)

        positions = self.service.positions

        assert isinstance(positions, dict)
        assert "BTC-USD" in positions
        assert positions["BTC-USD"] == position

    def test_update_position_metrics_public_method(self) -> None:
        from gpt_trader.features.brokerages.coinbase.utilities import PositionState

        position = PositionState(
            symbol="BTC-USD", side="LONG", quantity=Decimal("0.1"), entry_price=Decimal("50000")
        )
        self.position_store.set("BTC-USD", position)
        self.market_data.get_mark.return_value = Decimal("51000")
        self.product_catalog.get_funding.return_value = (Decimal("0"), None)

        self.service.update_position_metrics("BTC-USD")

        self.market_data.get_mark.assert_called_once_with("BTC-USD")
