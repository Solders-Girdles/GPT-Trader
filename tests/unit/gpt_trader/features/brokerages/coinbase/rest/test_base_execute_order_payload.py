"""Tests for CoinbaseRestServiceCore order payload execution."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

import gpt_trader.features.brokerages.coinbase.rest.base as rest_base_module
from gpt_trader.core import (
    InsufficientFunds,
    InvalidRequestError,
    Order,
)
from gpt_trader.errors import ValidationError
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.rest_service_core_test_base import (
    RestServiceCoreTestBase,
)


class TestCoinbaseRestServiceCoreExecuteOrderPayload(RestServiceCoreTestBase):
    def test_execute_order_payload_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {"product_id": "BTC-USD", "side": "BUY"}
        mock_order = Mock(spec=Order)
        mock_order.id = "order_123"
        self.client.place_order.return_value = {"order_id": "order_123"}
        monkeypatch.setattr(rest_base_module, "to_order", lambda x: mock_order)

        result = self.service._execute_order_payload("BTC-USD", payload, "client_123")

        assert result == mock_order
        self.client.place_order.assert_called_once_with(payload)

    def test_execute_order_payload_with_preview(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {"product_id": "BTC-USD", "side": "BUY"}
        mock_order = Mock(spec=Order)
        self.client.preview_order.return_value = {"success": True}
        self.client.place_order.return_value = {"order_id": "order_123"}

        mock_bot_config = Mock()
        mock_bot_config.enable_order_preview = True
        self.service.bot_config = mock_bot_config

        monkeypatch.setattr(rest_base_module, "to_order", lambda x: mock_order)

        result = self.service._execute_order_payload("BTC-USD", payload, "client_123")

        assert result == mock_order
        self.client.preview_order.assert_called_once_with(payload)
        self.client.place_order.assert_called_once_with(payload)

    def test_execute_order_payload_preview_failure_still_places(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = {"product_id": "BTC-USD", "side": "BUY"}
        mock_order = Mock(spec=Order)
        self.client.preview_order.side_effect = Exception("preview failed")
        self.client.place_order.return_value = {"order_id": "order_123"}

        mock_bot_config = Mock()
        mock_bot_config.enable_order_preview = True
        self.service.bot_config = mock_bot_config

        monkeypatch.setattr(rest_base_module, "to_order", lambda x: mock_order)

        result = self.service._execute_order_payload("BTC-USD", payload, "client_123")

        assert result == mock_order
        self.client.preview_order.assert_called_once_with(payload)
        self.client.place_order.assert_called_once_with(payload)

    def test_execute_order_payload_insufficient_funds(self) -> None:
        payload = {"product_id": "BTC-USD", "side": "BUY"}
        self.client.place_order.side_effect = InsufficientFunds("Insufficient balance")

        with pytest.raises(InsufficientFunds):
            self.service._execute_order_payload("BTC-USD", payload, "client_123")

    def test_execute_order_payload_validation_error(self) -> None:
        payload = {"product_id": "BTC-USD", "side": "BUY"}
        self.client.place_order.side_effect = ValidationError("Invalid order")

        with pytest.raises(ValidationError):
            self.service._execute_order_payload("BTC-USD", payload, "client_123")

    def test_execute_order_payload_duplicate_client_id(self) -> None:
        payload = {"product_id": "BTC-USD", "side": "BUY"}
        mock_order = Mock(spec=Order)
        mock_order.id = "existing_order_123"

        self.client.place_order.side_effect = InvalidRequestError("duplicate client_order_id")
        self.service._find_existing_order_by_client_id = Mock(return_value=mock_order)

        result = self.service._execute_order_payload("BTC-USD", payload, "client_123")

        assert result == mock_order
        self.service._find_existing_order_by_client_id.assert_called_once_with(
            "BTC-USD", "client_123"
        )

    def test_execute_order_payload_duplicate_client_id_retry_failure(self) -> None:
        payload = {"product_id": "BTC-USD", "side": "BUY"}
        error = InvalidRequestError("duplicate client_order_id")
        self.client.place_order.side_effect = [error, Exception("retry failed")]
        self.service._find_existing_order_by_client_id = Mock(return_value=None)

        with pytest.raises(InvalidRequestError) as exc:
            self.service._execute_order_payload("BTC-USD", payload, "client_123")

        assert "duplicate client_order_id" in str(exc.value)
        assert self.client.place_order.call_count == 2
        self.service._find_existing_order_by_client_id.assert_called_once_with(
            "BTC-USD", "client_123"
        )

    def test_execute_order_payload_network_error(self) -> None:
        payload = {"product_id": "BTC-USD", "side": "BUY"}
        self.client.place_order.side_effect = ConnectionError("network down")

        with pytest.raises(ConnectionError):
            self.service._execute_order_payload("BTC-USD", payload, "client_123")

    def test_execute_order_payload_unexpected_error(self) -> None:
        payload = {"product_id": "BTC-USD", "side": "BUY"}
        self.client.place_order.side_effect = Exception("Unexpected error")

        with pytest.raises(Exception, match="Unexpected error"):
            self.service._execute_order_payload("BTC-USD", payload, "client_123")
