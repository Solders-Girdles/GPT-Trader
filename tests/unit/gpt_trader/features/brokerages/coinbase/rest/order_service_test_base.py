"""Shared fixtures for `OrderService` tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpt_trader.features.brokerages.coinbase.rest.order_service import OrderService


class OrderServiceTestBase:
    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock Coinbase client."""
        return MagicMock()

    @pytest.fixture
    def mock_payload_builder(self) -> MagicMock:
        """Create a mock OrderPayloadBuilder."""
        return MagicMock()

    @pytest.fixture
    def mock_payload_executor(self) -> MagicMock:
        """Create a mock OrderPayloadExecutor."""
        return MagicMock()

    @pytest.fixture
    def mock_position_provider(self) -> MagicMock:
        """Create a mock PositionProvider."""
        provider = MagicMock()
        provider.list_positions.return_value = []
        return provider

    @pytest.fixture
    def order_service(
        self,
        mock_client: MagicMock,
        mock_payload_builder: MagicMock,
        mock_payload_executor: MagicMock,
        mock_position_provider: MagicMock,
    ) -> OrderService:
        """Create an OrderService instance with mocked dependencies."""
        return OrderService(
            client=mock_client,
            payload_builder=mock_payload_builder,
            payload_executor=mock_payload_executor,
            position_provider=mock_position_provider,
        )

    @pytest.fixture
    def sample_order_response(self) -> dict:
        """Create a sample order response from the API."""
        return {
            "order_id": "order-123",
            "client_order_id": "client-123",
            "product_id": "BTC-USD",
            "side": "BUY",
            "order_type": "LIMIT",
            "base_size": "1.0",
            "limit_price": "50000",
            "status": "PENDING",
            "created_time": "2024-01-01T00:00:00Z",
            "last_fill_time": None,
        }
