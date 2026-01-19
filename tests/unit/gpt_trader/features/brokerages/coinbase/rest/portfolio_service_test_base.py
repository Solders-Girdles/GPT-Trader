"""Shared fixtures for `PortfolioService` tests."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from gpt_trader.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
from gpt_trader.features.brokerages.coinbase.rest.portfolio_service import PortfolioService
from gpt_trader.persistence.event_store import EventStore


class PortfolioServiceTestBase:
    @pytest.fixture
    def mock_client(self) -> Mock:
        """Create a mock Coinbase REST client."""
        return Mock()  # Don't use spec to allow dynamic method mocking

    @pytest.fixture
    def mock_endpoints(self) -> Mock:
        """Create a mock CoinbaseEndpoints."""
        return Mock(spec=CoinbaseEndpoints)

    @pytest.fixture
    def mock_event_store(self) -> Mock:
        """Create a mock EventStore."""
        return Mock(spec=EventStore)

    @pytest.fixture
    def portfolio_service(
        self,
        mock_client: Mock,
        mock_endpoints: Mock,
        mock_event_store: Mock,
    ) -> PortfolioService:
        """Create a PortfolioService instance with mocked dependencies."""
        return PortfolioService(
            client=mock_client,
            endpoints=mock_endpoints,
            event_store=mock_event_store,
        )
