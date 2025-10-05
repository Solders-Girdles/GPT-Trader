"""Tests for broker health check functionality."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.orchestration.deterministic_broker import DeterministicBroker


class TestDeterministicBrokerHealth:
    """Test health checking for DeterministicBroker."""

    def test_health_when_connected(self):
        """Health check should report healthy when broker is connected."""
        broker = DeterministicBroker()
        broker.connect()

        health = broker.check_health()

        assert health.connected is True
        assert health.api_responsive is True
        assert health.error_message is None
        assert health.last_check_timestamp > 0

    def test_health_when_disconnected(self):
        """Health check should report unhealthy when broker is disconnected."""
        broker = DeterministicBroker()
        # Don't call connect()

        health = broker.check_health()

        assert health.connected is False
        assert health.api_responsive is False
        assert health.error_message == "Broker not connected"
        assert health.last_check_timestamp > 0

    def test_health_after_disconnect(self):
        """Health check should report unhealthy after disconnect."""
        broker = DeterministicBroker()
        broker.connect()
        broker.disconnect()

        health = broker.check_health()

        assert health.connected is False
        assert health.api_responsive is False
        assert health.error_message == "Broker not connected"


class TestCoinbaseBrokerageHealth:
    """Test health checking for CoinbaseBrokerage."""

    @pytest.fixture
    def api_config(self):
        """Create test API config."""
        return APIConfig(
            api_key="test_key",
            api_secret="test_secret",
            passphrase="test_pass",
            base_url="https://api.coinbase.com",
            sandbox=False,
            ws_url="wss://advanced-trade-ws.coinbase.com",
            enable_derivatives=False,
            auth_type="HMAC",
            api_mode="advanced",
        )

    def test_health_when_disconnected(self, api_config):
        """Health check should report unhealthy when not connected."""
        broker = CoinbaseBrokerage(api_config)
        # Don't call connect()

        health = broker.check_health()

        assert health.connected is False
        assert health.api_responsive is False
        assert health.error_message == "Broker not connected"
        assert health.last_check_timestamp > 0

    def test_health_when_connected_and_api_responsive(self, api_config):
        """Health check should report healthy when connected and API responds."""
        broker = CoinbaseBrokerage(api_config)

        # Mock the connect call
        with patch.object(
            broker.client, "get_accounts", return_value={"accounts": [{"uuid": "test"}]}
        ):
            broker.connect()

        # Mock successful API health check
        with patch.object(broker.client, "get", return_value={"iso": "2024-01-01T00:00:00Z"}):
            health = broker.check_health()

        assert health.connected is True
        assert health.api_responsive is True
        assert health.error_message is None
        assert health.last_check_timestamp > 0

    def test_health_when_connected_but_api_fails(self, api_config):
        """Health check should report degraded when API call fails."""
        broker = CoinbaseBrokerage(api_config)

        # Mock the connect call
        with patch.object(
            broker.client, "get_accounts", return_value={"accounts": [{"uuid": "test"}]}
        ):
            broker.connect()

        # Mock failed API health check
        with patch.object(broker.client, "get", side_effect=Exception("API timeout")):
            health = broker.check_health()

        assert health.connected is True
        assert health.api_responsive is False
        assert "API health check failed" in health.error_message
        assert "API timeout" in health.error_message
        assert health.last_check_timestamp > 0

    def test_health_uses_correct_endpoint_for_advanced_mode(self, api_config):
        """Health check should use correct endpoint for advanced mode."""
        api_config.api_mode = "advanced"
        broker = CoinbaseBrokerage(api_config)

        # Mock connect
        with patch.object(
            broker.client, "get_accounts", return_value={"accounts": [{"uuid": "test"}]}
        ):
            broker.connect()

        # Mock and verify endpoint
        with patch.object(broker.client, "get", return_value={"iso": "2024-01-01"}) as mock_get:
            broker.check_health()
            mock_get.assert_called_once_with("/api/v3/brokerage/time")

    def test_health_uses_correct_endpoint_for_exchange_mode(self, api_config):
        """Health check should use correct endpoint for exchange mode."""
        api_config.api_mode = "exchange"
        broker = CoinbaseBrokerage(api_config)

        # Mock connect
        with patch.object(
            broker.client, "get_accounts", return_value={"accounts": [{"uuid": "test"}]}
        ):
            broker.connect()

        # Mock and verify endpoint
        with patch.object(broker.client, "get", return_value={"iso": "2024-01-01"}) as mock_get:
            broker.check_health()
            mock_get.assert_called_once_with("/v2/time")

    def test_health_after_disconnect(self, api_config):
        """Health check should report unhealthy after disconnect."""
        broker = CoinbaseBrokerage(api_config)

        # Connect then disconnect
        with patch.object(
            broker.client, "get_accounts", return_value={"accounts": [{"uuid": "test"}]}
        ):
            broker.connect()

        broker.disconnect()

        health = broker.check_health()

        assert health.connected is False
        assert health.api_responsive is False
        assert health.error_message == "Broker not connected"


class TestMetricsServerBrokerIntegration:
    """Test broker health integration with MetricsServer."""

    def test_health_endpoint_includes_broker_when_available(self):
        """Health endpoint should include broker health when broker is set."""
        from bot_v2.monitoring.metrics_server import MetricsServer

        broker = DeterministicBroker()
        broker.connect()

        server = MetricsServer(host="127.0.0.1", port=19090, broker=broker)

        # Verify broker is set
        assert server._broker is broker

    def test_health_endpoint_handles_missing_broker(self):
        """Health endpoint should work even when broker is not set."""
        from bot_v2.monitoring.metrics_server import MetricsServer

        server = MetricsServer(host="127.0.0.1", port=19091, broker=None)

        # Should not raise
        assert server._broker is None

    def test_set_broker_updates_reference(self):
        """set_broker() should update broker reference."""
        from bot_v2.monitoring.metrics_server import MetricsServer

        server = MetricsServer(host="127.0.0.1", port=19092)
        broker = DeterministicBroker()

        server.set_broker(broker)

        assert server._broker is broker
