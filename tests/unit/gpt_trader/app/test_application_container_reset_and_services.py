"""Unit tests for ApplicationContainer resets and secondary services."""

from __future__ import annotations

from unittest.mock import MagicMock

from gpt_trader.app.config import BotConfig
from gpt_trader.app.container import ApplicationContainer


class TestApplicationContainerResets:
    """Test cases for ApplicationContainer reset helpers."""

    def test_reset_broker(self, mock_config: BotConfig) -> None:
        """Test that broker can be reset."""
        from gpt_trader.app.containers.brokerage import BrokerageContainer

        mock_broker = MagicMock()
        mock_create_brokerage = MagicMock()
        mock_create_brokerage.return_value = (
            mock_broker,
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

        container = ApplicationContainer(mock_config)
        container._brokerage = BrokerageContainer(
            config=mock_config,
            event_store_provider=lambda: container.event_store,
            broker_factory=mock_create_brokerage,
        )

        broker = container.broker
        assert broker == mock_broker
        assert container._brokerage._broker == mock_broker

        container.reset_broker()
        assert container._brokerage._broker is None

        broker2 = container.broker
        assert broker2 == mock_broker
        assert container._brokerage._broker == mock_broker
        assert mock_create_brokerage.call_count == 2

    def test_reset_config(self, mock_config: BotConfig) -> None:
        """Test that config controller can be reset."""
        container = ApplicationContainer(mock_config)

        config_controller = container.config_controller
        assert config_controller is not None
        assert container._config_container._config_controller == config_controller

        container.reset_config()
        assert container._config_container._config_controller is None

        config_controller2 = container.config_controller
        assert config_controller2 is not None
        assert config_controller2 is not config_controller


class TestApplicationContainerSecondaryServices:
    """Test lazy construction for secondary services."""

    def test_validation_failure_tracker_creation(self, mock_config: BotConfig) -> None:
        """Test that validation failure tracker is created correctly."""
        from gpt_trader.features.live_trade.execution.validation import ValidationFailureTracker

        container = ApplicationContainer(mock_config)

        tracker = container.validation_failure_tracker

        assert isinstance(tracker, ValidationFailureTracker)
        assert container._risk_validation._validation_failure_tracker == tracker
        assert tracker.escalation_threshold == 5
        assert tracker.escalation_callback is None

        tracker2 = container.validation_failure_tracker
        assert tracker is tracker2

    def test_profile_loader_creation(self, mock_config: BotConfig) -> None:
        """Test that profile loader is created correctly."""
        from gpt_trader.app.config.profile_loader import ProfileLoader

        container = ApplicationContainer(mock_config)

        loader = container.profile_loader

        assert isinstance(loader, ProfileLoader)
        assert container._config_container._profile_loader == loader

        loader2 = container.profile_loader
        assert loader is loader2

    def test_health_state_creation(self, mock_config: BotConfig) -> None:
        """Test that health state is created correctly."""
        from gpt_trader.app.health_server import HealthState

        container = ApplicationContainer(mock_config)

        health_state = container.health_state

        assert isinstance(health_state, HealthState)
        assert container._observability._health_state == health_state
        assert health_state.ready is False
        assert health_state.live is True

        health_state2 = container.health_state
        assert health_state is health_state2

    def test_secrets_manager_creation(self, mock_config: BotConfig) -> None:
        """Test that secrets manager is created correctly."""
        from gpt_trader.security.secrets_manager import SecretsManager

        container = ApplicationContainer(mock_config)

        secrets_manager = container.secrets_manager

        assert isinstance(secrets_manager, SecretsManager)
        assert container._observability._secrets_manager == secrets_manager
        assert secrets_manager._config == mock_config

        secrets_manager2 = container.secrets_manager
        assert secrets_manager is secrets_manager2
