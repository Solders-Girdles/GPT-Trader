"""Connection and execution registry for the legacy live-trade facade."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bot_v2.errors import NetworkError, ValidationError, log_error
from bot_v2.errors.handler import RecoveryStrategy, get_error_handler
from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
from bot_v2.features.live_trade.brokers import BrokerInterface, SimulatedBroker
from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.features.live_trade.types import BrokerConnection

logger = logging.getLogger(__name__)


@dataclass
class _SessionState:
    connection: BrokerConnection | None = None
    broker_client: BrokerInterface | None = None
    risk_manager: LiveRiskManager | None = None
    execution_engine: AdvancedExecutionEngine | None = None


def _build_connection_record(
    broker_name: str,
    api_key: str,
    api_secret: str,
    is_paper: bool,
    base_url: str | None,
) -> BrokerConnection:
    return BrokerConnection(
        broker_name=broker_name,
        api_key=api_key,
        api_secret=api_secret,
        is_paper=is_paper,
        is_connected=False,
        account_id=None,
        base_url=base_url,
    )


class SessionRegistry:
    """Stateful registry used by the legacy live-trade facade."""

    def __init__(
        self,
        *,
        broker_factory: Callable[[str, str, str, bool, str | None], BrokerInterface] | None = None,
        risk_manager_factory: Callable[[], LiveRiskManager] | None = None,
        error_handler_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._state = _SessionState()
        self._broker_factory = broker_factory or self._default_broker_factory
        self._enforce_known_brokers = broker_factory is None
        self._risk_manager_factory = risk_manager_factory or LiveRiskManager
        self._error_handler_factory = error_handler_factory or get_error_handler

    def connect_broker(
        self,
        broker_name: str = "simulated",
        api_key: str = "",
        api_secret: str = "",
        is_paper: bool = True,
        base_url: str | None = None,
        *,
        risk_factory: Callable[[], LiveRiskManager] | None = None,
    ) -> BrokerConnection:
        """Connect to the configured broker implementation."""

        requested_name = broker_name.lower().strip() or "simulated"
        alias_map = {"alpaca": "simulated", "ibkr": "simulated"}
        normalized_name = alias_map.get(requested_name, requested_name)

        if self._enforce_known_brokers and normalized_name not in {"simulated"}:
            raise ValidationError(
                f"Invalid broker name: {broker_name}", field="broker_name", value=broker_name
            )

        if normalized_name != requested_name:
            logger.warning(
                "Broker '%s' is deprecated; using simulated broker stub instead",
                requested_name,
            )

        error_handler = self._error_handler_factory()

        def _initialize() -> tuple[
            BrokerConnection,
            BrokerInterface,
            LiveRiskManager,
            AdvancedExecutionEngine,
        ]:
            broker_client = self._broker_factory(
                normalized_name, api_key, api_secret, is_paper, base_url
            )
            connection = _build_connection_record(
                normalized_name, api_key, api_secret, is_paper, base_url
            )

            if not broker_client.connect():
                raise NetworkError(
                    f"Failed to establish connection to {normalized_name}",
                    context={"broker": normalized_name, "is_paper": is_paper},
                )

            connection.is_connected = True
            connection.account_id = broker_client.get_account_id()

            risk_manager_factory = risk_factory or self._risk_manager_factory
            risk_manager = risk_manager_factory()
            execution_engine = AdvancedExecutionEngine(
                broker=broker_client,
                risk_manager=risk_manager,
            )

            logger.info(
                "Connected to %s (%s mode)",
                normalized_name,
                "paper" if is_paper else "live",
            )
            logger.info("Account ID: %s", connection.account_id)

            return connection, broker_client, risk_manager, execution_engine

        try:
            connection, broker_client, risk_manager, execution_engine = error_handler.with_retry(
                _initialize, recovery_strategy=RecoveryStrategy.RETRY
            )
        except (ValidationError, NetworkError):
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            network_error = NetworkError(
                f"Unexpected error connecting to {normalized_name}",
                context={
                    "broker": normalized_name,
                    "is_paper": is_paper,
                    "original_error": str(exc),
                },
            )
            log_error(network_error)
            raise network_error from exc

        self._state = _SessionState(
            connection=connection,
            broker_client=broker_client,
            risk_manager=risk_manager,
            execution_engine=execution_engine,
        )
        return connection

    def disconnect(self) -> None:
        """Disconnect the active broker session and reset registry state."""

        broker_client = self._state.broker_client
        if broker_client is not None:
            try:  # pragma: no cover - defensive clean-up
                broker_client.disconnect()
            except Exception:
                pass

        self._state = _SessionState()
        logger.info("Disconnected from broker")
        print("Disconnected from broker")

    def get_connection(self) -> BrokerConnection | None:
        """Return the active broker connection metadata, if any."""

        return self._state.connection

    def get_broker_client(self) -> BrokerInterface | None:
        """Return the active broker client instance used by the legacy facade."""

        return self._state.broker_client

    def get_risk_manager(self) -> LiveRiskManager | None:
        """Return the risk manager guarding the legacy execution path."""

        return self._state.risk_manager

    def get_execution_engine(self) -> AdvancedExecutionEngine | None:
        """Return the execution engine used by the legacy facade."""

        return self._state.execution_engine

    def teardown(self) -> None:
        """Explicitly release broker resources."""

        self.disconnect()

    @staticmethod
    def _default_broker_factory(
        broker_name: str,
        api_key: str,
        api_secret: str,
        is_paper: bool,
        base_url: str | None,
    ) -> BrokerInterface:
        if broker_name != "simulated":
            raise ValidationError(
                f"Invalid broker name: {broker_name}", field="broker_name", value=broker_name
            )
        return SimulatedBroker()


_DEFAULT_REGISTRY = SessionRegistry()


def connect_broker(
    broker_name: str = "simulated",
    api_key: str = "",
    api_secret: str = "",
    is_paper: bool = True,
    base_url: str | None = None,
    *,
    risk_factory: Callable[[], LiveRiskManager] | None = None,
) -> BrokerConnection:
    """Connect to the simulated broker stub used by legacy demos."""

    return _DEFAULT_REGISTRY.connect_broker(
        broker_name=broker_name,
        api_key=api_key,
        api_secret=api_secret,
        is_paper=is_paper,
        base_url=base_url,
        risk_factory=risk_factory,
    )


def disconnect() -> None:
    """Disconnect the simulated broker session."""

    _DEFAULT_REGISTRY.disconnect()


def get_connection() -> BrokerConnection | None:
    """Return the active broker connection metadata, if any."""

    return _DEFAULT_REGISTRY.get_connection()


def get_broker_client() -> BrokerInterface | None:
    """Return the active broker client instance used by the legacy facade."""

    return _DEFAULT_REGISTRY.get_broker_client()


def get_risk_manager() -> LiveRiskManager | None:
    """Return the risk manager guarding the legacy execution path."""

    return _DEFAULT_REGISTRY.get_risk_manager()


def get_execution_engine() -> AdvancedExecutionEngine | None:
    """Return the execution engine used by the legacy facade."""

    return _DEFAULT_REGISTRY.get_execution_engine()


__all__ = [
    "SessionRegistry",
    "connect_broker",
    "disconnect",
    "get_connection",
    "get_broker_client",
    "get_risk_manager",
    "get_execution_engine",
]
