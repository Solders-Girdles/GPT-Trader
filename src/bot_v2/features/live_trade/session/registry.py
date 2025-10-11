"""Connection and execution registry for the legacy live-trade facade."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

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


_SESSION = _SessionState()


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

    requested_name = broker_name.lower().strip() or "simulated"
    alias_map = {"alpaca": "simulated", "ibkr": "simulated"}
    normalized_name = alias_map.get(requested_name, requested_name)

    if normalized_name not in {"simulated"}:
        raise ValidationError(
            f"Invalid broker name: {broker_name}", field="broker_name", value=broker_name
        )

    if normalized_name != requested_name:
        logger.warning(
            "Broker '%s' is deprecated; using simulated broker stub instead",
            requested_name,
        )

    error_handler = get_error_handler()

    def _initialize() -> tuple[
        BrokerConnection,
        BrokerInterface,
        LiveRiskManager,
        AdvancedExecutionEngine,
    ]:
        broker_client = SimulatedBroker()
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

        risk_manager = risk_factory() if risk_factory else LiveRiskManager()
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

    _SESSION.connection = connection
    _SESSION.broker_client = broker_client
    _SESSION.risk_manager = risk_manager
    _SESSION.execution_engine = execution_engine
    return connection


def disconnect() -> None:
    """Disconnect the simulated broker session."""

    broker_client = _SESSION.broker_client
    if broker_client is not None:
        try:  # pragma: no cover - defensive clean-up
            broker_client.disconnect()
        except Exception:
            pass

    _SESSION.connection = None
    _SESSION.broker_client = None
    _SESSION.risk_manager = None
    _SESSION.execution_engine = None

    logger.info("Disconnected from broker")
    print("Disconnected from broker")


def get_connection() -> BrokerConnection | None:
    """Return the active broker connection metadata, if any."""

    return _SESSION.connection


def get_broker_client() -> BrokerInterface | None:
    """Return the active broker client instance used by the legacy facade."""

    return _SESSION.broker_client


def get_risk_manager() -> LiveRiskManager | None:
    """Return the risk manager guarding the legacy execution path."""

    return _SESSION.risk_manager


def get_execution_engine() -> AdvancedExecutionEngine | None:
    """Return the execution engine used by the legacy facade."""

    return _SESSION.execution_engine


__all__ = [
    "connect_broker",
    "disconnect",
    "get_connection",
    "get_broker_client",
    "get_risk_manager",
    "get_execution_engine",
]
