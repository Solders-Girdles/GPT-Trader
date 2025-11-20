from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, cast

from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.utilities.logging_patterns import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from bot_v2.orchestration.engines.base import CoordinatorContext

logger = get_logger(__name__, component="telemetry_coordinator")


class NullAccountTelemetry:
    async def run(self, _interval_seconds: int) -> None:  # pragma: no cover - async placeholder
        return None

    def supports_snapshots(self) -> bool:
        return False


def ensure_account_telemetry_stub(coordinator: "TelemetryEngine") -> None:
    extras = getattr(coordinator.context.registry, "extras", None)
    if isinstance(extras, dict) and "account_telemetry" not in extras:
        extras["account_telemetry"] = NullAccountTelemetry()


def initialize_services(
    coordinator: "TelemetryEngine", ctx: "CoordinatorContext"
) -> "CoordinatorContext":
    try:
        from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
        from bot_v2.features.brokerages.coinbase.account_manager import CoinbaseAccountManager
        from bot_v2.orchestration.intx_portfolio_service import IntxPortfolioService
        from bot_v2.orchestration.account_telemetry import AccountTelemetryService
        from bot_v2.orchestration.market_monitor import MarketActivityMonitor
        from bot_v2.orchestration.engines import telemetry_coordinator as telemetry_module

        _patched_get_plog = telemetry_module._get_plog
    except ImportError as exc:
        logger.warning(
            "Failed to import required telemetry dependencies",
            error=str(exc),
            operation="telemetry_init",
            stage="import_failed",
        )
        return ctx

    ensure_account_telemetry_stub(coordinator)

    broker = ctx.broker
    if broker is None:
        logger.warning(
            "Telemetry initialization skipped: no broker available",
            operation="telemetry_init",
            stage="missing_broker",
        )
        return ctx

    try:
        broker_cls = CoinbaseBrokerage
    except Exception:  # pragma: no cover - fallback
        logger.warning(
            "Coinbase adapter unavailable; telemetry coordinator skipping setup",
            operation="telemetry_init",
            stage="adapter_missing",
        )
        return ctx

    if not isinstance(broker_cls, type):
        broker = broker_cls(ctx)
    elif not isinstance(broker, broker_cls):
        logger.warning(
            "Telemetry coordinator requires a Coinbase brokerage; skipping setup",
            operation="telemetry_init",
            stage="adapter_mismatch",
        )
        return ctx

    account_manager = CoinbaseAccountManager(
        cast("CoinbaseBrokerage", broker), event_store=ctx.event_store
    )
    intx_service = IntxPortfolioService(
        account_manager=account_manager,
        runtime_settings=ctx.registry.runtime_settings if ctx.registry else None,
    )
    account_telemetry = AccountTelemetryService(
        broker=broker,
        account_manager=account_manager,
        event_store=ctx.event_store,
        bot_id=ctx.bot_id,
        profile=ctx.config.profile.value,
    )
    if not account_telemetry.supports_snapshots():
        logger.info(
            "Account snapshot telemetry disabled; broker lacks required endpoints",
            operation="telemetry_init",
            stage="snapshot_disabled",
        )

    def _log_market_heartbeat(**payload: Any) -> None:
        try:
            _patched_get_plog().log_market_heartbeat(**payload)
        except Exception as exc:
            logger.debug(
                "Failed to record market heartbeat",
                symbol=payload.get("symbol") or payload.get("source"),
                error=str(exc),
                exc_info=True,
                operation="market_monitor",
                stage="heartbeat",
            )

    try:
        market_monitor = MarketActivityMonitor(ctx.symbols, heartbeat_logger=_log_market_heartbeat)
        coordinator._market_monitor = market_monitor
    except Exception as exc:
        logger.warning(
            "Failed to initialize market monitor; continuing without streaming metrics",
            error=str(exc),
            operation="telemetry_init",
            stage="market_monitor",
        )
        market_monitor = None

    source_extras = getattr(ctx.registry, "extras", {}) or {}
    extras: Dict[str, Any]
    if isinstance(source_extras, dict):
        extras = dict(source_extras)
    else:  # pragma: no cover - defensive copy
        try:
            extras = dict(source_extras)
        except Exception:
            extras = {}
    extras.update(
        {
            "account_manager": account_manager,
            "account_telemetry": account_telemetry,
            "intx_portfolio_service": intx_service,
            "market_monitor": market_monitor,
        }
    )
    updated_registry = ctx.registry.with_updates(extras=extras)
    if not isinstance(updated_registry, ServiceRegistry):
        setattr(updated_registry, "extras", extras)
    return ctx.with_updates(registry=updated_registry)


def init_market_services(coordinator: "TelemetryEngine") -> None:
    updated = initialize_services(coordinator, coordinator.context)
    coordinator.update_context(updated)


async def run_account_telemetry(
    coordinator: "TelemetryEngine", interval_seconds: int
) -> None:
    extras = getattr(coordinator.context.registry, "extras", {})
    account_telemetry = extras.get("account_telemetry") if isinstance(extras, dict) else None
    if not account_telemetry or not account_telemetry.supports_snapshots():
        return
    await account_telemetry.run(interval_seconds)


__all__ = [
    "NullAccountTelemetry",
    "ensure_account_telemetry_stub",
    "initialize_services",
    "init_market_services",
    "run_account_telemetry",
]
