"""Broker bootstrap helpers for the runtime coordinator."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

from bot_v2.orchestration.broker_factory import create_brokerage
from bot_v2.orchestration.configuration import Profile
from bot_v2.orchestration.derivatives_discovery import (
    DerivativesEligibility,
    discover_derivatives_eligibility,
)
from bot_v2.orchestration.deterministic_broker import DeterministicBroker
from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings
from bot_v2.orchestration.state_manager import ReduceOnlyModeSource
from bot_v2.utilities.telemetry import emit_metric

from .logging_utils import logger
from .models import BrokerBootstrapArtifacts, BrokerBootstrapError

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.features.brokerages.coinbase.market_data_service import MarketDataService
    from bot_v2.features.brokerages.coinbase.utilities import ProductCatalog
    from bot_v2.features.brokerages.core.interfaces import IBrokerage
    from bot_v2.persistence.event_store import EventStore

    from ..base import CoordinatorContext
    from .coordinator import RuntimeCoordinator


class RuntimeCoordinatorBrokerMixin:
    """Encapsulate broker bootstrap logic."""

    def _init_broker(
        self: RuntimeCoordinator, context: CoordinatorContext | None = None
    ) -> CoordinatorContext:
        ctx = context or self.context
        registry = ctx.registry

        if registry.broker is not None:
            broker = registry.broker
            ctx = ctx.with_updates(broker=broker)
            return ctx

        try:
            if self._should_use_mock_broker(ctx):
                artifacts = self._build_mock_broker()
                updated_ctx = ctx
            else:
                artifacts, updated_ctx = self._build_real_broker(ctx)
        except Exception as exc:  # pragma: no cover - fatal boot failure
            logger.error(
                "Failed to initialize broker",
                error=str(exc),
                operation="broker_bootstrap",
                stage="init",
                exc_info=True,
            )
            raise BrokerBootstrapError("Broker initialization failed") from exc

        return self._apply_broker_bootstrap(updated_ctx, artifacts)

    def _should_use_mock_broker(self: RuntimeCoordinator, context: CoordinatorContext) -> bool:
        config = context.config
        paper_env = bool(getattr(config, "perps_paper_trading", False))
        force_mock = bool(getattr(config, "perps_force_mock", False))
        is_dev = config.profile == Profile.DEV
        return paper_env or force_mock or is_dev or bool(getattr(config, "mock_broker", False))

    def _build_mock_broker(self: RuntimeCoordinator) -> BrokerBootstrapArtifacts:
        broker = self._deterministic_broker_cls()
        logger.info(
            "Using deterministic broker (REST-first marks)",
            operation="broker_bootstrap",
            stage="mock",
        )
        return BrokerBootstrapArtifacts(broker=broker, registry_updates={"broker": broker})

    def _build_real_broker(
        self: RuntimeCoordinator, context: CoordinatorContext
    ) -> tuple[BrokerBootstrapArtifacts, CoordinatorContext]:
        settings, ctx = self._resolve_settings(context)
        self._validate_broker_environment(ctx, settings)

        registry = ctx.registry
        broker, event_store, market_data, product_catalog = self._create_brokerage(
            registry,
            event_store=ctx.event_store,
            market_data=registry.market_data_service,
            product_catalog=registry.product_catalog,
            settings=settings,
        )
        if not broker.connect():
            raise RuntimeError("Failed to connect to broker")
        products = broker.list_products()
        logger.info(
            "Connected to broker",
            products=len(products),
            operation="broker_bootstrap",
            stage="connect",
        )

        derivatives_eligibility = None
        if bool(getattr(ctx.config, "derivatives_enabled", False)):
            derivatives_eligibility = self._discover_derivatives_eligibility(broker, ctx)

        artifacts = BrokerBootstrapArtifacts(
            broker=broker,
            event_store=event_store,
            registry_updates={
                "broker": broker,
                "event_store": event_store,
                "market_data_service": market_data,
                "product_catalog": product_catalog,
                "derivatives_eligibility": derivatives_eligibility,
            },
            products=products,
        )
        return artifacts, ctx

    def _resolve_settings(
        self: RuntimeCoordinator, context: CoordinatorContext
    ) -> tuple[RuntimeSettings, CoordinatorContext]:
        registry = context.registry
        settings = registry.runtime_settings
        if isinstance(settings, RuntimeSettings):
            return settings, context

        resolved = load_runtime_settings()
        registry = registry.with_updates(runtime_settings=resolved)
        context = context.with_updates(registry=registry)
        return resolved, context

    def _apply_broker_bootstrap(
        self: RuntimeCoordinator,
        context: CoordinatorContext,
        artifacts: BrokerBootstrapArtifacts,
    ) -> CoordinatorContext:
        registry = context.registry.with_updates(**artifacts.registry_updates)
        event_store = (
            artifacts.event_store if artifacts.event_store is not None else context.event_store
        )

        ctx = context.with_updates(
            broker=artifacts.broker,
            registry=registry,
            event_store=event_store,
        )

        self._hydrate_product_cache(artifacts.products)
        if self._product_cache is not None:
            ctx = ctx.with_updates(product_cache=self._product_cache)

        return ctx

    def _hydrate_product_cache(self: RuntimeCoordinator, products: Sequence[object]) -> None:
        if not products:
            return
        if not isinstance(self._product_cache, dict):
            self._product_cache = {}
        for product in products:
            symbol = getattr(product, "symbol", None)
            if symbol:
                self._product_cache[symbol] = product

    def _validate_broker_environment(
        self: RuntimeCoordinator,
        context: CoordinatorContext,
        settings: RuntimeSettings | None = None,
    ) -> None:
        config = context.config
        if settings is None:
            settings, updated = self._resolve_settings(context)
            context = updated

        if self._should_use_mock_broker(context):
            logger.info(
                "Paper/mock mode enabled — skipping production env checks",
                operation="broker_env_validate",
                stage="mock",
            )
            return

        broker_hint = settings.broker_hint or ""
        if broker_hint != "coinbase":
            raise RuntimeError("BROKER must be set to 'coinbase' for perps trading")

        if settings.coinbase_sandbox_enabled:
            raise RuntimeError(
                "COINBASE_SANDBOX=1 is not supported for live trading. "
                "Remove it or enable PERPS_PAPER=1."
            )

        derivatives_enabled = bool(getattr(config, "derivatives_enabled", False))
        symbols = context.symbols or ()
        if not derivatives_enabled:
            for sym in symbols:
                if sym.upper().endswith("-PERP"):
                    raise RuntimeError(
                        f"Symbol {sym} is perpetual but COINBASE_ENABLE_DERIVATIVES is not enabled."
                    )

        raw_env = settings.raw_env
        api_mode = settings.coinbase_api_mode
        if api_mode != "advanced":
            raise RuntimeError(
                "Perpetuals require Advanced Trade API in production. "
                "Set COINBASE_API_MODE=advanced and unset COINBASE_SANDBOX, "
                "or set PERPS_PAPER=1 for mock mode."
            )

        cdp_key = raw_env.get("COINBASE_PROD_CDP_API_KEY") or raw_env.get("COINBASE_CDP_API_KEY")
        cdp_priv = raw_env.get("COINBASE_PROD_CDP_PRIVATE_KEY") or raw_env.get(
            "COINBASE_CDP_PRIVATE_KEY"
        )
        if derivatives_enabled and not (cdp_key and cdp_priv):
            raise RuntimeError(
                "Missing CDP JWT credentials. Set COINBASE_PROD_CDP_API_KEY and "
                "COINBASE_PROD_CDP_PRIVATE_KEY, or enable PERPS_PAPER=1 for mock trading."
            )

        api_key_present = any(
            raw_env.get(env) for env in ("COINBASE_API_KEY", "COINBASE_PROD_API_KEY")
        )
        api_secret_present = any(
            raw_env.get(env) for env in ("COINBASE_API_SECRET", "COINBASE_PROD_API_SECRET")
        )

        if not (api_key_present and api_secret_present):
            raise RuntimeError(
                "Spot trading requires Coinbase production API key/secret. "
                "Set COINBASE_API_KEY/SECRET (or PROD variants)."
            )

    def _discover_derivatives_eligibility(
        self: RuntimeCoordinator,
        broker: IBrokerage,
        context: CoordinatorContext,
    ) -> DerivativesEligibility:
        requested_market = "BOTH"

        eligibility = discover_derivatives_eligibility(
            broker,
            requested_market=requested_market,
            fail_on_inaccessible=True,
        )

        logger.info(
            "Derivatives eligibility discovered",
            operation="derivatives_discovery",
            stage="complete",
            us_enabled=eligibility.us_derivatives_enabled,
            intx_enabled=eligibility.intx_derivatives_enabled,
            cfm_accessible=eligibility.cfm_portfolio_accessible,
            intx_accessible=eligibility.intx_portfolio_accessible,
            reduce_only_required=eligibility.reduce_only_required,
        )

        if eligibility.reduce_only_required:
            logger.warning(
                "Derivatives requested but not accessible - enabling reduce-only mode",
                operation="derivatives_discovery",
                stage="safety_gate",
                error_message=eligibility.error_message,
            )
            state_manager = getattr(self.context.registry, "reduce_only_state_manager", None)
            if state_manager is not None:
                state_manager.set_reduce_only_mode(
                    enabled=True,
                    reason="derivatives_not_accessible",
                    source=ReduceOnlyModeSource.DERIVATIVES_NOT_ACCESSIBLE,
                    metadata={"context": "derivatives_discovery"},
                )
            else:
                self.set_reduce_only_mode(True, reason="derivatives_not_accessible")

        event_store = context.event_store
        if event_store is not None:
            try:
                emit_metric(
                    event_store,
                    context.bot_id,
                    {
                        "event_type": "derivatives_eligibility_discovered",
                        "us_enabled": eligibility.us_derivatives_enabled,
                        "intx_enabled": eligibility.intx_derivatives_enabled,
                        "cfm_accessible": eligibility.cfm_portfolio_accessible,
                        "intx_accessible": eligibility.intx_portfolio_accessible,
                        "reduce_only_required": eligibility.reduce_only_required,
                        "error_message": eligibility.error_message,
                        "discovery_data": eligibility.discovery_data,
                    },
                    logger=logger,
                )
            except Exception as exc:
                logger.debug(
                    "Failed to emit derivatives discovery metric",
                    error=str(exc),
                    operation="derivatives_discovery",
                    stage="emit_metric",
                )

        return eligibility

    @property
    def _deterministic_broker_cls(self) -> Callable[[], DeterministicBroker]:
        return cast(
            Callable[[], DeterministicBroker],
            DeterministicBroker,
        )

    @property
    def _create_brokerage(
        self,
    ) -> Callable[
        ...,
        tuple[IBrokerage, EventStore, MarketDataService, ProductCatalog],
    ]:
        return cast(
            Callable[
                ...,
                tuple["IBrokerage", "EventStore", "MarketDataService", "ProductCatalog"],
            ],
            create_brokerage,
        )


__all__ = ["RuntimeCoordinatorBrokerMixin"]
