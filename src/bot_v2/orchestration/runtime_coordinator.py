"""Runtime coordination helpers for the perps bot orchestration layer."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.live_trade.risk import LiveRiskManager, RiskRuntimeState
from bot_v2.orchestration.broker_factory import create_brokerage
from bot_v2.orchestration.configuration import DEFAULT_SPOT_RISK_PATH, Profile
from bot_v2.orchestration.deterministic_broker import DeterministicBroker
from bot_v2.orchestration.order_reconciler import OrderReconciler

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.orchestration.perps_bot import PerpsBot


logger = logging.getLogger(__name__)


class BrokerBootstrapError(RuntimeError):
    """Raised when broker initialization fails."""


class RuntimeCoordinator:
    """Handles broker/risk bootstrapping and runtime safety toggles."""

    def __init__(self, bot: PerpsBot) -> None:
        self._bot = bot

    # ------------------------------------------------------------------
    def bootstrap(self) -> None:
        """Initialize broker, risk, strategy, and execution services."""

        self._init_broker()
        self._init_risk_manager()
        self._bot.strategy_orchestrator.init_strategy()
        self._bot.execution_coordinator.init_execution()

    # ------------------------------------------------------------------
    def _init_broker(self) -> None:
        bot = self._bot
        if bot.registry.broker is not None:
            bot.broker = bot.registry.broker
            logger.info("Using broker from service registry")
            return

        paper_env = bool(getattr(bot.config, "perps_paper_trading", False))
        force_mock = bool(getattr(bot.config, "perps_force_mock", False))
        is_dev = bot.config.profile == Profile.DEV

        if paper_env or force_mock or is_dev or bot.config.mock_broker:
            bot.broker = DeterministicBroker()
            logger.info("Using deterministic broker (REST-first marks)")
            registry_updates = {"broker": bot.broker}
        else:
            try:
                self._validate_broker_environment()
                broker, event_store, market_data, product_catalog = create_brokerage(
                    bot.registry,
                    event_store=bot.event_store,
                    market_data=bot.registry.market_data_service,
                    product_catalog=bot.registry.product_catalog,
                )
                bot.broker = broker
                bot.event_store = event_store
                registry_updates = {
                    "broker": bot.broker,
                    "event_store": event_store,
                    "market_data_service": market_data,
                    "product_catalog": product_catalog,
                }
                if not bot.broker.connect():
                    raise RuntimeError("Failed to connect to broker")
                products = bot.broker.list_products()
                logger.info("Connected to broker, found %d products", len(products))
                for product in products:
                    if hasattr(product, "symbol"):
                        bot._product_map[product.symbol] = product
            except Exception as exc:  # pragma: no cover - fatal boot failure
                logger.error("Failed to initialize real broker: %s", exc)
                raise BrokerBootstrapError("Broker initialization failed") from exc

        bot.registry = bot.registry.with_updates(**registry_updates)

    def _validate_broker_environment(self) -> None:
        bot = self._bot

        paper_env = bool(getattr(bot.config, "perps_paper_trading", False))
        force_mock = bool(getattr(bot.config, "perps_force_mock", False))
        if paper_env or force_mock or bot.config.mock_broker or bot.config.profile == Profile.DEV:
            logger.info("Paper/mock mode enabled — skipping production env checks")
            return

        broker = os.getenv("BROKER", "").lower()
        if broker != "coinbase":
            raise RuntimeError("BROKER must be set to 'coinbase' for perps trading")

        if os.getenv("COINBASE_SANDBOX", "0") == "1":
            raise RuntimeError(
                "COINBASE_SANDBOX=1 is not supported for live trading. Remove it or enable PERPS_PAPER=1."
            )

        derivatives_enabled = bool(getattr(bot.config, "derivatives_enabled", False))

        if not derivatives_enabled:
            for sym in bot.config.symbols or []:
                if sym.upper().endswith("-PERP"):
                    raise RuntimeError(
                        f"Symbol {sym} is perpetual but COINBASE_ENABLE_DERIVATIVES is not enabled."
                    )

            api_key_present = any(
                os.getenv(env) for env in ("COINBASE_API_KEY", "COINBASE_PROD_API_KEY")
            )
            api_secret_present = any(
                os.getenv(env) for env in ("COINBASE_API_SECRET", "COINBASE_PROD_API_SECRET")
            )
            if not (api_key_present and api_secret_present):
                raise RuntimeError(
                    "Spot trading requires Coinbase production API key/secret. "
                    "Set COINBASE_API_KEY/SECRET (or PROD variants)."
                )
            return

        api_mode = os.getenv("COINBASE_API_MODE", "advanced").lower()
        if api_mode != "advanced":
            raise RuntimeError(
                "Perpetuals require Advanced Trade API in production. "
                "Set COINBASE_API_MODE=advanced and unset COINBASE_SANDBOX, or set PERPS_PAPER=1 for mock mode."
            )

        cdp_key = os.getenv("COINBASE_PROD_CDP_API_KEY") or os.getenv("COINBASE_CDP_API_KEY")
        cdp_priv = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY") or os.getenv(
            "COINBASE_CDP_PRIVATE_KEY"
        )
        if not (cdp_key and cdp_priv):
            raise RuntimeError(
                "Missing CDP JWT credentials. Set COINBASE_PROD_CDP_API_KEY and COINBASE_PROD_CDP_PRIVATE_KEY, "
                "or enable PERPS_PAPER=1 for mock trading."
            )

    # ------------------------------------------------------------------
    def _init_risk_manager(self) -> None:
        bot = self._bot
        controller = bot.config_controller

        if bot.registry.risk_manager is not None:
            bot.risk_manager = bot.registry.risk_manager
            bot.risk_manager.set_state_listener(self.on_risk_state_change)
            controller.sync_with_risk_manager(bot.risk_manager)
            bot.risk_manager.set_reduce_only_mode(controller.reduce_only_mode, reason="config_init")
            return

        env_risk_config_path = os.getenv("RISK_CONFIG_PATH")
        resolved_risk_path = env_risk_config_path
        if not resolved_risk_path and bot.config.profile in {
            Profile.SPOT,
            Profile.DEV,
            Profile.DEMO,
        }:
            if DEFAULT_SPOT_RISK_PATH.exists():
                resolved_risk_path = str(DEFAULT_SPOT_RISK_PATH)
                logger.info("Loading spot risk profile from %s", resolved_risk_path)

        try:
            if resolved_risk_path and os.path.exists(resolved_risk_path):
                risk_config = RiskConfig.from_json(resolved_risk_path)
            else:
                risk_config = RiskConfig.from_env()
        except Exception:
            logger.exception("Failed to load risk config; using defaults")
            risk_config = RiskConfig()

        try:
            if getattr(bot.config, "max_leverage", None):
                risk_config.max_leverage = int(bot.config.max_leverage)
        except Exception as exc:
            logger.warning("Failed to apply max leverage override: %s", exc, exc_info=True)
        try:
            risk_config.reduce_only_mode = bool(bot.config.reduce_only_mode)
        except Exception as exc:
            logger.warning(
                "Failed to sync reduce-only override into risk config: %s", exc, exc_info=True
            )

        bot.risk_manager = LiveRiskManager(config=risk_config, event_store=bot.event_store)
        try:
            if hasattr(bot.broker, "get_position_risk"):
                bot.risk_manager.set_risk_info_provider(bot.broker.get_position_risk)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("Failed to set broker risk info provider: %s", exc, exc_info=True)

        bot.risk_manager.set_state_listener(self.on_risk_state_change)
        controller.sync_with_risk_manager(bot.risk_manager)
        bot.risk_manager.set_reduce_only_mode(controller.reduce_only_mode, reason="config_init")
        bot.registry = bot.registry.with_updates(risk_manager=bot.risk_manager)

    # ------------------------------------------------------------------
    def set_reduce_only_mode(self, enabled: bool, reason: str) -> None:
        bot = self._bot
        controller = bot.config_controller
        if not controller.set_reduce_only_mode(
            enabled, reason=reason, risk_manager=bot.risk_manager
        ):
            return
        logger.warning("Reduce-only mode %s (%s)", "enabled" if enabled else "disabled", reason)
        self._emit_reduce_only_metric(enabled, reason)

    def is_reduce_only_mode(self) -> bool:
        bot = self._bot
        controller = bot.config_controller
        return controller.is_reduce_only_mode(bot.risk_manager)

    # ------------------------------------------------------------------
    def on_risk_state_change(self, state: RiskRuntimeState) -> None:
        bot = self._bot
        controller = bot.config_controller
        reduce_only = bool(state.reduce_only_mode)
        if not controller.apply_risk_update(reduce_only):
            return
        reason = state.last_reduce_only_reason or "unspecified"
        logger.warning(
            "Risk manager toggled reduce-only mode to %s (reason=%s)",
            "enabled" if reduce_only else "disabled",
            reason,
        )
        self._emit_reduce_only_metric(reduce_only, reason)

    def _emit_reduce_only_metric(self, enabled: bool, reason: str) -> None:
        bot = self._bot
        try:
            bot.event_store.append_metric(
                bot_id=bot.bot_id,
                metrics={
                    "event_type": "reduce_only_mode_changed",
                    "enabled": enabled,
                    "reason": reason,
                },
            )
        except Exception:
            logger.exception("Failed to persist bot reduce-only mode change")

    # ------------------------------------------------------------------
    async def reconcile_state_on_startup(self) -> None:
        bot = self._bot
        if bot.config.dry_run or getattr(bot.config, "perps_skip_startup_reconcile", False):
            logger.info("Skipping startup reconciliation (dry-run or PERPS_SKIP_RECONCILE)")
            return

        logger.info("Reconciling state with exchange...")
        try:
            reconciler = OrderReconciler(
                broker=bot.broker,
                orders_store=bot.orders_store,
                event_store=bot.event_store,
                bot_id=bot.bot_id,
            )

            local_open = reconciler.fetch_local_open_orders()
            exchange_open = await reconciler.fetch_exchange_open_orders()

            logger.info(
                "Reconciliation snapshot: local_open=%s exchange_open=%s",
                len(local_open),
                len(exchange_open),
            )
            await reconciler.record_snapshot(local_open, exchange_open)

            diff = reconciler.diff_orders(local_open, exchange_open)
            await reconciler.reconcile_missing_on_exchange(diff)
            reconciler.reconcile_missing_locally(diff)

            try:
                snapshot = await reconciler.snapshot_positions()
                if snapshot:
                    bot._last_positions = snapshot
            except Exception as exc:
                logger.debug("Failed to snapshot initial positions: %s", exc, exc_info=True)

            logger.info("State reconciliation complete.")
        except Exception as exc:
            logger.error("Failed to reconcile state on startup: %s", exc, exc_info=True)
            try:
                bot.event_store.append_error(
                    bot_id=bot.bot_id,
                    message="startup_reconcile_failed",
                    context={"error": str(exc)},
                )
            except Exception:
                logger.exception("Failed to persist startup reconciliation error")
            self.set_reduce_only_mode(True, reason="startup_reconcile_failed")
