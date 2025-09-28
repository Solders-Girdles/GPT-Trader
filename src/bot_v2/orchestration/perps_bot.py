from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sys
import threading
import time as _time
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import psutil
from bot_v2.backtest.profile import load_profile as _load_spot_profile
from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.config.path_registry import RUNTIME_DATA_DIR
from bot_v2.errors import ExecutionError, ValidationError
from bot_v2.features.brokerages.coinbase.account_manager import CoinbaseAccountManager
from bot_v2.features.brokerages.core.interfaces import (
    Balance,
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Product,
    TimeInForce,
)
from bot_v2.features.live_trade.indicators import (
    mean_decimal as _mean_decimal,
)
from bot_v2.features.live_trade.indicators import (
    relative_strength_index as _rsi_from_closes,
)
from bot_v2.features.live_trade.indicators import (
    to_decimal as _to_decimal,
)
from bot_v2.features.live_trade.indicators import (
    true_range as _true_range,
)
from bot_v2.features.live_trade.risk import LiveRiskManager, RiskRuntimeState
from bot_v2.features.live_trade.risk import ValidationError as RiskValidationError
from bot_v2.features.live_trade.strategies.perps_baseline import (
    Action,
    BaselinePerpsStrategy,
    Decision,
    StrategyConfig,
)
from bot_v2.features.monitor import LogLevel
from bot_v2.features.monitor import get_logger as _get_plog
from bot_v2.orchestration.broker_factory import create_brokerage
from bot_v2.orchestration.configuration import (
    DEFAULT_SPOT_RISK_PATH,
    BotConfig,
    ConfigManager,
    ConfigValidationError,
    Profile,
)
from bot_v2.orchestration.live_execution import LiveExecutionEngine
from bot_v2.orchestration.market_monitor import MarketActivityMonitor
from bot_v2.orchestration.mock_broker import MockBroker
from bot_v2.orchestration.perps_bootstrap import prepare_perps_bot
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.orchestration.session_guard import TradingSessionGuard

logger = logging.getLogger(__name__)


class PerpsBot:
    """Main Coinbase trading bot (spot by default, perps optional)."""

    def __init__(self, config: BotConfig, registry: ServiceRegistry | None = None):
        bootstrap = prepare_perps_bot(config, registry)
        self.config = bootstrap.config
        self.registry = bootstrap.registry
        for record in bootstrap.logs:
            logger.log(record.level, record.message, *record.args)

        try:
            self._config_manager = ConfigManager.from_config(self.config)
        except ConfigValidationError as exc:
            raise ExecutionError(f"Invalid configuration: {exc}") from exc
        self._pending_config_update: BotConfig | None = None
        self.start_time = datetime.now(UTC)
        self._session_guard = TradingSessionGuard(
            start=self.config.trading_window_start,
            end=self.config.trading_window_end,
            trading_days=self.config.trading_days,
        )
        self.event_store = bootstrap.event_store
        self.orders_store = bootstrap.orders_store
        self.registry = bootstrap.registry

        self.running = False
        self._reduce_only_mode_state = bool(self.config.reduce_only_mode)

        # Initialize product map BEFORE _init_broker() uses it
        self._product_map: dict[str, Product] = {}

        self._spot_rules: dict[str, dict[str, Any]] = {}
        self._symbol_strategies: dict[str, BaselinePerpsStrategy] = {}

        self._init_broker()
        self._init_risk_manager()
        self._init_strategy()
        self._init_execution()

        # Bot identifier aligned with execution engine
        self.bot_id = "perps_bot"

        # High-level account helper and cache
        self.account_manager = CoinbaseAccountManager(self.broker, event_store=self.event_store)
        self._latest_account_snapshot: dict[str, Any] = {}
        self._required_account_attrs = (
            "get_key_permissions",
            "get_fee_schedule",
            "get_account_limits",
            "get_transaction_summary",
            "list_payment_methods",
            "list_portfolios",
        )
        self._account_snapshot_supported = all(
            hasattr(self.broker, attr) for attr in self._required_account_attrs
        )
        if not self._account_snapshot_supported:
            logger.info("Account snapshot telemetry disabled; broker lacks required endpoints")

        # Basic order success tracking
        self.order_stats = {"attempted": 0, "successful": 0, "failed": 0}

        self.mark_windows: dict[str, list[Decimal]] = {s: [] for s in config.symbols}
        self.orderbook_signals: dict[str, dict[str, Decimal]] = {}
        self._mark_lock = threading.RLock()
        self._orderbook_lock = threading.RLock()
        self.last_decisions: dict[str, Any] = {}
        # Baseline for position reconciliation
        self._last_positions: dict[str, dict[str, Any]] = {}
        # Order placement lock to avoid race conditions with reconciliation
        self._order_lock: asyncio.Lock | None = None

        # Minimal market monitor for reliability tests (WS-free)
        def _log_market_heartbeat(**payload: dict) -> None:
            try:
                _get_plog().log_market_heartbeat(**payload)
            except Exception as exc:
                logger.debug(
                    "Failed to record market heartbeat for %s: %s",
                    payload.get("symbol") or payload.get("source"),
                    exc,
                    exc_info=True,
                )

        self._market_monitor = MarketActivityMonitor(
            self.config.symbols,
            heartbeat_logger=_log_market_heartbeat,
        )

        # Optional streaming: gated by env and profile (Market Data Reliability - Phase 1)
        enable_stream = os.getenv("PERPS_ENABLE_STREAMING", "").lower() in ("1", "true", "yes")
        if enable_stream and self.config.profile in {Profile.CANARY, Profile.PROD}:
            try:
                self._start_streaming_background()
            except Exception:
                logger.exception("Failed to start streaming background worker")

    def _supports_account_snapshot(self) -> bool:
        return all(hasattr(self.broker, attr) for attr in self._required_account_attrs)

    def _start_streaming_background(self) -> None:  # pragma: no cover - gated by env/profile
        """Start a background thread to consume broker streams and update marks/timestamps."""
        try:
            symbols = list(self.config.symbols)
            if not symbols:
                return
            level = int(os.getenv("PERPS_STREAM_LEVEL", "1") or "1")
            # Init stop event
            try:
                self._ws_stop = threading.Event()
            except Exception:
                self._ws_stop = None
            self._ws_thread = threading.Thread(
                target=self._run_stream_loop, args=(symbols, level), daemon=True
            )
            self._ws_thread.start()
            logger.info(f"Started WS streaming thread for symbols={symbols} level={level}")
        except Exception as e:
            logger.warning(f"Failed to start streaming: {e}")

    def _run_stream_loop(self, symbols: list[str], level: int) -> None:
        """Consume stream_orderbook and update marks + staleness timestamps."""
        try:
            stream = None
            # Prefer orderbook (ticker) for L1 mid; fallback to trades if unavailable
            try:
                stream = self.broker.stream_orderbook(symbols, level=level)
            except Exception as exc:
                logger.warning("Orderbook stream unavailable, falling back to trades: %s", exc)
                try:
                    stream = self.broker.stream_trades(symbols)
                except Exception as trade_exc:
                    logger.error("Failed to start streaming trades: %s", trade_exc)
                    return

            # Iterate messages until process stop; underlying WS handles reconnect
            for msg in stream:
                # Allow graceful shutdown
                if hasattr(self, "_ws_stop") and self._ws_stop and self._ws_stop.is_set():
                    break
                try:
                    if not isinstance(msg, dict):
                        continue
                    sym = str(msg.get("product_id") or msg.get("symbol") or "")
                    if not sym:
                        continue
                    mark = None
                    bid = msg.get("best_bid") or msg.get("bid")
                    ask = msg.get("best_ask") or msg.get("ask")
                    if bid is not None and ask is not None:
                        try:
                            from decimal import Decimal as _D

                            mark = (_D(str(bid)) + _D(str(ask))) / _D("2")
                        except Exception:
                            mark = None
                    if mark is None:
                        mark = msg.get("last") or msg.get("price")
                    if mark is None:
                        continue
                    mark = Decimal(str(mark))
                    if mark <= 0:
                        continue

                    # Update MA window and risk staleness timestamp
                    self._update_mark_window(sym, mark)
                    try:
                        self._market_monitor.record_update(sym)
                        # Update risk staleness timestamp
                        self.risk_manager.last_mark_update[sym] = datetime.utcnow()
                        # Emit metric snapshot
                        self.event_store.append_metric(
                            "perps_bot",
                            {"event_type": "ws_mark_update", "symbol": sym, "mark": str(mark)},
                        )
                    except Exception:
                        logger.exception("WS mark update bookkeeping failed for %s", sym)
                except Exception:
                    logger.debug("Dropped malformed WS message: %s", msg, exc_info=True)
                    continue
        except Exception as e:
            try:
                self.event_store.append_metric(
                    "perps_bot",
                    {
                        "event_type": "ws_stream_error",
                        "message": str(e),
                    },
                )
            except Exception:
                logger.exception("Failed to record WS stream error metric")
        finally:
            # Mark thread exit
            try:
                self.event_store.append_metric("perps_bot", {"event_type": "ws_stream_exit"})
            except Exception:
                logger.exception("Failed to record WS stream exit metric")

    async def _reconcile_state_on_startup(self):
        # Allow skipping in dry-run or when explicitly requested via env
        if self.config.dry_run or os.getenv("PERPS_SKIP_RECONCILE", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            logger.info("Skipping startup reconciliation (dry-run or PERPS_SKIP_RECONCILE)")
            return
        logger.info("Reconciling state with exchange...")
        try:
            local_open_orders = {o.order_id: o for o in self.orders_store.get_open_orders()}

            exchange_open_orders: dict[str, Order] = {}
            interested_statuses = [
                OrderStatus.PENDING,
                OrderStatus.SUBMITTED,
                OrderStatus.PARTIALLY_FILLED,
            ]
            for status in interested_statuses:
                try:
                    orders = await asyncio.to_thread(self.broker.list_orders, status=status)
                    for order in orders or []:
                        exchange_open_orders[order.id] = order
                except TypeError:
                    if not exchange_open_orders:
                        try:
                            all_orders = await asyncio.to_thread(self.broker.list_orders)
                            for order in all_orders or []:
                                if order.status in interested_statuses:
                                    exchange_open_orders[order.id] = order
                        except Exception:
                            exchange_open_orders = {}
                    break
                except Exception:
                    continue

            logger.info(
                "Reconciliation snapshot: local_open=%s exchange_open=%s",
                len(local_open_orders),
                len(exchange_open_orders),
            )
            try:
                self.event_store.append_metric(
                    bot_id=self.bot_id,
                    metrics={
                        "event_type": "order_reconcile_snapshot",
                        "local_open": len(local_open_orders),
                        "exchange_open": len(exchange_open_orders),
                    },
                )
            except Exception as exc:
                logger.exception("Failed to persist order reconciliation snapshot: %s", exc)

            for order_id, local_order in local_open_orders.items():
                if order_id not in exchange_open_orders:
                    logger.warning(
                        f"Order {order_id} is OPEN locally but not on exchange. Fetching final status..."
                    )
                    try:
                        final_order_status = await asyncio.to_thread(
                            self.broker.get_order, order_id
                        )
                    except Exception:
                        final_order_status = None
                    if final_order_status:
                        self.orders_store.upsert(final_order_status)
                        try:
                            self.event_store.append_metric(
                                bot_id=self.bot_id,
                                metrics={
                                    "event_type": "order_reconciled",
                                    "order_id": order_id,
                                    "status": final_order_status.status.value,
                                },
                            )
                        except Exception:
                            logger.exception("Failed to log order reconciliation for %s", order_id)
                        logger.info(
                            f"Updated order {order_id} to status: {final_order_status.status.value}"
                        )
                    else:
                        logger.error(f"Could not retrieve final status for order {order_id}.")
                        try:
                            cancelled_order = Order(
                                id=local_order.order_id,
                                client_id=local_order.client_id,
                                symbol=local_order.symbol,
                                side=OrderSide(local_order.side.lower()),
                                type=OrderType(local_order.order_type.lower()),
                                qty=Decimal(str(local_order.qty)),
                                price=(
                                    Decimal(str(local_order.price)) if local_order.price else None
                                ),
                                stop_price=None,
                                tif=TimeInForce.GTC,
                                status=OrderStatus.CANCELLED,
                                filled_qty=(
                                    Decimal(str(local_order.filled_qty))
                                    if local_order.filled_qty
                                    else Decimal("0")
                                ),
                                avg_fill_price=(
                                    Decimal(str(local_order.avg_fill_price))
                                    if local_order.avg_fill_price
                                    else None
                                ),
                                submitted_at=datetime.fromisoformat(local_order.created_at),
                                updated_at=datetime.utcnow(),
                            )
                            self.orders_store.upsert(cancelled_order)
                            try:
                                self.event_store.append_metric(
                                    bot_id=self.bot_id,
                                    metrics={
                                        "event_type": "order_reconciled",
                                        "order_id": order_id,
                                        "status": OrderStatus.CANCELLED.value,
                                        "reason": "assumed_cancelled",
                                    },
                                )
                            except Exception:
                                logger.exception(
                                    "Failed to log assumed cancellation for %s", order_id
                                )
                            logger.info(
                                "Marked order %s as cancelled due to missing on exchange", order_id
                            )
                        except Exception as exc:
                            logger.debug(
                                "Failed to mark %s cancelled during reconciliation: %s",
                                order_id,
                                exc,
                                exc_info=True,
                            )

                for order_id, exchange_order in exchange_open_orders.items():
                    if order_id not in local_open_orders:
                        logger.warning(
                            f"Found untracked OPEN order on exchange: {order_id}. Adding to store."
                        )
                    try:
                        self.orders_store.upsert(exchange_order)
                    except Exception as ex:
                        logger.debug(f"Failed to upsert exchange order {order_id}: {ex}")

            # Prime the position baseline after successful reconciliation
            try:
                positions = await asyncio.to_thread(self.broker.list_positions)
                self._last_positions = {
                    p.symbol: {
                        "qty": str(getattr(p, "qty", "0")),
                        "side": str(getattr(p, "side", "")),
                    }
                    for p in (positions or [])
                    if getattr(p, "symbol", None)
                }
            except Exception as exc:
                logger.debug("Failed to snapshot initial positions: %s", exc, exc_info=True)

            logger.info("State reconciliation complete.")
        except Exception as e:
            logger.error(f"Failed to reconcile state on startup: {e}", exc_info=True)
            try:
                self.event_store.append_error(
                    bot_id=self.bot_id,
                    message="startup_reconcile_failed",
                    context={"error": str(e)},
                )
            except Exception:
                logger.exception("Failed to persist startup reconciliation error")
            # Enter reduce-only mode as a safety fallback
            self._set_reduce_only_mode(True, reason="startup_reconcile_failed")

    def _init_risk_manager(self):
        if self.registry.risk_manager is not None:
            self.risk_manager = self.registry.risk_manager
            self.risk_manager.set_state_listener(self._on_risk_state_change)
            initial_reduce_only = (
                bool(self.config.reduce_only_mode) or self.risk_manager.is_reduce_only_mode()
            )
            self._reduce_only_mode_state = initial_reduce_only
            self.config.reduce_only_mode = initial_reduce_only
            self.risk_manager.set_reduce_only_mode(initial_reduce_only, reason="config_init")
            return

        # Load risk config from JSON path or environment, then apply safe overrides
        env_risk_cfg_path = os.environ.get("RISK_CONFIG_PATH")
        resolved_risk_path = env_risk_cfg_path
        if not resolved_risk_path and self.config.profile in {
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
            # Fallback to minimal config if loading fails
            logger.exception("Failed to load risk config; using defaults")
            risk_config = RiskConfig()

        # Apply BotConfig overrides conservatively
        try:
            if hasattr(self.config, "max_leverage") and self.config.max_leverage:
                risk_config.max_leverage = int(self.config.max_leverage)
        except Exception as exc:
            logger.warning("Failed to apply max leverage override: %s", exc, exc_info=True)
        try:
            if hasattr(self.config, "reduce_only_mode"):
                risk_config.reduce_only_mode = bool(self.config.reduce_only_mode)
        except Exception as exc:
            logger.warning(
                "Failed to sync reduce-only override into risk config: %s", exc, exc_info=True
            )

        # Seed risk manager and provide exchange risk info provider when available
        self.risk_manager = LiveRiskManager(config=risk_config, event_store=self.event_store)
        try:
            if hasattr(self.broker, "get_position_risk"):
                self.risk_manager.set_risk_info_provider(self.broker.get_position_risk)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("Failed to set broker risk info provider: %s", exc, exc_info=True)

        self.risk_manager.set_state_listener(self._on_risk_state_change)
        initial_reduce_only = (
            bool(self.config.reduce_only_mode) or self.risk_manager.is_reduce_only_mode()
        )
        self._reduce_only_mode_state = initial_reduce_only
        self.config.reduce_only_mode = initial_reduce_only
        self.risk_manager.set_reduce_only_mode(initial_reduce_only, reason="config_init")
        self.registry = self.registry.with_updates(risk_manager=self.risk_manager)

    def _on_risk_state_change(self, state: RiskRuntimeState) -> None:
        reduce_only = bool(state.reduce_only_mode)
        if reduce_only == self._reduce_only_mode_state:
            return
        self._reduce_only_mode_state = reduce_only
        previous = bool(self.config.reduce_only_mode)
        self.config.reduce_only_mode = reduce_only
        logger.warning(
            "Risk manager toggled reduce-only mode to %s (reason=%s)",
            "enabled" if reduce_only else "disabled",
            state.last_reduce_only_reason or "unspecified",
        )
        if reduce_only != previous:
            try:
                self.event_store.append_metric(
                    bot_id=self.bot_id,
                    metrics={
                        "event_type": "reduce_only_mode_changed",
                        "enabled": reduce_only,
                        "reason": state.last_reduce_only_reason or "unspecified",
                    },
                )
            except Exception:
                logger.exception("Failed to persist bot reduce-only mode change")

    def _set_reduce_only_mode(self, enabled: bool, reason: str) -> None:
        # Keep bot config and risk manager state synchronized.
        if enabled == self._reduce_only_mode_state:
            return
        self._reduce_only_mode_state = enabled
        self.config.reduce_only_mode = enabled
        logger.warning("Reduce-only mode %s (%s)", "enabled" if enabled else "disabled", reason)
        if hasattr(self, "risk_manager") and self.risk_manager:
            self.risk_manager.set_reduce_only_mode(enabled, reason=reason)

    def is_reduce_only_mode(self) -> bool:
        if hasattr(self, "risk_manager") and self.risk_manager:
            # Risk manager is authoritative; config mirrors its state via listener
            return bool(self._reduce_only_mode_state or self.risk_manager.is_reduce_only_mode())
        return bool(self._reduce_only_mode_state)

    def _init_strategy(self):
        derivatives_enabled = os.environ.get("COINBASE_ENABLE_DERIVATIVES", "0") == "1"
        if self.config.profile == Profile.SPOT:
            self._load_spot_rules()
            for symbol in self.config.symbols:
                rule = self._spot_rules.get(symbol, {})
                short = int(rule.get("short_window", self.config.short_ma))
                long = int(rule.get("long_window", self.config.long_ma))
                strategy_kwargs = {
                    "short_ma_period": short,
                    "long_ma_period": long,
                    "target_leverage": 1,
                    "trailing_stop_pct": self.config.trailing_stop_pct,
                    "enable_shorts": False,
                }
                fraction_override = rule.get("position_fraction") or os.environ.get(
                    "PERPS_POSITION_FRACTION"
                )
                if fraction_override is not None:
                    try:
                        strategy_kwargs["position_fraction"] = float(fraction_override)
                    except (TypeError, ValueError):
                        logger.warning(
                            "Invalid position_fraction=%s for %s; using default",
                            fraction_override,
                            symbol,
                        )
                self._symbol_strategies[symbol] = BaselinePerpsStrategy(
                    config=StrategyConfig(**strategy_kwargs),
                    risk_manager=self.risk_manager,
                )
        else:
            strategy_kwargs = {
                "short_ma_period": self.config.short_ma,
                "long_ma_period": self.config.long_ma,
                "target_leverage": self.config.target_leverage if derivatives_enabled else 1,
                "trailing_stop_pct": self.config.trailing_stop_pct,
                "enable_shorts": self.config.enable_shorts if derivatives_enabled else False,
            }

            fraction_override = os.environ.get("PERPS_POSITION_FRACTION")
            if fraction_override:
                try:
                    strategy_kwargs["position_fraction"] = float(fraction_override)
                except ValueError:
                    logger.warning(
                        "Invalid PERPS_POSITION_FRACTION=%s; using default", fraction_override
                    )

            self.strategy = BaselinePerpsStrategy(
                config=StrategyConfig(**strategy_kwargs), risk_manager=self.risk_manager
            )

    def _init_execution(self):
        # Optional per-symbol slippage multipliers (e.g., "BTC-PERP:0.0005,ETH-PERP:0.0007")
        slippage_env = os.environ.get("SLIPPAGE_MULTIPLIERS", "")
        slippage_map = {}
        if slippage_env:
            try:
                parts = [p for p in slippage_env.split(",") if ":" in p]
                for part in parts:
                    k, v = part.split(":", 1)
                    slippage_map[k.strip()] = float(v)
            except Exception as exc:
                logger.warning(
                    "Invalid SLIPPAGE_MULTIPLIERS entry '%s': %s", slippage_env, exc, exc_info=True
                )

        self.exec_engine = LiveExecutionEngine(
            broker=self.broker,
            risk_manager=self.risk_manager,
            event_store=self.event_store,
            bot_id="perps_bot",
            slippage_multipliers=slippage_map or None,
            enable_preview=self.config.enable_order_preview,
        )
        logger.info("Initialized LiveExecutionEngine with risk integration")
        extras = dict(self.registry.extras)
        extras["execution_engine"] = self.exec_engine
        self.registry = self.registry.with_updates(extras=extras)

    def _init_broker(self):
        if self.registry.broker is not None:
            self.broker = self.registry.broker
            logger.info("Using broker from service registry")
            return
        paper_env = os.environ.get("PERPS_PAPER", "").lower() in ("1", "true", "yes")
        force_mock = os.environ.get("PERPS_FORCE_MOCK", "").lower() in ("1", "true", "yes")
        is_dev = self.config.profile == Profile.DEV
        if paper_env or force_mock or is_dev or self.config.mock_broker:
            self.broker = MockBroker()
            logger.info("Using mock broker (REST-first marks)")
        else:
            try:
                self._validate_broker_environment()
                self.broker = create_brokerage()
                if not self.broker.connect():
                    raise RuntimeError("Failed to connect to broker")
                products = self.broker.list_products()
                logger.info(f"Connected to broker, found {len(products)} products")
                for product in products:
                    if hasattr(product, "symbol"):
                        self._product_map[product.symbol] = product
                # Streaming removed; rely on REST get_quote()
            except Exception as e:
                logger.error(f"Failed to initialize real broker: {e}")
                sys.exit(1)
        self.registry = self.registry.with_updates(broker=self.broker)

    def _validate_broker_environment(self):
        import os

        # Allow safe paper/mock mode without any production credentials
        paper_env = os.environ.get("PERPS_PAPER", "").lower() in ("1", "true", "yes")
        force_mock = os.environ.get("PERPS_FORCE_MOCK", "").lower() in ("1", "true", "yes")
        if paper_env or force_mock or self.config.mock_broker or self.config.profile == Profile.DEV:
            logger.info("Paper/mock mode enabled — skipping production env checks")
            return

        # Enforce broker selection
        broker = os.getenv("BROKER", "").lower()
        if broker != "coinbase":
            raise RuntimeError("BROKER must be set to 'coinbase' for perps trading")

        sandbox = os.getenv("COINBASE_SANDBOX", "0") == "1"
        if sandbox:
            raise RuntimeError(
                "COINBASE_SANDBOX=1 is not supported for live trading. Remove it or enable PERPS_PAPER=1."
            )

        derivatives_enabled = os.getenv("COINBASE_ENABLE_DERIVATIVES", "0") == "1"

        # Spot mode: forbid -PERP symbols and require production API credentials.
        if not derivatives_enabled:
            for sym in self.config.symbols:
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
                    "Spot trading requires Coinbase production API key/secret. Set COINBASE_API_KEY/SECRET (or PROD variants)."
                )
            return

        # Perpetuals require Advanced Trade in production.
        api_mode = os.getenv("COINBASE_API_MODE", "advanced").lower()
        if api_mode != "advanced":
            raise RuntimeError(
                "Perpetuals require Advanced Trade API in production. "
                "Set COINBASE_API_MODE=advanced and unset COINBASE_SANDBOX, or set PERPS_PAPER=1 for mock mode."
            )

        # Authentication: require CDP JWT credentials explicitly (no HMAC fallback for perps)
        cdp_key = os.getenv("COINBASE_PROD_CDP_API_KEY") or os.getenv("COINBASE_CDP_API_KEY")
        cdp_priv = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY") or os.getenv(
            "COINBASE_CDP_PRIVATE_KEY"
        )
        if not (cdp_key and cdp_priv):
            raise RuntimeError(
                "Missing CDP JWT credentials. Set COINBASE_PROD_CDP_API_KEY and COINBASE_PROD_CDP_PRIVATE_KEY, "
                "or enable PERPS_PAPER=1 for mock trading."
            )

        # Streaming removed; no WebSocket dependency required

    def _load_spot_rules(self) -> None:
        self._spot_rules = {}
        profile_path = Path(os.environ.get("SPOT_PROFILE_PATH", "config/profiles/spot.yaml"))
        if not profile_path.exists():
            logger.info("Spot profile %s not found; using default parameters", profile_path)
            return
        try:
            profile_doc = _load_spot_profile(profile_path)
        except Exception as exc:
            logger.warning("Failed to load spot profile %s: %s", profile_path, exc, exc_info=True)
            return

        strategies = profile_doc.get("strategy", {}) if isinstance(profile_doc, dict) else {}
        for symbol in self.config.symbols:
            keys = [symbol, symbol.lower(), symbol.upper()]
            if "-" in symbol:
                base = symbol.split("-")[0]
                keys.extend([base, base.lower(), base.upper()])
            for key in keys:
                if key in strategies:
                    self._spot_rules[symbol] = strategies.get(key) or {}
                    break
            else:
                logger.warning(
                    "No strategy entry for %s in %s; defaults will be used", symbol, profile_path
                )

    def get_product(self, symbol: str) -> Product:
        if symbol in self._product_map:
            return self._product_map[symbol]
        base, _, quote = symbol.partition("-")
        quote = quote or os.environ.get("COINBASE_DEFAULT_QUOTE", "USD").upper()
        market_type = MarketType.PERPETUAL if symbol.upper().endswith("-PERP") else MarketType.SPOT
        # Provide conservative defaults; execution will re-quantize via product catalog once populated
        return Product(
            symbol=symbol,
            base_asset=base,
            quote_asset=quote,
            market_type=market_type,
            step_size=Decimal("0.00000001"),
            min_size=Decimal("0.00000001"),
            price_increment=Decimal("0.01"),
            min_notional=Decimal("10"),
        )

    async def update_marks(self):
        for symbol in self.config.symbols:
            try:
                quote = await asyncio.to_thread(self.broker.get_quote, symbol)
                assert quote is not None, f"No quote for {symbol}"
                # IBrokerage Quote uses 'last'
                last_price = getattr(quote, "last", getattr(quote, "last_price", None))
                assert last_price is not None, f"Quote missing price for {symbol}"
                mark = Decimal(str(last_price))
                assert mark > 0, f"Invalid mark price: {mark} for {symbol}"
                ts = getattr(quote, "ts", datetime.now(UTC))
                self._update_mark_window(symbol, mark)
                # Update risk manager mark timestamp for staleness guards
                try:
                    self.risk_manager.last_mark_update[symbol] = (
                        ts if isinstance(ts, datetime) else datetime.utcnow()
                    )
                except Exception as exc:
                    logger.debug(
                        "Failed to update mark timestamp for %s: %s", symbol, exc, exc_info=True
                    )
            except Exception as e:
                logger.error(f"Error updating mark for {symbol}: {e}")

    async def process_symbol(
        self,
        symbol: str,
        balances: Sequence[Balance] | None = None,
        position_map: dict[str, Position] | None = None,
    ):
        try:
            if balances is None:
                balances = await asyncio.to_thread(self.broker.list_balances)
            cash_assets = {"USD", "USDC"}
            usd_balance = next(
                (b for b in balances if getattr(b, "asset", "").upper() in cash_assets),
                None,
            )
            equity = usd_balance.total if usd_balance else Decimal("0")
            # Global kill switch before doing any work
            if getattr(self.risk_manager.config, "kill_switch_enabled", False):
                logger.warning("Kill switch enabled - skipping trading loop")
                return

            if position_map is None:
                positions = await asyncio.to_thread(self.broker.list_positions)
                positions_lookup = {p.symbol: p for p in positions if hasattr(p, "symbol")}
            else:
                positions_lookup = position_map

            position_state = None
            position_qty = Decimal("0")
            if symbol in positions_lookup:
                pos = positions_lookup[symbol]
                qty_val = getattr(pos, "qty", Decimal("0"))
                position_state = {
                    "qty": qty_val,
                    "side": getattr(pos, "side", "long"),
                    "entry": getattr(pos, "entry_price", None),
                }
                try:
                    position_qty = Decimal(str(qty_val))
                except Exception:
                    position_qty = Decimal("0")

            marks = self.mark_windows.get(symbol, [])
            if not marks:
                logger.warning(f"No marks for {symbol}")
                return

            if position_qty and marks:
                try:
                    equity += abs(position_qty) * marks[-1]
                except Exception as exc:
                    logger.debug(
                        "Failed to adjust equity for %s position: %s", symbol, exc, exc_info=True
                    )

            if equity == Decimal("0"):
                logger.error(f"No equity info for {symbol}")
                return
            # Volatility circuit breaker (progressive)
            try:
                cb = self.risk_manager.check_volatility_circuit_breaker(
                    symbol, marks[-max(self.config.long_ma, 20) :]
                )
                if cb.get("triggered") and cb.get("action") == "kill_switch":
                    # Halt immediately for this cycle
                    logger.warning(f"Kill switch tripped by volatility CB for {symbol}")
                    return
            except Exception as exc:
                logger.debug(
                    "Volatility circuit breaker check failed for %s: %s", symbol, exc, exc_info=True
                )
            # Staleness guard – skip trading on stale data
            try:
                if self.risk_manager.check_mark_staleness(symbol):
                    logger.warning(f"Skipping {symbol} due to stale market data")
                    return
            except Exception as exc:
                logger.debug("Mark staleness check failed for %s: %s", symbol, exc, exc_info=True)
            # Measure strategy decision time
            strategy_obj = self._get_strategy(symbol)
            _t0 = _time.perf_counter()
            decision = strategy_obj.decide(
                symbol=symbol,
                current_mark=marks[-1],
                position_state=position_state,
                recent_marks=marks[:-1] if len(marks) > 1 else [],
                equity=equity,
                product=self.get_product(symbol),
            )
            _dt_ms = (_time.perf_counter() - _t0) * 1000.0
            try:
                _get_plog().log_strategy_duration(
                    strategy=type(strategy_obj).__name__, duration_ms=_dt_ms
                )
            except Exception as exc:
                logger.debug("Failed to log strategy duration: %s", exc, exc_info=True)

            if self.config.profile == Profile.SPOT:
                decision = await self._apply_spot_filters(symbol, decision, position_state)

            self.last_decisions[symbol] = decision
            logger.info(f"{symbol} Decision: {decision.action.value} - {decision.reason}")

            if decision.action in {Action.BUY, Action.SELL, Action.CLOSE}:
                await self.execute_decision(
                    symbol, decision, marks[-1], self.get_product(symbol), position_state
                )
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    async def execute_decision(
        self,
        symbol: str,
        decision: Any,
        mark: Decimal,
        product: Product,
        position_state: dict[str, Any] | None,
    ):
        try:
            assert product is not None, "Missing product metadata"
            assert mark is not None and mark > 0, f"Invalid mark: {mark}"
            assert position_state is None or "qty" in position_state, "Position state missing qty"
            if self.config.dry_run:
                logger.info(f"DRY RUN: Would execute {decision.action.value} for {symbol}")
                return

            if decision.action == Action.CLOSE:
                if not position_state or position_state.get("qty", 0) == 0:
                    logger.warning(f"No position to close for {symbol}")
                    return
                qty = abs(Decimal(str(position_state["qty"])))
            elif decision.target_notional:
                qty = decision.target_notional / mark
            elif decision.qty:
                qty = decision.qty
            else:
                logger.warning(f"No qty or notional in decision for {symbol}")
                return

            side = OrderSide.BUY if decision.action == Action.BUY else OrderSide.SELL
            if decision.action == Action.CLOSE:
                side = (
                    OrderSide.SELL
                    if position_state and position_state.get("side") == "long"
                    else OrderSide.BUY
                )

            reduce_only_global = self.is_reduce_only_mode()
            reduce_only = (
                decision.reduce_only or reduce_only_global or decision.action == Action.CLOSE
            )

            # Advanced order params (optional)
            order_type = getattr(decision, "order_type", OrderType.MARKET)
            limit_price = getattr(decision, "limit_price", None)
            stop_price = getattr(decision, "stop_trigger", None)
            tif = getattr(decision, "time_in_force", None)
            # Map TIF: decision override (enum or str) -> enum; else use config default
            try:
                if isinstance(tif, str):
                    tif = TimeInForce[tif.upper()]
                elif tif is None and isinstance(self.config.time_in_force, str):
                    tif = TimeInForce[self.config.time_in_force.upper()]
            except Exception:
                tif = None

            order = await self._place_order(
                symbol=symbol,
                side=side,
                qty=qty,
                order_type=(
                    order_type
                    if isinstance(order_type, OrderType)
                    else (
                        OrderType[order_type.upper()]
                        if isinstance(order_type, str)
                        else OrderType.MARKET
                    )
                ),
                product=product,
                reduce_only=reduce_only,
                leverage=decision.leverage,
                price=limit_price,
                stop_price=stop_price,
                tif=tif or None,
            )

            if order:
                logger.info(f"Order placed successfully: {order.id}")
            else:
                logger.warning(f"Order rejected or failed for {symbol}")
        except Exception as e:
            logger.error(f"Error executing decision for {symbol}: {e}")

    def _ensure_order_lock(self) -> asyncio.Lock:
        """Ensure the order placement lock is available when called inside the event loop."""
        if self._order_lock is None:
            try:
                self._order_lock = asyncio.Lock()
            except RuntimeError as exc:
                logger.error("Unable to initialize async order lock: %s", exc)
                raise
        return self._order_lock

    async def _place_order(self, **kwargs) -> Order | None:
        try:
            # Ensure single in-flight placement to avoid duplicate orders
            lock = self._ensure_order_lock()
            async with lock:
                return await self._place_order_inner(**kwargs)
        except (ValidationError, RiskValidationError, ExecutionError) as e:
            logger.warning(f"Order validation/execution failed: {e}")
            self.order_stats["failed"] += 1
            # Re-raise so callers/tests can assert on validation failures
            raise
        except Exception as e:
            logger.error(f"Failed to place order: {e}", exc_info=True)
            self.order_stats["failed"] += 1
            return None

    async def _place_order_inner(self, **kwargs) -> Order | None:
        # Track attempts
        self.order_stats["attempted"] += 1
        order_id = await asyncio.to_thread(self.exec_engine.place_order, **kwargs)
        if order_id:
            order = await asyncio.to_thread(self.broker.get_order, order_id)
            if order:
                self.orders_store.upsert(order)
                self.order_stats["successful"] += 1
                logger.info(
                    f"Order recorded: {order.id} {order.side.value} {order.qty} {order.symbol}"
                )
                return order
        # Failure path
        self.order_stats["failed"] += 1
        logger.warning("Order attempt failed (no order_id returned)")
        return None

    async def run_cycle(self):
        logger.debug("Running update cycle")
        await self.update_marks()
        if not self._session_guard.should_trade():
            logger.info("Outside trading window; skipping trading actions this cycle")
            await self.log_status()
            return
        balances: Sequence[Balance] = []
        positions: Sequence[Position] = []
        position_map: dict[str, Position] = {}

        try:
            balances = await asyncio.to_thread(self.broker.list_balances)
        except Exception as exc:
            logger.warning("Unable to fetch balances for trading cycle: %s", exc)

        try:
            positions = await asyncio.to_thread(self.broker.list_positions)
        except Exception as exc:
            logger.warning("Unable to fetch positions for trading cycle: %s", exc)

        if positions:
            position_map = {p.symbol: p for p in positions if hasattr(p, "symbol")}

        process_sig = inspect.signature(self.process_symbol)
        expects_context = len(process_sig.parameters) > 1
        tasks = []
        for symbol in self.config.symbols:
            if expects_context:
                tasks.append(self.process_symbol(symbol, balances, position_map))
            else:
                tasks.append(self.process_symbol(symbol))
        await asyncio.gather(*tasks)
        await self.log_status()

    async def log_status(self):
        positions = []
        balances = []
        try:
            positions = await asyncio.to_thread(self.broker.list_positions)
        except Exception as e:
            logger.warning(f"Unable to fetch positions for status log: {e}")
        try:
            balances = await asyncio.to_thread(self.broker.list_balances)
        except Exception as e:
            logger.warning(f"Unable to fetch balances for status log: {e}")

        usd_balance = next((b for b in balances if getattr(b, "asset", "").upper() == "USD"), None)
        equity = usd_balance.available if usd_balance else Decimal("0")

        logger.info("=" * 60)
        logger.info(
            f"Bot Status - {datetime.now()} - Profile: {self.config.profile.value} - Equity: ${equity} - Positions: {len(positions)}"
        )
        for symbol, decision in self.last_decisions.items():
            logger.info(f"  {symbol}: {decision.action.value} ({decision.reason})")
        logger.info("=" * 60)

        try:
            open_orders_count = len(self.orders_store.get_open_orders())
        except Exception:
            open_orders_count = 0

        metrics_payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "profile": self.config.profile.value,
            "equity": float(equity) if isinstance(equity, Decimal) else equity,
            "positions": [
                {
                    "symbol": getattr(p, "symbol", ""),
                    "qty": float(getattr(p, "qty", 0) or 0),
                    "side": getattr(p, "side", ""),
                    "entry_price": float(getattr(p, "entry_price", 0) or 0),
                    "mark_price": float(getattr(p, "mark_price", 0) or 0),
                }
                for p in positions
                if getattr(p, "symbol", None)
            ],
            "decisions": {
                sym: {"action": decision.action.value, "reason": decision.reason}
                for sym, decision in self.last_decisions.items()
            },
            "order_stats": dict(self.order_stats),
            "open_orders": open_orders_count,
            "uptime_seconds": (datetime.now(UTC) - self.start_time).total_seconds(),
        }

        if self._latest_account_snapshot:
            metrics_payload["account_snapshot"] = self._latest_account_snapshot

        try:
            cpu_percent = psutil.cpu_percent(interval=0.0)
            mem = psutil.virtual_memory()
            metrics_payload["system"] = {
                "cpu_percent": cpu_percent,
                "memory_percent": mem.percent,
                "memory_used_mb": mem.used / (1024 * 1024),
            }
        except Exception as exc:
            logger.debug("Unable to collect system metrics: %s", exc, exc_info=True)

        self._publish_metrics(metrics_payload)

    def _publish_metrics(self, metrics: dict[str, Any]) -> None:
        try:
            self.event_store.append_metric(
                bot_id=self.bot_id, metrics={"event_type": "cycle_metrics", **metrics}
            )
        except Exception as exc:
            logger.exception("Failed to persist cycle metrics: %s", exc)

        try:
            metrics_path = (
                RUNTIME_DATA_DIR / "perps_bot" / self.config.profile.value / "metrics.json"
            )
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to write metrics file: {e}")

        try:
            _get_plog().log_event(
                level=LogLevel.INFO,
                event_type="metrics_update",
                message="Cycle metrics updated",
                component="PerpsBot",
                **{k: v for k, v in metrics.items() if k not in {"positions", "decisions"}},
            )
        except Exception as exc:
            logger.debug("Failed to emit metrics update event: %s", exc, exc_info=True)

    def _check_config_updates(self) -> None:
        if not getattr(self, "_config_manager", None):
            return
        try:
            updated = self._config_manager.refresh_if_changed()
        except ConfigValidationError as exc:
            logger.error("Configuration update rejected: %s", exc)
            return

        if not updated:
            return

        diff = self._summarize_config_diff(self.config, updated)
        if diff:
            logger.warning(
                "Configuration change detected for profile %s: %s",
                self.config.profile.value,
                diff,
            )
        else:
            logger.warning(
                "Configuration inputs changed for profile %s; restart recommended to apply updates",
                self.config.profile.value,
            )
        self._pending_config_update = updated

    @staticmethod
    def _summarize_config_diff(current: BotConfig, updated: BotConfig) -> dict[str, Any]:
        tracked = (
            "symbols",
            "update_interval",
            "max_position_size",
            "max_leverage",
            "reduce_only_mode",
            "time_in_force",
            "daily_loss_limit",
        )
        diff: dict[str, Any] = {}
        for field_name in tracked:
            if getattr(current, field_name) != getattr(updated, field_name):
                diff[field_name] = {
                    "current": getattr(current, field_name),
                    "new": getattr(updated, field_name),
                }
        return diff

    async def run(self, single_cycle: bool = False):
        logger.info(f"Starting Perps Bot - Profile: {self.config.profile.value}")
        self.running = True
        background_tasks: list[asyncio.Task[Any]] = []
        try:
            # Only reconcile and start background guard loops when not in dry-run
            if not self.config.dry_run:
                await self._reconcile_state_on_startup()
                if not single_cycle:
                    background_tasks.append(asyncio.create_task(self._run_runtime_guards()))
                    background_tasks.append(asyncio.create_task(self._run_order_reconciliation()))
                    background_tasks.append(
                        asyncio.create_task(self._run_position_reconciliation())
                    )
                    if self._account_snapshot_supported:
                        background_tasks.append(
                            asyncio.create_task(
                                self._run_account_telemetry(self.config.account_telemetry_interval)
                            )
                        )
            else:
                logger.info("Dry-run: skipping startup reconciliation and background guard loops")

            await self.run_cycle()
            self.write_health_status(ok=True)
            self._check_config_updates()
            if not single_cycle and not self.config.dry_run:
                while self.running:
                    await asyncio.sleep(self.config.update_interval)
                    if self.running:
                        await self.run_cycle()
                        self.write_health_status(ok=True)
                        self._check_config_updates()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Bot error: {e}", exc_info=True)
            self.write_health_status(ok=False, error=str(e))
        finally:
            self.running = False
            for task in background_tasks:
                if not task.done():
                    task.cancel()
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)
            await self.shutdown()

    async def _run_runtime_guards(self):
        while self.running:
            try:
                await asyncio.to_thread(self.exec_engine.run_runtime_guards)
            except Exception as e:
                logger.error(f"Error in runtime guards: {e}", exc_info=True)
            await asyncio.sleep(60)

    def write_health_status(self, ok: bool, message: str = "", error: str = ""):
        status = {
            "ok": ok,
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "error": error,
        }
        status_file = RUNTIME_DATA_DIR / "perps_bot" / self.config.profile.value / "health.json"
        status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)

    async def shutdown(self):
        logger.info("Shutting down bot...")
        self.running = False
        # Stop WS streaming thread if running
        try:
            if hasattr(self, "_ws_stop") and self._ws_stop:
                self._ws_stop.set()
            if hasattr(self, "_ws_thread") and self._ws_thread and self._ws_thread.is_alive():
                self._ws_thread.join(timeout=2.0)
        except Exception as exc:
            logger.debug("Failed to stop WS thread cleanly: %s", exc, exc_info=True)

    async def _run_order_reconciliation(self, interval_seconds: int = 45):
        """Periodically reconcile local order store with exchange to prevent drift."""
        while self.running:
            try:
                # Build map of local open orders
                local_open = {o.order_id: o for o in self.orders_store.get_open_orders()}

                # Fetch all exchange orders and filter open ones
                try:
                    all_orders = await asyncio.to_thread(self.broker.list_orders)
                except Exception:
                    all_orders = []

                from bot_v2.features.brokerages.core.interfaces import OrderStatus as _OS

                open_statuses = {_OS.PENDING, _OS.SUBMITTED, _OS.PARTIALLY_FILLED}
                exch_open = {
                    o.id: o
                    for o in (all_orders or [])
                    if getattr(o, "status", None) in open_statuses
                }

                # Log count mismatch for visibility (no metrics store)
                if len(local_open) != len(exch_open):
                    logger.info(
                        f"Order count mismatch: local={len(local_open)} exchange={len(exch_open)}"
                    )

                # Update store with any exchange orders we don't have or that changed
                for oid, ex_order in exch_open.items():
                    try:
                        self.orders_store.upsert(ex_order)
                    except Exception as exc:
                        logger.debug(
                            "Failed to upsert exchange order %s during reconciliation: %s",
                            oid,
                            exc,
                            exc_info=True,
                        )

                # For any locally open order missing on exchange, fetch final status and upsert
                for oid, loc in local_open.items():
                    if oid not in exch_open:
                        try:
                            final = await asyncio.to_thread(self.broker.get_order, oid)
                            if final:
                                self.orders_store.upsert(final)
                                logger.info(f"Reconciled order {oid} → {final.status.value}")
                        except Exception as exc:
                            logger.debug(
                                "Unable to reconcile order %s: %s", oid, exc, exc_info=True
                            )
                            continue
            except Exception as e:
                logger.debug(f"Order reconciliation error: {e}", exc_info=True)
            await asyncio.sleep(interval_seconds)

    # Streaming helpers removed: REST-first market data via get_quote()

    def _get_strategy(self, symbol: str) -> BaselinePerpsStrategy:
        if self.config.profile == Profile.SPOT:
            strat = self._symbol_strategies.get(symbol)
            if strat is None:
                strat = BaselinePerpsStrategy(risk_manager=self.risk_manager)
                self._symbol_strategies[symbol] = strat
            return strat
        return self.strategy  # type: ignore[attr-defined]

    async def _apply_spot_filters(
        self, symbol: str, decision: Decision, position_state: dict[str, Any] | None
    ) -> Decision:
        rules = self._spot_rules.get(symbol)
        if not rules or decision.action != Action.BUY:
            return decision
        if position_state and Decimal(str(position_state.get("qty", "0"))) != Decimal("0"):
            return decision

        needs_data = False
        max_window = 0

        vol_cfg = rules.get("volatility_filter") if isinstance(rules, dict) else None
        if isinstance(vol_cfg, dict):
            window = int(vol_cfg.get("window", 0))
            if window > 0:
                needs_data = True
                max_window = max(max_window, window)

        volma_cfg = rules.get("volume_filter") if isinstance(rules, dict) else None
        if isinstance(volma_cfg, dict):
            window = int(volma_cfg.get("window", 0))
            if window > 0:
                needs_data = True
                max_window = max(max_window, window)

        momentum_cfg = rules.get("momentum_filter") if isinstance(rules, dict) else None
        if isinstance(momentum_cfg, dict):
            window = int(momentum_cfg.get("window", 0))
            if window > 0:
                needs_data = True
                max_window = max(max_window, window)

        trend_cfg = rules.get("trend_filter") if isinstance(rules, dict) else None
        if isinstance(trend_cfg, dict):
            window = int(trend_cfg.get("window", 0))
            if window > 0:
                needs_data = True
                max_window = max(max_window, window)

        if not needs_data:
            return decision

        candles = await self._fetch_spot_candles(symbol, max_window)
        if not candles:
            logger.debug("Insufficient candle data for %s; deferring entry", symbol)
            return Decision(action=Action.HOLD, reason="indicator_data_unavailable")

        closes = [_to_decimal(getattr(c, "close", 0)) for c in candles]
        volumes = [_to_decimal(getattr(c, "volume", 0)) for c in candles]
        highs = [_to_decimal(getattr(c, "high", 0)) for c in candles]
        lows = [_to_decimal(getattr(c, "low", 0)) for c in candles]

        # Volume confirmation
        if isinstance(volma_cfg, dict):
            window = int(volma_cfg.get("window", 0))
            multiplier = _to_decimal(volma_cfg.get("multiplier", 1))
            if window > 0:
                if len(volumes) < window + 1:
                    return Decision(action=Action.HOLD, reason="volume_filter_wait")
                recent = volumes[-(window + 1) : -1]
                avg_vol = _mean_decimal(recent)
                latest_vol = volumes[-1]
                if avg_vol <= Decimal("0") or latest_vol < avg_vol * multiplier:
                    logger.info("%s entry blocked by volume filter", symbol)
                    return Decision(action=Action.HOLD, reason="volume_filter_blocked")

        # RSI momentum
        if isinstance(momentum_cfg, dict):
            window = int(momentum_cfg.get("window", 0))
            overbought = _to_decimal(momentum_cfg.get("overbought", 70))
            oversold = _to_decimal(momentum_cfg.get("oversold", 30))
            if window > 0:
                if len(closes) < window + 1:
                    return Decision(action=Action.HOLD, reason="momentum_filter_wait")
                rsi = _rsi_from_closes(closes[-(window + 1) :])
                if rsi > oversold:
                    logger.info(
                        "%s entry blocked by momentum filter (RSI=%.2f)", symbol, float(rsi)
                    )
                    return Decision(action=Action.HOLD, reason="momentum_filter_blocked")

        # Trend strength
        if isinstance(trend_cfg, dict):
            window = int(trend_cfg.get("window", 0))
            min_slope = _to_decimal(trend_cfg.get("min_slope", 0))
            if window > 0:
                if len(closes) < window + 1:
                    return Decision(action=Action.HOLD, reason="trend_filter_wait")
                current_ma = _mean_decimal(closes[-window:])
                prev_ma = _mean_decimal(closes[-(window + 1) : -1])
                slope = (current_ma - prev_ma) / Decimal(window)
                if slope < min_slope:
                    logger.info(
                        "%s entry blocked by trend filter (slope=%.6f)", symbol, float(slope)
                    )
                    return Decision(action=Action.HOLD, reason="trend_filter_blocked")

        # Volatility filter (ATR)
        if isinstance(vol_cfg, dict):
            window = int(vol_cfg.get("window", 0))
            min_vol = _to_decimal(vol_cfg.get("min_vol", 0))
            max_vol = _to_decimal(vol_cfg.get("max_vol", 1))
            if window > 0:
                if len(closes) < window + 1:
                    return Decision(action=Action.HOLD, reason="volatility_filter_wait")
                atr_values: list[Decimal] = []
                prev_close: Decimal | None = None
                start_idx = max(len(closes) - window - 1, 0)
                for idx in range(start_idx, len(closes)):
                    if prev_close is None and idx > 0:
                        prev_close = closes[idx - 1]
                    tr = _true_range(highs[idx], lows[idx], prev_close)
                    atr_values.append(tr)
                    prev_close = closes[idx]
                atr = _mean_decimal(atr_values[-window:])
                if atr <= Decimal("0"):
                    return Decision(action=Action.HOLD, reason="volatility_filter_blocked")
                vol_pct = atr / closes[-1]
                if vol_pct < min_vol or vol_pct > max_vol:
                    logger.info(
                        "%s entry blocked by volatility filter (%.6f)", symbol, float(vol_pct)
                    )
                    return Decision(action=Action.HOLD, reason="volatility_filter_blocked")

        return decision

    async def _fetch_spot_candles(self, symbol: str, window: int) -> list[Any]:
        limit = max(window + 2, 10)
        try:
            candles = await asyncio.to_thread(
                self.broker.get_candles,
                symbol,
                "ONE_HOUR",
                limit,
            )
        except Exception as exc:
            logger.debug("Failed to fetch candles for %s: %s", symbol, exc, exc_info=True)
            return []
        if not candles:
            return []
        return sorted(
            candles,
            key=lambda c: getattr(c, "ts", getattr(c, "timestamp", datetime.utcnow())),
        )

    def _update_mark_window(self, symbol: str, mark: Decimal) -> None:
        with self._mark_lock:
            if symbol not in self.mark_windows:
                self.mark_windows[symbol] = []
            self.mark_windows[symbol].append(mark)
            max_size = max(self.config.short_ma, self.config.long_ma) + 5
            if len(self.mark_windows[symbol]) > max_size:
                self.mark_windows[symbol] = self.mark_windows[symbol][-max_size:]

    @staticmethod
    def _calculate_spread_bps(bid_price: Decimal, ask_price: Decimal) -> Decimal:
        try:
            mid = (bid_price + ask_price) / Decimal("2")
            if mid <= 0:
                return Decimal("0")
            return ((ask_price - bid_price) / mid) * Decimal("10000")
        except Exception:
            return Decimal("0")

    async def _run_position_reconciliation(self, interval_seconds: int = 90):
        """Periodically reconcile positions to detect drift against the last known snapshot.

        Emits a 'position_drift' metric when symbols, side, or sizes change compared to baseline.
        """
        while self.running:
            try:
                # Pull current positions from broker (source of truth)
                try:
                    positions = await asyncio.to_thread(self.broker.list_positions)
                except Exception:
                    positions = []

                current: dict[str, dict[str, Any]] = {}
                for p in positions:
                    try:
                        sym = getattr(p, "symbol", None)
                        if not sym:
                            continue
                        qty = getattr(p, "qty", Decimal("0"))
                        side = getattr(p, "side", "")
                        current[str(sym)] = {"qty": str(qty), "side": str(side)}
                    except Exception:
                        continue

                # Initialize baseline without emitting drift
                if not self._last_positions and current:
                    self._last_positions = current
                else:
                    # Compare against baseline
                    changes: dict[str, dict[str, Any]] = {}
                    # Detect new/changed symbols
                    for sym, data in current.items():
                        prev = self._last_positions.get(sym)
                        if (
                            not prev
                            or prev.get("qty") != data.get("qty")
                            or prev.get("side") != data.get("side")
                        ):
                            changes[sym] = {"old": prev or {}, "new": data}
                    # Detect removed symbols
                    for sym in list(self._last_positions.keys()):
                        if sym not in current:
                            changes[sym] = {"old": self._last_positions[sym], "new": {}}

                    if changes:
                        logger.info(f"Position changes detected: {len(changes)} updates")
                        try:
                            plog = _get_plog()
                            for sym, change in changes.items():
                                newd = change.get("new", {}) or {}
                                sided = newd.get("side")
                                qstr = newd.get("qty")
                                sizef = float(qstr) if qstr not in (None, "") else 0.0
                                plog.log_position_change(
                                    symbol=sym,
                                    side=str(sided) if sided is not None else "",
                                    size=sizef,
                                )
                        except Exception as exc:
                            logger.debug(
                                "Failed to log position change metric: %s", exc, exc_info=True
                            )
                        try:
                            self.event_store.append_metric(
                                bot_id=self.bot_id,
                                metrics={"event_type": "position_drift", "changes": changes},
                            )
                        except Exception as exc:
                            logger.exception("Failed to emit position drift metric: %s", exc)
                        self._last_positions = current
            except Exception as e:
                logger.debug(f"Position reconciliation error: {e}", exc_info=True)
            await asyncio.sleep(interval_seconds)

    async def _run_account_telemetry(self, interval_seconds: int = 300):
        """Periodically emit account snapshots (fees, limits, permissions)."""
        if not self._account_snapshot_supported:
            collector = getattr(self._collect_account_snapshot, "__func__", None)
            default_collector = getattr(type(self), "_collect_account_snapshot", None)
            if collector is default_collector:
                if self._supports_account_snapshot():
                    self._account_snapshot_supported = True
                else:
                    return
        while self.running:
            try:
                snapshot = await asyncio.to_thread(self._collect_account_snapshot)
                if snapshot:
                    try:
                        self.event_store.append_metric(
                            bot_id=self.bot_id,
                            metrics={"event_type": "account_snapshot", **snapshot},
                        )
                    except Exception as exc:
                        logger.debug(
                            "Failed to append account snapshot metric: %s", exc, exc_info=True
                        )
                    try:
                        output_path = (
                            RUNTIME_DATA_DIR
                            / "perps_bot"
                            / self.config.profile.value
                            / "account.json"
                        )
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with output_path.open("w") as fh:
                            json.dump(snapshot, fh, indent=2)
                    except Exception as exc:
                        logger.debug("Failed to write account snapshot: %s", exc, exc_info=True)
            except Exception as e:
                logger.debug(f"Account telemetry error: {e}", exc_info=True)
            await asyncio.sleep(interval_seconds)

    def _collect_account_snapshot(self) -> dict[str, Any]:
        """Gather fee schedule, limits, key permissions, and transaction summary."""
        snapshot: dict[str, Any] = {}
        try:
            snapshot.update(self.account_manager.snapshot(emit_metric=False))
        except Exception as exc:
            logger.debug("Failed to capture account manager snapshot: %s", exc, exc_info=True)
        try:
            snapshot["server_time"] = self.broker.get_server_time().isoformat()
        except Exception:
            snapshot.setdefault("server_time", None)
        snapshot["timestamp"] = datetime.now(UTC).isoformat()
        self._latest_account_snapshot = snapshot
        return snapshot
