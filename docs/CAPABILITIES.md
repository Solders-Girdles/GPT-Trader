# GPT-Trader Capability Map

A comprehensive guide to system capabilities, runtime flow, and where to make changes.

## System Overview

GPT-Trader is a Coinbase trading system supporting spot and CFM futures markets. It uses a vertical slice architecture with dependency injection via `ApplicationContainer`. The system executes configurable trading strategies with multi-layered risk guards, graceful degradation on failures, and comprehensive observability.

## Runtime Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI ENTRY                                       │
│  gpt-trader run --profile dev                                               │
│  cli/__main__.py → cli/commands/run.py                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONFIG & CONTAINER                                   │
│  cli/services.py::build_config_from_args()                                  │
│  Precedence: CLI > Profile > Env > Defaults                                 │
│                                    │                                         │
│  app/container.py::ApplicationContainer                                     │
│  ├── ConfigContainer (profiles, settings)                                   │
│  ├── ObservabilityContainer (health, secrets, notifications)                │
│  ├── PersistenceContainer (event store, orders store)                       │
│  ├── BrokerageContainer (Coinbase client, products)                         │
│  └── RiskValidationContainer (risk manager, validation tracker)             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TRADING BOT                                       │
│  features/live_trade/bot.py::TradingBot                                     │
│  Creates TradingEngine with CoordinatorContext                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRADING ENGINE                                      │
│  features/live_trade/engines/strategy.py::TradingEngine                     │
│                                                                              │
│  Background Tasks:                                                           │
│  ├── Main trading loop (_run_loop)                                          │
│  ├── Heartbeat service                                                       │
│  ├── Status reporter                 ┌──────────────────────────────────┐   │
│  ├── Health check runner ───────────▶│      OBSERVABILITY (continuous)  │   │
│  ├── WS health watchdog              │  src/gpt_trader/monitoring/health_checks.py │   │
│  └── System maintenance              │  src/gpt_trader/observability/tracing.py   │   │
│                                      │  src/gpt_trader/monitoring/metrics_exporter.py │   │
│  ┌─ RUNTIME GUARD SWEEP ──────────┐  │                                  │   │
│  │  features/live_trade/execution/│  │  • Health endpoints (/health)   │   │
│  │    guard_manager.py            │  │  • Prometheus metrics (/metrics)│   │
│  │  Periodic checks (not per-     │  │  • OpenTelemetry trace spans    │   │
│  │  order):                       │  │  • Status file updates          │   │
│  │  • Daily loss limit            │  └──────────────────────────────────┘   │
│  │  • Liquidation buffer          │                                         │
│  │  • Volatility circuit breaker  │                                         │
│  │  • API health monitoring       │                                         │
│  └────────────────────────────────┘                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRADING CYCLE (every 60s)                           │
│                                                                              │
│  1. Fetch positions & orders                                                 │
│  2. Calculate equity                                                         │
│  3. Track daily PnL (trigger reduce-only if limit breached)                 │
│  4. Batch fetch tickers                                                      │
│  5. Per-symbol loop:                                                         │
│     ├── Fetch ticker + candles                                               │
│     ├── Record price tick (crash recovery)                                   │
│     └── Strategy.decide() → Decision(action, reason, confidence)            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                          (if BUY/SELL action)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PRE-TRADE GUARD STACK                                   │
│  Canonical entry (live loop): TradingEngine._validate_and_place_order()     │
│  External entry: TradingEngine.submit_order()                               │
│  (executed synchronously before each order)                                 │
│                                                                              │
│  ┌─ Degradation Gate ────────────────────────────────────────────────────┐  │
│  │  DegradationState.is_paused(symbol)                                   │  │
│  │  → Reject if paused (graceful degradation)                            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│  ┌─ Position Sizing ─────────────────────────────────────────────────────┐  │
│  │  Kelly criterion + fraction of account                                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│  ┌─ Security Validation ─────────────────────────────────────────────────┐  │
│  │  security/security_validator.py                                       │  │
│  │  Hard limits: max position, max leverage, max daily loss              │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│  ┌─ Risk Manager Pre-Trade ──────────────────────────────────────────────┐  │
│  │  features/live_trade/risk/manager/                                    │  │
│  │  Leverage caps, exposure limits, MMR projection                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│  ┌─ Mark Staleness Guard ────────────────────────────────────────────────┐  │
│  │  Reject if price data > 120s old (allow reduce-only if configured)   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│  ┌─ Order Validator ─────────────────────────────────────────────────────┐  │
│  │  features/live_trade/execution/validation.py                          │  │
│  │  Exchange rules, slippage guard, order preview                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                          (all guards pass)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ORDER EXECUTION                                     │
│  features/live_trade/execution/order_submission.py                          │
│                                    │                                         │
│  broker.place_order(symbol, side, type, quantity)                           │
│                                    │                                         │
│  features/brokerages/coinbase/client/orders.py                              │
│  → REST API POST to Coinbase                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          POST-EXECUTION                                      │
│  • Track order ID for auditing                                              │
│  • Send success notification                                                 │
│  • Record in status reporter (TUI display)                                  │
│  • Persist trade event to EventStore                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Capability Matrix

| Capability | Primary Modules | Key Classes/Functions | Config Flags | Health/Metrics | Degradation Behavior |
|------------|-----------------|----------------------|--------------|----------------|---------------------|
| **Configuration + Feature Flags** | `app/config/bot_config.py`, `features/live_trade/risk/config.py` | `BotConfig`, `RiskConfig`, `from_env()` | See [FEATURE_FLAGS.md](FEATURE_FLAGS.md) | Config load success/warnings | Sync warnings on mismatch |
| **Configuration Guardian** | `app/config/`, `src/gpt_trader/monitoring/configuration_guardian/` | `ProfileLoader`, `DriftDetector` | Profile-specific overrides | Drift detection, validation errors | Blocks invalid config at startup |
| **Trading Decisioning** | `features/live_trade/strategies/` | `BaselinePerpsStrategy.decide()`, `MeanReversionStrategy.decide()`, `EnsembleStrategy.decide()` | `strategy_type`, `kill_switch_enabled` | Decision confidence, indicator values | Kill switch stops all decisions |
| **Pre-trade Validation** | `features/live_trade/engines/strategy.py`, `security/security_validator.py` | `TradingEngine._validate_and_place_order()`, `OrderValidator`, `SecurityValidator` | `enable_order_preview`, `ORDER_PREVIEW_FAIL_CLOSED` | Validation pass/fail counts | Escalates to reduce-only after N failures |
| **Order Execution** | `features/live_trade/engines/strategy.py`, `features/live_trade/execution/order_submission.py` | `OrderSubmitter.submit_order()`, `BrokerExecutor.execute_order()` | `dry_run`, `mock_broker` | Order latency, success rate | Pauses symbol on repeated failures |
| **Risk & Degradation** | `features/live_trade/risk/manager/`, `features/live_trade/degradation.py` | `LiveRiskManager`, `DegradationState` | `reduce_only_mode`, `RISK_DAILY_LOSS_LIMIT_PCT` | Daily PnL %, leverage ratio | Reduce-only mode, symbol pause |
| **Runtime Guards** | `features/live_trade/execution/guards/`, `features/live_trade/execution/guard_manager.py` | `GuardManager`, individual guards | `RISK_*` thresholds | Guard trip counts, cooldowns | Triggers reduce-only or pause |
| **Streaming (WS)** | `features/brokerages/coinbase/ws.py` | `CoinbaseWebSocket`, `WSHealthWatchdog` | `PERPS_ENABLE_STREAMING`, `RISK_WS_*` | Last message age, heartbeat age | Pause on stale data, reconnect backoff |
| **Observability** | `src/gpt_trader/monitoring/health_checks.py`, `src/gpt_trader/observability/tracing.py`, `src/gpt_trader/monitoring/metrics_exporter.py` | `HealthState`, `init_tracing()`, `MetricsExporter` | `GPT_TRADER_OTEL_ENABLED`, `GPT_TRADER_METRICS_ENDPOINT_ENABLED` | Health endpoints, trace spans | N/A |
| **Preflight Diagnostics** | `preflight/` | `PreflightCheck`, `preflight.cli.main()` | N/A | Check pass/fail | Blocks startup on critical failures |
| **TUI Monitoring** | `tui/` | `TraderApp`, screen/widget hierarchy | N/A | Real-time display | Visual degradation indicators |
| **Testing Harness** | `tests/integration/`, `tests/property/` | Chaos tests, property invariants | N/A | Test coverage | N/A |

## Where to Change Things

| Intent | File(s) to Modify |
|--------|-------------------|
| Add a new trading strategy | `features/live_trade/strategies/` + register in `factory.py` |
| Add a new runtime guard | `features/live_trade/execution/guards/` + register in `guard_manager.py` |
| Add a new pre-trade check | `features/live_trade/engines/strategy.py::_validate_and_place_order()` |
| Submit order programmatically | `TradingEngine.submit_order()` — external entry point (wraps `_validate_and_place_order`) |
| Change order execution flow | `features/live_trade/engines/strategy.py` (canonical path) |
| Add a new env config flag | `app/config/bot_config.py` + `.env.template` + `docs/FEATURE_FLAGS.md` |
| Add a new health check | `src/gpt_trader/monitoring/health_checks.py` |
| Modify degradation behavior | `features/live_trade/degradation.py` |
| Add Coinbase API endpoint | `features/brokerages/coinbase/client/` (appropriate mixin) |
| Update TUI display | `tui/widgets/` or `tui/screens/` |

**Note:** `features/live_trade/execution/engine.py::LiveExecutionEngine` is deprecated. Use the TradingEngine guard stack (live loop `_validate_and_place_order`, external `submit_order`).

## Intentional Guard-Stack Bypasses

The canonical order path routes through `TradingEngine._validate_and_place_order()` (the live loop), with `TradingEngine.submit_order()` as the external wrapper. The following locations intentionally bypass guards:

| Location | Purpose | Justification |
|----------|---------|---------------|
| `features/live_trade/bot.py::flatten_and_stop()` | Emergency position closure | Must succeed even when guards would block (e.g., during risk trip) |
| `features/optimize/` (batch_runner, walk_forward) | Backtesting/optimization | Uses simulated broker, not production |
| `features/live_trade/execution/router.py::execute()` | Legacy sync path | **Deprecated** — emits warning, used in tests only |

All other `broker.place_order` calls are either:
- **Internal to the canonical path** (TradingEngine → broker)
- **Broker implementation** (CoinbaseRestService internals)

## Related Documentation

- **Feature Flags**: [docs/FEATURE_FLAGS.md](FEATURE_FLAGS.md) - Configuration precedence and canonical sources
- **Reliability**: [docs/RELIABILITY.md](RELIABILITY.md) - Resilience patterns and failure handling
- **Architecture**: [docs/ARCHITECTURE.md](ARCHITECTURE.md) - System design and vertical slices

## Out of Scope / Future Work

- **Multi-exchange support** - Currently Coinbase-only; broker interface exists but not abstracted
- **ML-based strategies** - Intelligence module exists but not production-ready
- **Distributed deployment** - Single-process design; no horizontal scaling
- **Historical backtesting integration** - Backtesting exists but separate from live engine
- **Advanced order types** - Currently market orders only in live trading
