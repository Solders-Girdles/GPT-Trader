# System Architecture

---
status: current
last-updated: 2025-03-01
---

## Current State

GPT-Trader V2 is a production-ready Coinbase **spot** trading system that retains future-ready perpetuals logic. Perps execution remains disabled in live environments until Coinbase grants INTX access, but the architecture keeps those paths compiled and testable.

## Trading Capabilities Matrix

| Environment | Products | Authentication | API Version | WebSocket | Use Case |
|------------|----------|----------------|-------------|-----------|----------|
| **Production (default)** | Spot (BTC-USD, ETH-USD, …) | HMAC | Advanced v3 | Real-time | Live trading |
| **Production (perps)** | Perpetuals (INTX-gated) | CDP (JWT) + `COINBASE_ENABLE_DERIVATIVES=1` | Advanced v3 | Real-time | Future activation |
| **Sandbox** | Not used (API diverges) | — | — | — | Paper/mock via `PERPS_PAPER=1` |

## Component Architecture

### Vertical Slice Design

The system is organized into vertical feature slices under `src/bot_v2/features/`. Production-critical slices (e.g., `live_trade`, `brokerages`, `position_sizing`) ship with full test coverage, while research/demo slices (`backtest`, `ml_strategy`, `market_regime`, `monitoring_dashboard`) are tagged `__experimental__` and excluded from the core trading loop. The former workflow engine was removed; recover it from repository history if you need a reference.

```
src/bot_v2/features/
├── live_trade/          # Production trading engine
├── paper_trade/         # Simulated trading
├── backtest/           # Historical testing
├── ml_strategy/        # ML-driven strategy selection
├── market_regime/      # Market condition detection
├── position_sizing/    # Kelly Criterion sizing
├── adaptive_portfolio/ # Tier-based portfolio management
├── analyze/           # Market analysis
├── optimize/          # Parameter optimization
├── monitor/           # System monitoring
└── data/             # Data management
```

### Key Design Principles

1. **Slice Isolation**: Production slices limit cross-dependencies; experimental ones stay sandboxed.
2. **Token Awareness**: Documentation highlights slice entry points so agents can load only what they need.
3. **Type Safety**: Shared interfaces defined in `features/brokerages/core/interfaces.py`.
4. **Environment Separation**: `perps_bot` normalizes symbols to spot unless INTX derivatives access is detected.

### Orchestration Infrastructure

- `orchestration/session_guard.py` and `orchestration/market_monitor.py` encapsulate trading window enforcement and market-data freshness so `perps_bot` stays focused on orchestration glue.
- `features/live_trade/indicators.py`, `features/live_trade/risk_calculations.py`, and `features/live_trade/risk_runtime.py` centralize indicator math plus leverage/MMR/risk guard helpers, giving strategies and the risk manager shared, tested primitives.
- `orchestration/configuration.py` centralizes profile-aware defaults (`BotConfig`, `ConfigManager`) and is now covered by unit tests (`tests/unit/bot_v2/orchestration/test_configuration.py`).
- `orchestration/service_registry.py` provides an explicit container for runtime dependencies so the main bot can accept a prepared bundle instead of instantiating stores/brokers inline. Future phases will wire this into the CLI bootstrapper.
- Legacy status reports that used to live under `src/bot_v2/*.md` were removed;
  pull them from repository history if you need to review them.
- Historical V1/V2 integration and system tests depending on the legacy `bot.*` package lived under `archived/legacy_tests/` before the cleanup. Recover them from git history if you need a reference. The active pytest suite now focuses exclusively on the `bot_v2` stack and passes via `poetry run pytest`.

## What's Actually Working

### ✅ Fully Operational
- Coinbase spot trading via Advanced Trade (REST/WebSocket); dev profile defaults to the enhanced `MockBroker` and can be pointed at live APIs with `SPOT_FORCE_LIVE=1`
- Order placement/management through `LiveExecutionEngine`
- Account telemetry snapshots and cycle metrics persisted for monitoring
- Runtime safety rails: daily loss guard, liquidation buffer enforcement, mark staleness detection, volatility circuit breaker, correlation checks
- 446 active tests selected at collection time (`poetry run pytest --collect-only`)

### ⚠️ Partially Working / Future Activation
- Perpetual futures execution: code paths compile and tests run, but live trading remains disabled without INTX
- Advanced WebSocket user-event handling: baseline support exists; enrichment/backfill still in progress
- Durable restart state (OrdersStore/EventStore) needs production hardening

### ❌ Not Yet Implemented
- Funding rate accrual in mock broker
- Order modification/amend flows beyond cancel
- Partial fill handling in mock (market fills remain immediate)

## Trading Profiles

### Development Profile
```yaml
profile: dev
broker: mock
positions: tiny (0.001 BTC)
logging: verbose
fills: deterministic
universe: [BTC, ETH, SOL, XRP, LTC, ADA, DOGE, BCH, AVAX, LINK]
```

### Canary Profile
```yaml
profile: canary
broker: coinbase
positions: 0.01 BTC max
daily_loss_limit: $10
trading_window: 14:00-15:00 UTC
circuit_breakers: multiple
```

### Production Profile
```yaml
profile: prod
broker: coinbase
positions: full sizing
risk_limits: production
monitoring: real-time
```

## Risk Management

### Pre-Trade Validation
- Position size limits
- Margin requirement checks
- Impact cost validation (<50bps)
- Reduce-only mode enforcement

> Spot profiles load `config/risk/spot_top10.json` by default, enforcing
> per-symbol notional caps and leverage=1 across the top-ten USD markets.

### Runtime Guards
- Daily loss limits
- Error rate monitoring (>50% triggers shutdown)
- Stale data detection (30s timeout)
- Drawdown protection (10% max)
- Guard instrumentation raises structured `RiskGuard*Error` exceptions: recoverable failures emit `risk.guards.<name>.recoverable_failures` counters and log warnings, while critical failures escalate to reduce-only mode and emit `risk.guards.<name>.critical_failures`

### Circuit Breakers
- Consecutive loss protection
- Volatility spike detection
- Liquidity monitoring
- API error thresholds

## Performance & Observability

- **Cycle Metrics**: persisted to `var/data/perps_bot/<profile>/metrics.json` and exposed via Prometheus exporter (`scripts/monitoring/export_metrics.py`)
- **Account Snapshots**: periodic telemetry via `CoinbaseAccountManager` with fee/limit tracking
- **System Footprint**: bot process typically <50MB RSS with sub-100ms WebSocket latency in spot mode
- **Test Discovery**: 455 collected / 446 selected / 9 deselected

## Verification Path

1. **Regression Suite**: `poetry run pytest -q`
2. **Smoke Test**: `poetry run perps-bot --profile dev --dev-fast`
3. **Validation**: `python scripts/validation/verify_core.py --check all`

## Dependencies

### Core Requirements
- Python 3.12+
- Poetry 1.0+
- Coinbase Advanced Trade API (for perpetuals)
- CDP API keys (for production)

### Key Libraries
- `coinbase-advanced-py` - Official Coinbase SDK
- `pandas` - Data manipulation
- `scikit-learn` - ML models
- `websocket-client` - Real-time data
- `pydantic` - Data validation

## Future Roadmap

### Near Term (Q1 2025)
- [ ] Full WebSocket user event handling
- [ ] Durable state recovery
- [ ] Order modification flows
- [ ] Partial fill handling

### Medium Term (Q2 2025)
- [ ] Multi-exchange support
- [ ] Advanced order types
- [ ] Portfolio-level risk management
- [ ] Real-time ML adaptation

### Long Term (2025+)
- [ ] Distributed execution
- [ ] Cross-exchange arbitrage
- [ ] Options integration
- [ ] Institutional features

---

*For implementation details, see [Trading Logic - Perpetuals](reference/trading_logic_perps.md)*
