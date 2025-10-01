# System Reality Check

---
last-updated: 2025-09-30
---

This document captures what's actually working versus what's scaffolded or experimental.

## Actually Working ✅

### Core Trading
- REST API connectivity via Coinbase client (adapter wired; mocked in dev)
- Spot and perpetual order placement (market/limit) through `LiveExecutionEngine`
- Dev profile mock broker exposes top-ten USD spot symbols (`BTC`, `ETH`, `SOL`, `XRP`, `LTC`, `ADA`, `DOGE`, `BCH`, `AVAX`, `LINK`)
- Spot risk defaults centralized in `config/risk/spot_top10.json` (auto-loaded for dev/demo/spot profiles)
- Reduce-only risk enforcement (pre-trade validation)

### Streaming & WebSocket
- Coinbase adapter streaming methods: `stream_trades`, `stream_orderbook`, `stream_user_events`
- Dev profile: Mock broker provides `stream_trades` with random-walk ticks to drive mark prices
- Live profile: Adapter exposes streams; consumer threads run successfully

### State Management & Persistence
- **EventStore**: Append-only JSONL audit trail with thread-safe concurrent writes
- **OrdersStore**: Order persistence and retrieval
- **ConfigStore**: Configuration management
- **Recovery orchestrator**: Comprehensive recovery infrastructure with failure detection, retry logic, and automatic recovery
- **State management**: Backup, checkpointing, and state tier management with full test coverage (5,911 lines across persistence + state test suites)

### Risk & Guards
- Slippage checks and mark staleness detection
- Daily PnL stops and liquidation buffer monitoring
- Volatility circuit breakers
- Correlation checks
- Per-symbol notional caps (configurable via risk config files)

## Partially Working ⚠️

### Position Tracking
- Mock broker: Basic side/quantity/entry updates working
- Unrealized PnL derived in runtime guards
- Full multi-leg position tracking needs additional testing

### WebSocket User Events
- Infrastructure present and streams connected
- User event handling (fills, order updates) functional but minimal telemetry
- Full order lifecycle tracking via WebSocket under development

## Not Working ❌

### Mock Broker Limitations
- Funding rate accrual in mock (full accrual only in Coinbase adapter)
- Partial fill handling (mock executes market fills immediately)
- Order modification/amend flows beyond cancel

### Future Enhancements
- Real-time P&L updates via WebSocket (currently poll-based)
- Advanced order types (stop-loss, trailing stops)
- Multi-portfolio position aggregation

## Verification Path

```bash
# Full spot regression suite (121 test files, 100% pass expected)
poetry run pytest -q

# Test discovery
poetry run pytest --collect-only

# Specific component tests
poetry run pytest tests/unit/bot_v2/persistence/ -v
poetry run pytest tests/unit/bot_v2/state/recovery/ -v
```

## Test Coverage Summary

- **Persistence layer**: 961 lines of tests across EventStore, OrdersStore, ConfigStore
- **State management**: 4,950 lines covering recovery (1,854 lines), backup, checkpointing, and state tiers
- **Combined persistence + state**: 5,911 lines of comprehensive test coverage
- **Coinbase adapter**: Unit tests for REST, WebSocket, and authentication
- **Risk guards**: Pre-trade validation and runtime monitoring tests
- **Total active tests**: 121 test files (see `pytest --collect-only`)

---

*For production readiness assessment, see [guides/production.md](../guides/production.md)*
