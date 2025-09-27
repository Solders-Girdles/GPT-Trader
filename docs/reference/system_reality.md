# System Reality Check

This document captures what’s actually working versus what’s scaffolded.

## Actually Working ✅
- REST API connectivity via Coinbase client (adapter wired; mocked in dev)
- Basic spot and perpetual order placement (market/limit) through `LiveExecutionEngine`
- Dev profile mock broker now exposes the top-ten USD spot symbols (`BTC`, `ETH`, `SOL`, `XRP`, `LTC`, `ADA`, `DOGE`, `BCH`, `AVAX`, `LINK`) so default smoke tests trade across a broader universe
- Spot risk defaults are centralized in `config/risk/spot_top10.json` (auto-loaded for dev/demo/spot), giving each asset tuned notional caps and leverage=1
- Reduce-only risk enforcement (pre-trade validation)
- Coinbase adapter streaming (`stream_trades`, `stream_orderbook`, `stream_user_events`)

## Partially Working ⚠️
- WebSocket streaming integration in bot:
  - Dev: Mock broker provides `stream_trades` (random-walk ticks) to drive marks
  - Real: Adapter exposes streams; consumer threads run, but full user-event handling is minimal
- Position tracking (mock): basic side/qty/entry updates; unrealized PnL derived in runtime guards
- Risk guards: slippage and staleness checks are present; funding/liq buffers are simplified

## Not Working ❌
- Funding rate accrual in mock; full accrual lives only in Coinbase adapter internals
- Durable state recovery after restart: OrdersStore/EventStore exist, but full reconciliation paths need hardening and tests
- Order modification/amend flows beyond cancel
- Partial fill handling in mock (market fills are immediate)

## Verification Path
1. `poetry run pytest -q` exercises the full spot regression suite (mock trades, risk guards, orchestration glue)
2. Add sandbox integration once credentials and environment are stable (networked tests remain opt-in)
