# P&L Calculation Differences: Our System vs Coinbase

---
status: current
---

This document describes where P&L is computed in our code and the known reasons
our numbers can differ from what Coinbase reports through their API.

> **Implementation status.** Live P&L tracking (realized + unrealized) is
> implemented in `PnLService`. Several reconciliation features described in
> older versions of this doc are **not built**: there is no funding field on
> live positions, fees are not folded into P&L, and there is no automated
> P&L-value reconciliation against Coinbase. Those are called out as **Gaps**
> below so they are not mistaken for existing behavior.

## Where P&L lives in code

| Concern | Code |
|---------|------|
| Live realized/unrealized P&L | `PnLService` — `src/gpt_trader/features/brokerages/coinbase/rest/pnl_service.py` |
| Position state (qty, side, entry) | `PositionStateStore` / `PositionState` — `src/gpt_trader/features/brokerages/coinbase/rest/position_state_store.py`, `.../coinbase/utilities.py` |
| Mark prices | `MarketDataService.get_mark()` — `src/gpt_trader/features/brokerages/coinbase/market_data_service.py` |
| Position drift detection | `PositionReconciler` — `src/gpt_trader/monitoring/system/positions.py` |
| CFM futures positions (broker) | `cfm_position()` / `cfm_positions()` — `src/gpt_trader/features/brokerages/coinbase/client/portfolio.py` |
| Backtest funding simulation | `FundingPnLTracker` — `src/gpt_trader/backtesting/simulation/funding_tracker.py` |

### How `PnLService` computes P&L

`process_fill_for_pnl(fill)` updates position state from each fill:

- **Increasing a position** recalculates a weighted-average entry price:
  ```python
  total_cost = (position.quantity * position.entry_price) + (size * price)
  position.entry_price = total_cost / (position.quantity + size)
  ```
- **Reducing a position** realizes P&L on the closed portion:
  ```python
  pnl = (price - position.entry_price) * close_quantity
  if position.side == "short":
      pnl = -pnl
  position.realized_pnl += pnl
  ```

`get_position_pnl(symbol)` computes unrealized P&L from the current mark:

```python
mark = market_data.get_mark(symbol)            # falls back to entry_price if unavailable
upnl = (mark - position.entry_price) * position.quantity
if position.side == "short":
    upnl = -upnl
```

`get_portfolio_pnl()` aggregates realized + unrealized across all tracked symbols.

> **Position flips (long → short in one fill) are handled simplistically.**
> `process_fill_for_pnl` only reduces the existing position down to zero; it does
> not open the opposite side with the remaining quantity. Treat flips as
> incompletely modeled until this is hardened.

## Why our numbers differ from Coinbase

### 1. Funding payments (CFM/derivatives)

- **Our system:** live positions (`core.account.Position`) have **no funding
  field**. Funding is only modeled in backtests via `FundingPnLTracker`. Live
  realized P&L from `PnLService` therefore excludes funding entirely.
- **Coinbase:** CFM endpoints may fold funding into `realized_pnl`.
- **Consequence:** for derivatives, our realized P&L and Coinbase's can diverge
  by accumulated funding. Spot trading (the active mode) is unaffected.

### 2. Fees (Gap — not implemented)

- **Our system:** trading fees are **not** included in P&L. `PnLService` works
  purely from fill price/size.
- **Coinbase:** maker/taker fees reduce realized P&L.
- **Consequence:** our P&L reads higher than Coinbase's by roughly the fees paid.

### 3. Mark price timing

- **Our system:** unrealized P&L uses the latest mark from `MarketDataService`,
  which can lag if the feed is stale. When no mark is available, `PnLService`
  falls back to `entry_price` (unrealized P&L of zero).
- **Coinbase:** uses real-time marks from its matching engine.
- **Best practice:** when comparing, pull Coinbase's mark and use it for both sides.

### 4. Weighted-average entry price

Both sides use the same weighted-average formula (see code above). Differences
here are limited to decimal rounding (see below).

### 5. Decimal precision

- **Our system:** uses Python `Decimal` throughout.
- **Coinbase:** returns strings we convert to `Decimal`; values may already be
  rounded to 2–8 places.
- **Tolerance for comparisons:** `max(0.01% of value, $0.10)`.

## Reconciliation: what actually runs

The only automated reconciliation today is **position drift detection**, not
P&L-value reconciliation:

- `PositionReconciler.run()` periodically (default every 90s) calls
  `broker.list_positions()` and diffs quantity/side against the cached
  `runtime_state.last_positions`.
- On change it logs and emits a `position_drift` event via the EventStore.

There is **no** automated job that compares our realized/unrealized P&L against
Coinbase-reported values and alerts on tolerance breaches. (A historical
reconciliation helper was removed during the 2025 cleanup; recover it from git
history if that capability is reintroduced.) For a manual check, compare
`PnLService.get_portfolio_pnl()` output against Coinbase, accounting for the
funding and fee gaps above.

### Relevant Coinbase endpoints

| Purpose | Endpoint |
|---------|----------|
| CFM positions | `GET /api/v3/brokerage/cfm/positions/{product_id}` (via `cfm_position()`) |
| Fills (rebuild history) | `GET /api/v3/brokerage/orders/historical/fills` |
| Portfolio aggregate | `GET /api/v3/brokerage/portfolios/{portfolio_uuid}` |

## Gaps / not yet built

- Funding on live positions (currently backtest-only).
- Fee-adjusted P&L.
- Automated P&L-value reconciliation against Coinbase with alerting.
- Complete position-flip handling in `process_fill_for_pnl`.

Treat the above as design intent, not current behavior.
