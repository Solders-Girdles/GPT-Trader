# Perpetual Futures Trading Plan (Template)

Use this one-pager before trading live. Keep it specific and operational. If a field is N/A, state why.

## 1) Objective & Scope
- Objective: e.g., grow equity 2–4% monthly with ≤10% max drawdown
- Instruments: Coinbase perps only — `BTC-PERP`, `ETH-PERP`, `SOL-PERP`, `XRP-PERP`
- Style: e.g., intraday trend-following; no overnight holdings until X
- Time windows (UTC): e.g., 13:00–17:00, Mon–Fri; avoid major news

## 2) Eligibility & Accounts
- Eligibility confirmed (jurisdiction, derivatives access): Yes/No
- Derivatives wallet enabled and funded: Yes/No
- Collateral: e.g., USDC only; min operational balance: $X
- Security: hardware 2FA, withdrawal allowlist, passkeys: Yes/No

## 3) Contract Specs (per symbol)
- Tick size / min size: e.g., 0.01 / 0.001 BTC
- Leverage tiers (init/maint margin): summarize key tiers
- Funding rate cadence & caps: e.g., hourly, typical range bps
- Liquidation: partial/whole, ADL rules: note specifics

## 4) Strategy Definition
- Setup: what market condition qualifies (trend/volatility/filters)
- Trigger: exact entry condition (signal/threshold/confirmation)
- Invalidation: condition that makes the idea wrong
- Initial stop: formula and placement (ATR, structure, fixed bps)
- Profit exit: target or trailing rule; scale-out rules if any
- No-trade filters: time, spread, volatility, events

## 5) Sizing & Risk
- Per-trade risk: e.g., 0.25–0.50% of equity
- Position size formula: risk / stop distance; min notional guard
- Max leverage: global and per-symbol caps; time-of-day 10x window (set exact UTC hours)
- Exposure caps: max symbol % and total % of equity
- Daily loss limit: e.g., 1–2% of equity or fixed $X
- Drawdown rules: pause/step-down at −Y% or Z consecutive losses

## 6) Order Policy
- Entry type: default `LIMIT post-only` or `MARKET` w/ slippage cap
- Reductions: `reduce-only` for exits by default: Yes/No
- Time-in-force: GTC/IOC/FOK
- OCO/brackets: usage and mapping to platform

## 7) Costs & Constraints
- Fees: maker/taker tier, expected mix
- Funding: expected hold time × typical bps impact
- Slippage: expected vs target; depth constraints for size

## 8) Monitoring & Alerting
- Metrics: PnL, exposure, leverage, funding, order rejects
- Alerts: loss breach, stale data, connection loss, position stuck
- Dashboards/logs locations

## 9) Operations
- Pre-trade checklist: env keys loaded, risk params, symbols, health
- Downtime plan: what to do if Coinbase is unavailable
- Kill switch: how to flatten/disable (env/toggle/procedure)
- Backup: secondary venue or hedging method (if any)

## 10) Validation & Review
- Backtest/paper stats: expectancy, hit-rate, avg win/loss, worst DD
- Live trial gates: size up only after N trades and adherence ≥X%
- Review cadence: daily notes; weekly metrics review; monthly changes

---

Non‑advice disclaimer: Perpetual futures are high risk; you may lose all posted collateral. This template is for operational readiness and risk hygiene, not financial advice.
