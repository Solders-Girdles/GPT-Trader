# Coinbase Perpetuals – Go‑Live Checklist

Short, actionable readiness checklist. Complete each section before increasing size.

## 1) Eligibility & Security
- Jurisdiction allows perps; derivatives wallet enabled
- Instruments: using only BTC/ETH/SOL/XRP perps (others blocked)
- 2FA with hardware key/passkey; SMS 2FA disabled
- Withdrawal allowlist + delay enabled; password manager in place
- Small test transfer to derivatives wallet completed

## 2) Environment & Keys
- `.env` copied from `config/environments/.env.template` and filled
- `BROKER=coinbase`, `COINBASE_SANDBOX=0`, `COINBASE_ENABLE_DERIVATIVES=1`
- Advanced Trade JWT present: `COINBASE_PROD_CDP_API_KEY` + `COINBASE_PROD_CDP_PRIVATE_KEY`
- Optional HMAC keys present for spot/dev smoke tests
- Secrets never committed; local `.env` git‑ignored

## 3) Risk Configuration
- Global caps: `RISK_MAX_LEVERAGE`, `RISK_MAX_EXPOSURE_PCT`, per‑symbol if needed
- Loss limits: `RISK_DAILY_LOSS_LIMIT` set to small fixed $ amount
- Per‑symbol caps: `RISK_LEVERAGE_MAX_PER_SYMBOL`, `RISK_MAX_NOTIONAL_PER_SYMBOL`
- Safety: `RISK_REDUCE_ONLY_MODE`=1 for canary; `RISK_KILL_SWITCH_ENABLED` initially 1 until supervised start
- Staleness/volatility: `RISK_MAX_MARK_STALENESS_SECONDS`, volatility CBs if used
- Time‑of‑day leverage (10x window): set `RISK_DAYTIME_START_UTC`/`RISK_DAYTIME_END_UTC`; apply `RISK_DAY_*`=10, `RISK_NIGHT_*`=lower

## 4) Dry Runs
- Dev mock: `python -m gpt_trader.cli run --profile dev --dry-run --symbols BTC-PERP` runs one cycle
- Canary smoke (reduce-only): `python -m gpt_trader.cli run --profile canary --symbols BTC-PERP --reduce-only` with tiny limits
- Logs created under `var/logs/` and event store under `var/data/coinbase_trader/{profile}`

## 5) Funding & Costs Sanity
- Maker/taker tier confirmed; fee schedule understood
- Funding rate mechanics understood; expected hold times validated
- Slippage tested with limit orders around top‑of‑book depth

## 6) Operational Playbook
- Kill switch procedure tested: set `RISK_KILL_SWITCH_ENABLED=1` and verify blocks
- Cancel‑all and close‑all procedures rehearsed
- Downtime plan documented (flatten or hedge elsewhere)
- Alerts configured (logs + optional Slack/PagerDuty)

## 7) First Live Steps
- Start with minimum notional size allowed by product specs
- Trade window restricted (e.g., 1 hour/day canary)
- Journal each trade (reason, sizes, PnL, variance from plan)
- Scale only after N trades and adherence ≥ X%

---

Commands reference
- Runner: `python -m gpt_trader.cli run --profile canary --symbols BTC-PERP --reduce-only`
- Kill switch: export `RISK_KILL_SWITCH_ENABLED=1` (then restart) to halt; set `0` to resume
- Logs: `var/logs/perps_trading.log`, JSONL at `var/logs/perps_trading.jsonl`
- Health: `var/data/coinbase_trader/{profile}/health.json`

Non‑advice disclaimer: Perpetual futures are high risk; you may lose all posted collateral.
