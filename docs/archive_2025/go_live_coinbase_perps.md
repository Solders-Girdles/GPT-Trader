# Coinbase Perpetuals: Go‑Live Readiness

This repo already includes the key building blocks for safe, real‑money trading on Coinbase perpetual futures. Use this checklist and the built‑in tools to verify you’re truly ready.

## Quick Path

- Prepare `.env` from `.env.template` with your credentials and risk limits.
- Run a readiness audit:
  - `python -m bot_v2.simple_cli readiness --symbol BTC-PERP`
- Run a dry‑run single cycle to verify orchestration:
  - `python -m bot_v2.simple_cli perps --profile dev --dry-run --dev-fast`
- Probe live connectivity (non‑destructive):
  - `bash scripts/perps_live_preflight_check.sh`
- When ready, start with tiny size in `demo` or `canary` profile.

## Configuration

- Core env (`.env`):
  - `BROKER=coinbase`
  - `COINBASE_API_MODE=advanced`
  - `COINBASE_SANDBOX=0`
  - `COINBASE_ENABLE_DERIVATIVES=1`
  - `COINBASE_PROD_CDP_API_KEY=...`
  - `COINBASE_PROD_CDP_PRIVATE_KEY=...`
- Risk env (conservative starting point):
  - `RISK_MAX_LEVERAGE=3`
  - `RISK_MIN_LIQUIDATION_BUFFER_PCT=0.15`
  - `RISK_DAILY_LOSS_LIMIT=50` (USD)
  - `RISK_MAX_EXPOSURE_PCT=0.5`
  - `RISK_MAX_POSITION_PCT_PER_SYMBOL=0.2`

## Built‑in Tools

- Readiness audit: `python -m bot_v2.simple_cli readiness --symbol BTC-PERP`
  - Verifies env, attempts a broker connection, lists perps, prints a quote, surfaces risk config and fee tier.
- Break‑even calculator: `python -m bot_v2.simple_cli breakeven --side long --entry 50000 --symbol BTC-PERP`
  - Computes fee‑aware minimum exit price (assumes worst‑case taker+taker, plus safety bps).
- Production preflight (shell): `bash scripts/perps_live_preflight_check.sh`
- Capability probe (Python): `python scripts/probe_capabilities.py --live`
- Diagnostics (perps discovery): `python scripts/diagnose_perpetuals.py`
- Risk report from events: `python scripts/generate_risk_report.py --days 1`

## Phased Rollout

1) Paper (dev): run with `--dry-run --dev-fast` and verify logs/metrics.
2) Canary: use the `canary` profile (reduce‑only, leverage capped, IOC) for a few minutes; monitor metrics and try the kill switch.
3) Tiny capital: enable live trading with trivial notional and strict `RISK_*` limits.
4) Scale gradually only after hitting predefined quality gates (profit factor, drawdown, error rates).

## Safety Reminders

- Keep `reduce_only_mode` on until you have verified exits and reconciliation.
- Always set a positive `RISK_DAILY_LOSS_LIMIT` before live trading.
- Prefer post‑only limit orders while validating execution.
- Monitor funding impact; avoid holding through extreme funding unless explicitly intended.

See also: `scripts/perps_real_broker_checklist.sh` and `scripts/production_readiness_integration.py` for deeper end‑to‑end checks.

