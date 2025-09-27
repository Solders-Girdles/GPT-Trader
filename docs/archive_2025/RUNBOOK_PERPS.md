# Perpetuals Runbook

Step‑by‑step to configure, validate, start, monitor, and stop the perps bot.

## 1) Install & Setup
- Install deps: `poetry install` or `pip install -e .[dev]`
- Copy `.env.template` → `.env` and fill required fields (see below)
- Ensure `BROKER=coinbase`

Supported instruments (Coinbase perps): BTC-PERP, ETH-PERP, SOL-PERP, XRP-PERP.
The bot filters any other symbols.

Required env for perps (production):
- `COINBASE_SANDBOX=0`
- `COINBASE_ENABLE_DERIVATIVES=1`
- `COINBASE_API_MODE=advanced`
- `COINBASE_PROD_CDP_API_KEY=...`
- `COINBASE_PROD_CDP_PRIVATE_KEY="""-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----"""`

Safety env (recommended for canary):
- `RISK_REDUCE_ONLY_MODE=1`
- `RISK_KILL_SWITCH_ENABLED=1` (enable, then disable just before supervised start)
- `RISK_MAX_LEVERAGE=1`, `RISK_MAX_EXPOSURE_PCT=0.01`, `RISK_DAILY_LOSS_LIMIT=10`

Optional:
- Per‑symbol caps: `RISK_LEVERAGE_MAX_PER_SYMBOL=BTC-PERP:1,ETH-PERP:1`
- Notional caps: `RISK_MAX_NOTIONAL_PER_SYMBOL=BTC-PERP:500,ETH-PERP:300`
- Tuning: `RISK_MAX_MARK_STALENESS_SECONDS`, slippage/funding, volatility CBs

Time‑of‑day leverage (10x only during select hours):
- Set `RISK_DAYTIME_START_UTC`/`RISK_DAYTIME_END_UTC` to the 10x window.
- Example caps: `RISK_DAY_LEVERAGE_MAX_PER_SYMBOL=BTC-PERP:10,ETH-PERP:10,SOL-PERP:10,XRP-PERP:10`
- Night caps: `RISK_NIGHT_LEVERAGE_MAX_PER_SYMBOL=BTC-PERP:5,ETH-PERP:5,SOL-PERP:5,XRP-PERP:5` (or stricter).

## 2) Quick Validations
- E2E wiring: `python scripts/validation/validate_perps_e2e.py`
- API enablement: `python scripts/validation/validate_perps_api_enablement.py` (non‑destructive)
- Risk smoke: `python scripts/validation/risk_smoke_test.py`
- Optional offline preflight: `python scripts/preflight/perps_preflight.py --offline`

## 3) Run Modes
- Dev mock (no orders):
  `python -m bot_v2.cli --profile dev --dry-run --symbols BTC-PERP`
- Canary (reduce‑only, tiny size):
  `python -m bot_v2.cli --profile canary --symbols BTC-PERP --reduce-only`
- Prod (after proving canary):
  `python -m bot_v2.cli --profile prod --symbols BTC-PERP ETH-PERP`

Notes
- For streaming marks in canary/prod, set `PERPS_ENABLE_STREAMING=1`
- Force IOC if desired in canary via `--tif IOC` (already defaulted)

## 4) Monitoring
- Logs: `logs/perps_trading.log`, JSONL at `logs/perps_trading.jsonl`
- Event store: `data/perps_bot/{profile}/events/*`
- Health: `data/perps_bot/{profile}/health.json`
- Risk metrics in event store under bot_id `risk_engine`

## 5) Operate Safely
- Kill switch: set `RISK_KILL_SWITCH_ENABLED=1` and restart to block all new orders
- Reduce‑only: `--reduce-only` flag or `RISK_REDUCE_ONLY_MODE=1`
- Cancel/flatten procedures: use broker UI if emergency; bot will reconcile on restart

## 6) Troubleshooting
- Stale marks: verify streaming or REST fallback; adjust `RISK_MAX_MARK_STALENESS_SECONDS`
- Order rejects: check quantization (tick/min size), reduce‑only blocks, leverage caps
- Connectivity: confirm `COINBASE_API_MODE`, JWT keys, and that derivatives are enabled

Non‑advice disclaimer: Perpetual futures are high risk; you may lose all posted collateral.
