Perpetuals Trading – Deployment Runbook
======================================

This runbook documents how to configure, validate, and operate the simplified Coinbase perpetuals trading system. It favors reliability and clarity over complexity.

1) Quick Start Guide
--------------------

- Paper Mode (safe, no credentials)
  - Set these in `.env` (or shell):
    - `BROKER=coinbase`
    - `PERPS_PAPER=1`
    - `LOG_LEVEL=INFO`
  - Validate end‑to‑end (no network):
    - `PYTHONPATH=src python scripts/validation/strategy_smoke_test.py`
    - `PYTHONPATH=src python scripts/validation/risk_smoke_test.py`
    - `PYTHONPATH=src python scripts/validation/paper_mode_e2e.py`

- Production Mode (live, JWT based)
  - Set these 12 variables (see section 2):
    - `BROKER=coinbase`
    - `PERPS_PAPER=0`
    - `LOG_LEVEL=INFO`
    - `COINBASE_API_MODE=advanced`
    - `COINBASE_SANDBOX=0`
    - `COINBASE_ENABLE_DERIVATIVES=1`
    - `COINBASE_PROD_CDP_API_KEY=...`
    - `COINBASE_PROD_CDP_PRIVATE_KEY=... (PEM)`
    - `RISK_MAX_LEVERAGE=5`
    - `RISK_MAX_POSITION_PCT_PER_SYMBOL=0.25`
    - `RISK_DAILY_LOSS_LIMIT=100`
    - `RISK_MAX_MARK_STALENESS_SECONDS=180`
  - Validate before trading:
    - `PYTHONPATH=src python scripts/validation/validate_cdp_jwt.py`
    - `PYTHONPATH=src python scripts/validation/validate_coinbase_connectivity.py`

2) Environment Configuration
----------------------------

- Essential Variables (12)
  - `BROKER`: Must be `coinbase`.
  - `PERPS_PAPER`: `1` to run mock/paper mode with no credentials; `0` for live.
  - `LOG_LEVEL`: `DEBUG`|`INFO`|`WARNING`|`ERROR`.
  - `COINBASE_API_MODE`: Must be `advanced` for perps.
  - `COINBASE_SANDBOX`: Must be `0` for perps (sandbox doesn’t support Advanced Trade derivatives).
  - `COINBASE_ENABLE_DERIVATIVES`: `1` to enable CFM endpoints.
  - `COINBASE_PROD_CDP_API_KEY`: Your CDP API key name.
  - `COINBASE_PROD_CDP_PRIVATE_KEY`: EC private key (PEM string with header/footer).
  - `RISK_MAX_LEVERAGE`: Global leverage cap (default `5`).
  - `RISK_MAX_POSITION_PCT_PER_SYMBOL`: Per‑symbol notional cap as fraction of equity (default `0.25`).
  - `RISK_DAILY_LOSS_LIMIT`: Max daily loss in USD before reduce‑only is enabled (default `100`).
  - `RISK_MAX_MARK_STALENESS_SECONDS`: Soft warning threshold for mark age; halt at 2x (default `180`).

- JWT Setup Process
  - Create a project and API key in the Coinbase Developer Platform (CDP).
  - Download/generate the EC private key (PEM). Keep it secure.
  - Ensure derivatives access is enabled on the account.
  - Set `COINBASE_PROD_CDP_API_KEY` to the key name; set `COINBASE_PROD_CDP_PRIVATE_KEY` to the PEM.
  - Verify with: `PYTHONPATH=src python scripts/validation/validate_cdp_jwt.py`.

- Common Configurations
  - Dev (local, no creds): `PERPS_PAPER=1`.
  - Staging (dry live connectivity): `PERPS_PAPER=0` + live JWT + set `RISK_DAILY_LOSS_LIMIT` low and optionally `RISK_REDUCE_ONLY_MODE=1`.
  - Production: As above with operational risk limits as desired.

> See `.env.template` for a minimal, documented template you can copy to `.env`.

3) Operation Guide
-------------------

- Running the Bot
  - Paper mode E2E demo: `PYTHONPATH=src python scripts/validation/paper_mode_e2e.py`
  - For a long‑running process, create a small runner that instantiates `PerpsBot` with `BotConfig.from_profile('prod', symbols=[...])` and calls `await bot.run()` in an asyncio loop.

- Monitoring Logs
  - Look for:
    - `Decision: buy/sell/close - …`
    - `Order recorded: …`
    - `Order rejected: … reason=…`
    - `Position changes detected: …`
    - `Risk snapshot: equity=… exposure=… reduce_only=…`
  - Adjust verbosity with `LOG_LEVEL`.

- Troubleshooting
  - “Perpetuals require Advanced Trade …”: Ensure `COINBASE_API_MODE=advanced`, `COINBASE_SANDBOX=0`.
  - “Missing CDP JWT credentials …”: Set both `COINBASE_PROD_CDP_API_KEY` and `COINBASE_PROD_CDP_PRIVATE_KEY`.
  - “COINBASE_ENABLE_DERIVATIVES=1 is required …”: Set that variable.
  - Frequent order rejections: Check min size/notional and `RISK_MAX_POSITION_PCT_PER_SYMBOL`.
  - Stale marks: Increase `RISK_MAX_MARK_STALENESS_SECONDS` or verify `get_quote()` availability.
  - 429s/rate limits: Slow the loop (`update_interval`) or back off frequency.

4) Validation Workflow
----------------------

- Paper Mode Testing (no network/creds)
  - Strategy: `PYTHONPATH=src python scripts/validation/strategy_smoke_test.py`
  - Risk: `PYTHONPATH=src python scripts/validation/risk_smoke_test.py`
  - End‑to‑End: `PYTHONPATH=src python scripts/validation/paper_mode_e2e.py`

- Production Readiness
  - JWT: `PYTHONPATH=src python scripts/validation/validate_cdp_jwt.py`
  - Connectivity: `PYTHONPATH=src python scripts/validation/validate_coinbase_connectivity.py`

- Health Monitoring
  - The bot writes a health file per profile: `data/perps_bot/<profile>/health.json`.
  - It is updated each cycle with `{ ok, timestamp, message, error }`.

Notes
-----

- Market data is REST‑first (no WebSockets). The bot polls `get_quote()` each cycle.
- Positions/PnL are exchange‑authoritative (no local PnL cache).
- Configuration is intentionally minimal to reduce failure modes.

