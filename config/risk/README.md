Coinbase Perps Risk Config

Overview
- File `coinbase_perps.prod.yaml` provides production‑oriented risk limits for BTC/ETH/SOL/XRP.
- The bot loads this file when env var `RISK_CONFIG_PATH` points to it; otherwise it falls back to env defaults.

How To Use
- Set env: `RISK_CONFIG_PATH=config/risk/coinbase_perps.prod.yaml`
- Optionally set symbols via env: `TRADING_SYMBOLS=BTC-PERP,ETH-PERP,SOL-PERP,XRP-PERP`
- Or pass on CLI: `poetry run coinbase-trader --profile prod --symbols BTC-PERP ETH-PERP SOL-PERP XRP-PERP`

Key Settings
- max_leverage: Global cap; day/night per‑symbol caps override when stricter.
- day/night window (UTC): `daytime_start_utc`, `daytime_end_utc`.
- per‑symbol leverage caps: `day_leverage_max_per_symbol`, `night_leverage_max_per_symbol`.
- exposure caps: `max_exposure_pct` (portfolio) and `max_position_pct_per_symbol`.
- notional caps: `max_notional_per_symbol` hard per‑symbol ceilings (USD).
- liquidation safety: `min_liquidation_buffer_pct` with projection enabled.

Adjusting Leverage Hours
- If 10x is only available during certain hours, set `daytime_start_utc`/`daytime_end_utc` accordingly and put 10 for BTC, 8 for ETH (example) in `day_leverage_max_per_symbol`.
- Keep night caps conservative (e.g., 5x BTC/ETH, 3x SOL/XRP) until confident.

Suggested First‑Run Values
- daily_loss_limit: 50–100 USD while validating stability.
- slippage_guard_bps: 30 (0.30%). Tighten if you see rejections.
- max_notional_per_symbol: tune after observing fill quality and funding.

Notes
- YAML supports inline comments—capture rationale directly in the file and keep this history updated.
- Exchange rules (MMR, leverage) can change; these caps are your upper bounds.
- The loader validates input with a shared Pydantic schema: env errors raise `EnvVarError` with the offending key (e.g. `RISK_MAX_EXPOSURE_PCT`), and JSON parsing enforces percentage bounds plus mapping formats. Legacy aliases like `RISK_MAX_TOTAL_EXPOSURE_PCT` still map to `max_exposure_pct`.

Related Env
- `FEE_BPS_BY_SYMBOL` lets sizing tools use per‑symbol fee bps (e.g., `BTC-PERP:6,ETH-PERP:8`).
