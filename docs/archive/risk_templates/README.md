Risk Templates (Reference)

Overview
- Files in this directory are YAML reference templates for risk limits (legacy shape).
- Templates moved from `config/risk/` to keep runtime config separate from examples.
- The runtime does **not** load `RISK_CONFIG_PATH` by default. Active risk settings come from
  profile YAML (`config/profiles/*.yaml`) + env (`RISK_*`) via `BotConfig` and
  `LiveRiskManager`.

How To Use (Manual Wiring)
- Parse a template (YAML) and map the values into `RiskConfig` or `BotRiskConfig`.
- Provide the mapped config by extending `ApplicationContainer` or
  `RiskValidationContainer` in your deployment.

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
- Exchange rules (MMR, leverage) can change; treat these caps as upper bounds.
- There is no built-in Pydantic loader for these templates in v3.0. Use
  `features/live_trade/risk/config.py` as the canonical model when wiring manually.

Related Env
- `FEE_BPS_BY_SYMBOL` lets sizing tools use per‑symbol fee bps (e.g., `BTC-PERP:6,ETH-PERP:8`).
